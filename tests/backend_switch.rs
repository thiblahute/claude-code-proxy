//! Backend switching: control API guards, Claude passthrough, and 429 failover.
//! Everything runs against in-process apps and a loopback mock; no real network.

use async_trait::async_trait;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::ConnectInfo;
use axum::http::{HeaderMap, Method, Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use claude_code_proxy::{
    MessagesRequest,
    backend::{Backend, BackendOrder, BackendSettings, ClaudeUpstream},
    config::AliasProvider,
    provider::{CliHandlers, Provider, RequestContext},
    registry::Registry,
    request_identity::ConversationIdentity,
    server::{AppFeatures, app_with_backend_settings},
};
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tower::util::ServiceExt;

const LOCAL: &str = "127.0.0.1:50000";
const REMOTE: &str = "203.0.113.7:50000";
const MESSAGES_BODY: &str = r#"{ "model":"gpt-5.5",  "max_tokens": 16, "stream": true,
  "messages": [{"role":"user","content":"hi"}] }"#;

struct NoCli;

static NO_CLI: NoCli = NoCli;

impl CliHandlers for NoCli {
    fn login(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn device(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn status(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn logout(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Stands in for the Codex backend: answers every request with `status`.
struct FixedCodex {
    status: StatusCode,
}

#[async_trait]
impl Provider for FixedCodex {
    fn name(&self) -> &'static str {
        "codex"
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["gpt-5.5".to_string()]
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &NO_CLI
    }

    async fn handle_messages(&self, _body: MessagesRequest, _ctx: RequestContext) -> Response {
        self.respond()
    }

    async fn handle_messages_with_conversation_identity(
        &self,
        _body: MessagesRequest,
        _ctx: RequestContext,
        _conversation_identity: Option<ConversationIdentity>,
    ) -> Response {
        self.respond()
    }

    async fn handle_count_tokens(&self, _body: MessagesRequest, _ctx: RequestContext) -> Response {
        self.respond()
    }
}

impl FixedCodex {
    fn respond(&self) -> Response {
        if self.status == StatusCode::TOO_MANY_REQUESTS {
            return (
                self.status,
                [(header::RETRY_AFTER, "30")],
                "codex rate limited",
            )
                .into_response();
        }
        (self.status, "codex-ok").into_response()
    }
}

fn app(order: Vec<Backend>, codex_status: StatusCode, claude_base: Option<&str>) -> Router {
    let registry = Arc::new(Registry::from_providers(
        AliasProvider::Codex,
        [Arc::new(FixedCodex {
            status: codex_status,
        }) as Arc<dyn Provider>],
    ));
    let settings = BackendSettings {
        order: BackendOrder::new(order).unwrap(),
        upstream: ClaudeUpstream::from_base_url(claude_base).unwrap(),
    };
    app_with_backend_settings(registry, None, AppFeatures::default(), settings)
}

fn with_peer(mut request: Request<Body>, peer: &str) -> Request<Body> {
    let addr: SocketAddr = peer.parse().unwrap();
    request.extensions_mut().insert(ConnectInfo(addr));
    request
}

fn control(method: Method, path: &str, body: &str, content_type: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().method(method).uri(path);
    if let Some(content_type) = content_type {
        builder = builder.header(header::CONTENT_TYPE, content_type);
    }
    with_peer(builder.body(Body::from(body.to_string())).unwrap(), LOCAL)
}

async fn body_json(response: Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn body_text(response: Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[derive(Clone, Debug)]
struct Recorded {
    path_and_query: String,
    headers: HeaderMap,
    body: Bytes,
}

/// Loopback stand-in for api.anthropic.com. Records every request it gets.
async fn start_mock() -> (String, Arc<Mutex<Vec<Recorded>>>) {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = recorded.clone();
    let mock = Router::new().fallback(move |request: Request<Body>| {
        let sink = sink.clone();
        async move {
            let (parts, body) = request.into_parts();
            let body = axum::body::to_bytes(body, usize::MAX).await.unwrap();
            sink.lock().unwrap().push(Recorded {
                path_and_query: parts
                    .uri
                    .path_and_query()
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                headers: parts.headers,
                body,
            });
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
            )
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });
    (base, recorded)
}

fn messages_request(extra: &[(&str, &str)]) -> Request<Body> {
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages?beta=true")
        .header(header::CONTENT_TYPE, "application/json");
    for (name, value) in extra {
        builder = builder.header(*name, *value);
    }
    builder.body(Body::from(MESSAGES_BODY)).unwrap()
}

#[tokio::test]
async fn control_get_reports_order_state_and_no_store() {
    let response = app(vec![Backend::Codex, Backend::Claude], StatusCode::OK, None)
        .oneshot(control(Method::GET, "/control/backend", "", None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
    let body = body_json(response).await;
    assert_eq!(body["current"], "codex");
    assert_eq!(body["order"], json!(["codex", "claude"]));
    assert_eq!(body["last_switch_reason"], json!({"kind": "config"}));
    assert_eq!(body["next_check"], Value::Null);
    assert!(
        body["since"]
            .as_str()
            .is_some_and(|since| !since.is_empty())
    );
}

#[tokio::test]
async fn control_rejects_non_loopback_peers() {
    let router = app(vec![Backend::Codex, Backend::Claude], StatusCode::OK, None);
    let get = with_peer(
        Request::builder()
            .uri("/control/backend")
            .body(Body::empty())
            .unwrap(),
        REMOTE,
    );
    let response = router.clone().oneshot(get).await.unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);

    let post = with_peer(
        Request::builder()
            .method(Method::POST)
            .uri("/control/backend")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"backend":"claude"}"#))
            .unwrap(),
        REMOTE,
    );
    let response = router.oneshot(post).await.unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn control_rejects_requests_without_connect_info() {
    let response = app(vec![Backend::Codex], StatusCode::OK, None)
        .oneshot(
            Request::builder()
                .uri("/control/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn control_rejects_origin_header() {
    let request = with_peer(
        Request::builder()
            .uri("/control/backend")
            .header(header::ORIGIN, "https://example.test")
            .body(Body::empty())
            .unwrap(),
        LOCAL,
    );
    let response = app(vec![Backend::Codex], StatusCode::OK, None)
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn control_post_requires_json_content_type() {
    let router = app(vec![Backend::Codex, Backend::Claude], StatusCode::OK, None);
    let text = control(
        Method::POST,
        "/control/backend",
        r#"{"backend":"claude"}"#,
        Some("text/plain"),
    );
    let response = router.clone().oneshot(text).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);

    let missing = with_peer(
        Request::builder()
            .method(Method::POST)
            .uri("/control/backend")
            .body(Body::from(r#"{"backend":"claude"}"#))
            .unwrap(),
        LOCAL,
    );
    let response = router.clone().oneshot(missing).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);

    let with_charset = control(
        Method::POST,
        "/control/backend",
        r#"{"backend":"claude"}"#,
        Some("application/json; charset=utf-8"),
    );
    let response = router.oneshot(with_charset).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn control_post_rejects_unknown_and_unconfigured_backends() {
    let router = app(vec![Backend::Codex], StatusCode::OK, None);
    for body in [r#"{"backend":"kimi"}"#, r#"{"backend":"claude"}"#, "{}"] {
        let response = router
            .clone()
            .oneshot(control(
                Method::POST,
                "/control/backend",
                body,
                Some("application/json"),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "body {body}");
    }
}

#[tokio::test]
async fn control_manual_switch_changes_state() {
    let router = app(vec![Backend::Codex, Backend::Claude], StatusCode::OK, None);
    let response = router
        .clone()
        .oneshot(control(
            Method::POST,
            "/control/backend",
            r#"{"backend":"claude"}"#,
            Some("application/json"),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_json(response).await["current"], "claude");

    let response = router
        .oneshot(control(Method::GET, "/control/backend", "", None))
        .await
        .unwrap();
    let body = body_json(response).await;
    assert_eq!(body["current"], "claude");
    assert_eq!(body["last_switch_reason"], json!({"kind": "manual"}));
}

#[tokio::test]
async fn control_unknown_session_is_not_found() {
    let router = app(vec![Backend::Codex], StatusCode::OK, None);
    let response = router
        .clone()
        .oneshot(control(Method::GET, "/control/sessions/nope", "", None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");

    let response = router
        .oneshot(control(Method::GET, "/control/sessions", "", None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_json(response).await["sessions"], json!([]));
}

#[tokio::test]
async fn default_order_keeps_the_codex_path() {
    let registry = Arc::new(Registry::from_providers(
        AliasProvider::Codex,
        [Arc::new(FixedCodex {
            status: StatusCode::OK,
        }) as Arc<dyn Provider>],
    ));
    let router = app_with_backend_settings(
        registry,
        None,
        AppFeatures::default(),
        BackendSettings::defaults(),
    );
    let response = router
        .clone()
        .oneshot(control(Method::GET, "/control/backend", "", None))
        .await
        .unwrap();
    assert_eq!(body_json(response).await["current"], "codex");

    let response = router.oneshot(messages_request(&[])).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_text(response).await, "codex-ok");
}

#[tokio::test]
async fn claude_passthrough_forwards_body_bytes_and_allowed_headers() {
    let (base, recorded) = start_mock().await;
    let router = app(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::OK,
        Some(&base),
    );
    let switched = router
        .clone()
        .oneshot(control(
            Method::POST,
            "/control/backend",
            r#"{"backend":"claude"}"#,
            Some("application/json"),
        ))
        .await
        .unwrap();
    assert_eq!(switched.status(), StatusCode::OK);

    let response = router
        .clone()
        .oneshot(messages_request(&[
            ("authorization", "Bearer test-token"),
            ("anthropic-beta", "tools-2025-01-01"),
            ("anthropic-version", "2023-06-01"),
            ("x-stainless-os", "Linux"),
            ("cookie", "private=1"),
            ("x-claude-code-session-id", "sess-claude"),
        ]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    assert!(response.headers().contains_key("request-id"));
    assert!(body_text(response).await.contains("message_start"));

    let seen = recorded.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    let upstream = &seen[0];
    assert_eq!(upstream.path_and_query, "/v1/messages?beta=true");
    assert_eq!(upstream.body.as_ref(), MESSAGES_BODY.as_bytes());
    assert_eq!(upstream.headers["authorization"], "Bearer test-token");
    assert_eq!(upstream.headers["anthropic-beta"], "tools-2025-01-01");
    assert_eq!(upstream.headers["x-stainless-os"], "Linux");
    assert!(!upstream.headers.contains_key(header::COOKIE));
    assert!(!upstream.headers.contains_key("x-claude-code-session-id"));

    let session = router
        .oneshot(control(
            Method::GET,
            "/control/sessions/sess-claude",
            "",
            None,
        ))
        .await
        .unwrap();
    let session = body_json(session).await;
    assert_eq!(session["backend"], "claude");
    assert_eq!(session["model"], "gpt-5.5");
}

#[tokio::test]
async fn claude_passthrough_requires_authorization() {
    let (base, recorded) = start_mock().await;
    let router = app(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::OK,
        Some(&base),
    );
    router
        .clone()
        .oneshot(control(
            Method::POST,
            "/control/backend",
            r#"{"backend":"claude"}"#,
            Some("application/json"),
        ))
        .await
        .unwrap();
    let response = router.oneshot(messages_request(&[])).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(recorded.lock().unwrap().is_empty());
}

#[tokio::test]
async fn codex_429_fails_over_to_claude_with_retry_after() {
    let (base, recorded) = start_mock().await;
    let router = app(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::TOO_MANY_REQUESTS,
        Some(&base),
    );
    let response = router
        .clone()
        .oneshot(messages_request(&[
            ("authorization", "Bearer test-token"),
            ("x-claude-code-session-id", "sess-429"),
        ]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(body_text(response).await.contains("message_start"));

    let seen = recorded.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].body.as_ref(), MESSAGES_BODY.as_bytes());

    let state = body_json(
        router
            .clone()
            .oneshot(control(Method::GET, "/control/backend", "", None))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(state["current"], "claude");
    assert_eq!(
        state["last_switch_reason"],
        json!({"kind": "rate_limit_429", "from": "codex", "retry_after": "30"})
    );
    let session = body_json(
        router
            .oneshot(control(Method::GET, "/control/sessions/sess-429", "", None))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(session["backend"], "claude");
}

#[tokio::test]
async fn codex_429_without_claude_in_order_is_returned_unchanged() {
    let router = app(vec![Backend::Codex], StatusCode::TOO_MANY_REQUESTS, None);
    let response = router
        .clone()
        .oneshot(messages_request(&[("authorization", "Bearer test-token")]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()[header::RETRY_AFTER], "30");

    let state = body_json(
        router
            .oneshot(control(Method::GET, "/control/backend", "", None))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(state["current"], "codex");
    assert_eq!(state["last_switch_reason"], json!({"kind": "config"}));
}

#[tokio::test]
async fn codex_other_errors_do_not_switch() {
    let router = app(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::INTERNAL_SERVER_ERROR,
        Some("https://api.anthropic.com"),
    );
    let response = router
        .clone()
        .oneshot(messages_request(&[("authorization", "Bearer test-token")]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let state = body_json(
        router
            .oneshot(control(Method::GET, "/control/backend", "", None))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(state["current"], "codex");
}
