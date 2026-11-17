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
    backend::{Backend, BackendOrder, BackendSettings, ClaudeUpstream, FailoverPolicy},
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
        vec!["gpt-5.5".to_string(), "claude-opus-4-1".to_string()]
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
    app_with_failover(order, codex_status, claude_base, FailoverPolicy::default())
}

fn app_with_failover(
    order: Vec<Backend>,
    codex_status: StatusCode,
    claude_base: Option<&str>,
    failover: FailoverPolicy,
) -> Router {
    let registry = Arc::new(Registry::from_providers(
        AliasProvider::Codex,
        [Arc::new(FixedCodex {
            status: codex_status,
        }) as Arc<dyn Provider>],
    ));
    let settings = BackendSettings {
        order: BackendOrder::new(order).unwrap(),
        failover,
        upstream: ClaudeUpstream::from_base_url(claude_base).unwrap(),
        upstreams: Default::default(),
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

const CLAUDE_MODEL_BODY: &str = r#"{"model":"claude-sonnet-4-5[1m]","max_tokens":16,"stream":false,
  "system":[{"type":"text","text":"You are a security monitor for autonomous AI coding agents. Decide."}],
  "messages": [{"role":"user","content":"check this command"}]}"#;

fn post_to(path: &str, body: &str, extra: &[(&str, &str)]) -> Request<Body> {
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json");
    for (name, value) in extra {
        builder = builder.header(*name, *value);
    }
    builder.body(Body::from(body.to_string())).unwrap()
}

async fn switch_to(router: &Router, backend: &str) -> StatusCode {
    router
        .clone()
        .oneshot(control(
            Method::POST,
            "/control/backend",
            &format!(r#"{{"backend":"{backend}"}}"#),
            Some("application/json"),
        ))
        .await
        .unwrap()
        .status()
}

async fn set_allowed(router: &Router, session: &str, body: &str) -> Response {
    router
        .clone()
        .oneshot(control(
            Method::POST,
            &format!("/control/sessions/{session}/allowed"),
            body,
            Some("application/json"),
        ))
        .await
        .unwrap()
}

#[tokio::test]
async fn claude_passthrough_forwards_x_api_key() {
    let (base, recorded) = start_mock().await;
    let router = app(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::OK,
        Some(&base),
    );
    assert_eq!(switch_to(&router, "claude").await, StatusCode::OK);

    let response = router
        .clone()
        .oneshot(messages_request(&[("x-api-key", "sk-test-key")]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(body_text(response).await.contains("message_start"));

    let seen = recorded.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].headers["x-api-key"], "sk-test-key");
    assert!(!seen[0].headers.contains_key(header::AUTHORIZATION));
}

#[tokio::test]
async fn count_tokens_on_claude_goes_to_the_count_tokens_endpoint() {
    let (base, recorded) = start_mock().await;
    let router = app(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::OK,
        Some(&base),
    );
    assert_eq!(switch_to(&router, "claude").await, StatusCode::OK);

    let request = post_to(
        "/v1/messages/count_tokens?beta=true",
        MESSAGES_BODY,
        &[
            ("authorization", "Bearer test-token"),
            ("cookie", "private=1"),
        ],
    );
    let response = router.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let seen = recorded.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(
        seen[0].path_and_query,
        "/v1/messages/count_tokens?beta=true"
    );
    assert_eq!(seen[0].body.as_ref(), MESSAGES_BODY.as_bytes());
    assert_eq!(seen[0].headers["authorization"], "Bearer test-token");
    assert!(!seen[0].headers.contains_key(header::COOKIE));
}

#[tokio::test]
async fn count_tokens_on_codex_stays_on_the_codex_path() {
    let (base, recorded) = start_mock().await;
    let router = app(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::OK,
        Some(&base),
    );
    let response = router
        .oneshot(post_to(
            "/v1/messages/count_tokens",
            MESSAGES_BODY,
            &[("authorization", "Bearer test-token")],
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_text(response).await, "codex-ok");
    assert!(recorded.lock().unwrap().is_empty());
}

#[tokio::test]
async fn claude_path_sends_claude_model_body_unchanged() {
    let (base, recorded) = start_mock().await;
    let router = app(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::OK,
        Some(&base),
    );
    assert_eq!(switch_to(&router, "claude").await, StatusCode::OK);

    let response = router
        .oneshot(post_to(
            "/v1/messages",
            CLAUDE_MODEL_BODY,
            &[("authorization", "Bearer test-token")],
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let seen = recorded.lock().unwrap().clone();
    assert_eq!(seen.len(), 1, "no auto-review rewrite, one upstream call");
    assert_eq!(seen[0].body.as_ref(), CLAUDE_MODEL_BODY.as_bytes());
    let forwarded: Value = serde_json::from_slice(&seen[0].body).unwrap();
    assert_eq!(forwarded["model"], "claude-sonnet-4-5[1m]");
}

#[tokio::test]
async fn session_allowed_endpoint_sets_and_reports_allowed_backends() {
    let router = app(vec![Backend::Codex, Backend::Claude], StatusCode::OK, None);
    let response = router
        .clone()
        .oneshot(control(
            Method::POST,
            "/control/sessions/ghost/allowed",
            r#"{"backends":["claude"]}"#,
            Some("application/json"),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    router
        .clone()
        .oneshot(messages_request(&[(
            "x-claude-code-session-id",
            "sess-allowed",
        )]))
        .await
        .unwrap();
    let session = router
        .clone()
        .oneshot(control(
            Method::GET,
            "/control/sessions/sess-allowed",
            "",
            None,
        ))
        .await
        .unwrap();
    assert_eq!(
        body_json(session).await["allowed"],
        json!(["codex", "claude"])
    );

    let response = set_allowed(&router, "sess-allowed", r#"{"backends":["claude"]}"#).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
    let body = body_json(response).await;
    assert_eq!(body["id"], "sess-allowed");
    assert_eq!(body["allowed"], json!(["claude"]));
    assert_eq!(body["backend"], "codex");

    let session = router
        .clone()
        .oneshot(control(
            Method::GET,
            "/control/sessions/sess-allowed",
            "",
            None,
        ))
        .await
        .unwrap();
    assert_eq!(body_json(session).await["allowed"], json!(["claude"]));
    let list = router
        .clone()
        .oneshot(control(Method::GET, "/control/sessions", "", None))
        .await
        .unwrap();
    let list = body_json(list).await;
    assert_eq!(list["sessions"][0]["allowed"], json!(["claude"]));

    for body in [
        r#"{"backends":[]}"#,
        r#"{"backends":["kimi"]}"#,
        r#"{"backends":["claude","claude"]}"#,
        r#"{"backends":"claude"}"#,
        r#"{"backends":[1]}"#,
        "{}",
    ] {
        let response = set_allowed(&router, "sess-allowed", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "body {body}");
    }

    let codex_only = app(vec![Backend::Codex], StatusCode::OK, None);
    codex_only
        .clone()
        .oneshot(messages_request(&[(
            "x-claude-code-session-id",
            "sess-codex",
        )]))
        .await
        .unwrap();
    let response = set_allowed(&codex_only, "sess-codex", r#"{"backends":["claude"]}"#).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let text = control(
        Method::POST,
        "/control/sessions/sess-allowed/allowed",
        r#"{"backends":["claude"]}"#,
        Some("text/plain"),
    );
    let response = router.clone().oneshot(text).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);

    let remote = with_peer(
        Request::builder()
            .method(Method::POST)
            .uri("/control/sessions/sess-allowed/allowed")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"backends":["claude"]}"#))
            .unwrap(),
        REMOTE,
    );
    let response = router.clone().oneshot(remote).await.unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);

    let with_origin = with_peer(
        Request::builder()
            .method(Method::POST)
            .uri("/control/sessions/sess-allowed/allowed")
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::ORIGIN, "https://example.test")
            .body(Body::from(r#"{"backends":["claude"]}"#))
            .unwrap(),
        LOCAL,
    );
    let response = router.oneshot(with_origin).await.unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn codex_path_refuses_a_session_that_may_not_use_codex() {
    let (base, recorded) = start_mock().await;
    let router = app(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::OK,
        Some(&base),
    );
    router
        .clone()
        .oneshot(messages_request(&[(
            "x-claude-code-session-id",
            "sess-claude-only",
        )]))
        .await
        .unwrap();
    assert_eq!(
        set_allowed(&router, "sess-claude-only", r#"{"backends":["claude"]}"#)
            .await
            .status(),
        StatusCode::OK
    );

    let claude_model = MESSAGES_BODY.replace("gpt-5.5", "claude-opus-4-1");
    let (status, body) = {
        let response = router
            .clone()
            .oneshot(post_to(
                "/v1/messages",
                &claude_model,
                &[
                    ("authorization", "Bearer test-token"),
                    ("x-claude-code-session-id", "sess-claude-only"),
                ],
            ))
            .await
            .unwrap();
        (response.status(), body_json(response).await)
    };
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["type"], "error");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("claude-opus-4-1"), "{message}");
    assert!(
        message.contains("backend not allowed for this session"),
        "{message}"
    );
    assert!(recorded.lock().unwrap().is_empty());
}

#[tokio::test]
async fn claude_path_refuses_a_session_that_may_not_use_claude() {
    let (base, recorded) = start_mock().await;
    let router = app(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::OK,
        Some(&base),
    );
    router
        .clone()
        .oneshot(messages_request(&[(
            "x-claude-code-session-id",
            "sess-codex-only",
        )]))
        .await
        .unwrap();
    assert_eq!(switch_to(&router, "claude").await, StatusCode::OK);
    // Restricted after the switch: a restriction made first would block it.
    set_allowed(&router, "sess-codex-only", r#"{"backends":["codex"]}"#).await;

    let response = router
        .oneshot(messages_request(&[
            ("authorization", "Bearer test-token"),
            ("x-claude-code-session-id", "sess-codex-only"),
        ]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert!(recorded.lock().unwrap().is_empty());
}

#[tokio::test]
async fn claude_only_session_counts_tokens_on_claude_while_global_is_codex() {
    let (base, recorded) = start_mock().await;
    let router = app(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::OK,
        Some(&base),
    );
    router
        .clone()
        .oneshot(messages_request(&[(
            "x-claude-code-session-id",
            "sess-count",
        )]))
        .await
        .unwrap();
    set_allowed(&router, "sess-count", r#"{"backends":["claude"]}"#).await;

    let response = router
        .clone()
        .oneshot(post_to(
            "/v1/messages/count_tokens",
            MESSAGES_BODY,
            &[
                ("authorization", "Bearer test-token"),
                ("x-claude-code-session-id", "sess-count"),
            ],
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(recorded.lock().unwrap().len(), 1);
    assert_eq!(
        recorded.lock().unwrap()[0].path_and_query,
        "/v1/messages/count_tokens"
    );
}

#[tokio::test]
async fn codex_429_does_not_fail_over_for_a_session_that_may_not_use_claude() {
    let (base, recorded) = start_mock().await;
    let router = app(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::TOO_MANY_REQUESTS,
        Some(&base),
    );
    router
        .clone()
        .oneshot(messages_request(&[(
            "x-claude-code-session-id",
            "sess-429-codex",
        )]))
        .await
        .unwrap();
    set_allowed(&router, "sess-429-codex", r#"{"backends":["codex"]}"#).await;

    let response = router
        .clone()
        .oneshot(messages_request(&[
            ("authorization", "Bearer test-token"),
            ("x-claude-code-session-id", "sess-429-codex"),
        ]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()[header::RETRY_AFTER], "30");
    assert!(recorded.lock().unwrap().is_empty());
    let state = body_json(
        router
            .oneshot(control(Method::GET, "/control/backend", "", None))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(state["current"], "codex");
}

#[tokio::test]
async fn codex_429_is_returned_unchanged_when_failover_is_disabled() {
    let (base, recorded) = start_mock().await;
    let router = app_with_failover(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::TOO_MANY_REQUESTS,
        Some(&base),
        FailoverPolicy {
            on_429: false,
            retry_codex_after_secs: None,
        },
    );
    let response = router
        .clone()
        .oneshot(messages_request(&[("authorization", "Bearer test-token")]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()[header::RETRY_AFTER], "30");
    assert!(recorded.lock().unwrap().is_empty());
    let state = body_json(
        router
            .oneshot(control(Method::GET, "/control/backend", "", None))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(state["current"], "codex");
    assert_eq!(state["next_check"], Value::Null);
}

#[tokio::test]
async fn codex_429_failover_records_next_check_when_retry_is_configured() {
    let (base, _recorded) = start_mock().await;
    let router = app_with_failover(
        vec![Backend::Codex, Backend::Claude],
        StatusCode::TOO_MANY_REQUESTS,
        Some(&base),
        FailoverPolicy {
            on_429: true,
            retry_codex_after_secs: Some(600),
        },
    );
    let response = router
        .clone()
        .oneshot(messages_request(&[("authorization", "Bearer test-token")]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let state = body_json(
        router
            .oneshot(control(Method::GET, "/control/backend", "", None))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(state["current"], "claude");
    assert!(
        state["next_check"]
            .as_str()
            .is_some_and(|next| !next.is_empty())
    );
}

#[tokio::test]
async fn manual_switch_is_refused_when_a_session_may_not_use_the_target() {
    let router = app(vec![Backend::Codex, Backend::Claude], StatusCode::OK, None);
    router
        .clone()
        .oneshot(messages_request(&[(
            "x-claude-code-session-id",
            "sess-pinned",
        )]))
        .await
        .unwrap();
    set_allowed(&router, "sess-pinned", r#"{"backends":["codex"]}"#).await;

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
    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert!(
        body_json(response).await["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("sess-pinned"))
    );
    let state = body_json(
        router
            .oneshot(control(Method::GET, "/control/backend", "", None))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(state["current"], "codex");
}

/// Loopback stand-in for api.anthropic.com that answers every request with
/// a rate-limit 429 carrying Anthropic's limit headers.
async fn start_rate_limited_mock() -> String {
    let mock = Router::new().fallback(|| async {
        (
            StatusCode::TOO_MANY_REQUESTS,
            [
                (header::CONTENT_TYPE, "application/json"),
                (header::RETRY_AFTER, "42"),
                (
                    header::HeaderName::from_static("anthropic-ratelimit-unified-status"),
                    "rejected",
                ),
                (
                    header::HeaderName::from_static("anthropic-ratelimit-unified-reset"),
                    "1795000000",
                ),
                (header::SET_COOKIE, "upstream=1"),
            ],
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"limited"}}"#,
        )
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });
    base
}

#[tokio::test]
async fn claude_passthrough_returns_rate_limit_headers_verbatim() {
    let base = start_rate_limited_mock().await;
    let router = app(vec![Backend::Claude], StatusCode::OK, Some(&base));

    let response = router
        .oneshot(messages_request(&[("authorization", "Bearer test-token")]))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    let headers = response.headers().clone();
    assert_eq!(headers.get(header::RETRY_AFTER).unwrap(), "42");
    assert_eq!(
        headers.get("anthropic-ratelimit-unified-status").unwrap(),
        "rejected"
    );
    assert_eq!(
        headers.get("anthropic-ratelimit-unified-reset").unwrap(),
        "1795000000"
    );
    assert!(headers.get(header::SET_COOKIE).is_none());
    assert!(body_text(response).await.contains("rate_limit_error"));
}
