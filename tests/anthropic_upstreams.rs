//! Named Anthropic-compatible upstreams: model routing, auth replacement,
//! response model restoration, `/v1/models` listing and discovery, and session
//! records. Everything runs against in-process apps and loopback mocks; no real
//! network.

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
    upstream::{self, Upstreams},
};
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::util::ServiceExt;

const LOCAL: &str = "127.0.0.1:50000";
const UPSTREAM_KEY: &str = "upstream-key-123";
const CLIENT_TOKEN: &str = "Bearer client-oauth-token";
const CLIENT_API_KEY: &str = "client-api-key";

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

/// Stands in for the Codex backend; serves `gpt-5.5`.
struct FixedCodex;

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
        (StatusCode::OK, "codex-ok").into_response()
    }

    async fn handle_messages_with_conversation_identity(
        &self,
        _body: MessagesRequest,
        _ctx: RequestContext,
        _conversation_identity: Option<ConversationIdentity>,
    ) -> Response {
        (StatusCode::OK, "codex-ok").into_response()
    }

    async fn handle_count_tokens(&self, _body: MessagesRequest, _ctx: RequestContext) -> Response {
        (StatusCode::OK, "codex-ok").into_response()
    }
}

#[derive(Clone, Debug)]
struct Recorded {
    path_and_query: String,
    headers: HeaderMap,
    body: Bytes,
}

/// Loopback Anthropic-compatible gateway. Echoes the received model in its
/// answers, serves `/v1/models` (or 500 once `fail_models` is set), and records
/// every request.
struct Mock {
    base: String,
    recorded: Arc<Mutex<Vec<Recorded>>>,
    model_fetches: Arc<AtomicUsize>,
    fail_models: Arc<AtomicBool>,
}

impl Mock {
    fn messages(&self) -> Vec<Recorded> {
        self.recorded
            .lock()
            .unwrap()
            .iter()
            .filter(|request| request.path_and_query.starts_with("/v1/messages"))
            .cloned()
            .collect()
    }
}

async fn start_mock(listed: &'static [&'static str]) -> Mock {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let model_fetches = Arc::new(AtomicUsize::new(0));
    let fail_models = Arc::new(AtomicBool::new(false));
    let (sink, fetches, fail) = (recorded.clone(), model_fetches.clone(), fail_models.clone());
    let mock = Router::new().fallback(move |request: Request<Body>| {
        let (sink, fetches, fail) = (sink.clone(), fetches.clone(), fail.clone());
        async move {
            let (parts, body) = request.into_parts();
            let body = axum::body::to_bytes(body, usize::MAX).await.unwrap();
            let path = parts.uri.path().to_string();
            sink.lock().unwrap().push(Recorded {
                path_and_query: parts
                    .uri
                    .path_and_query()
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                headers: parts.headers,
                body: body.clone(),
            });
            if path == "/v1/models" {
                fetches.fetch_add(1, Ordering::SeqCst);
                if fail.load(Ordering::SeqCst) {
                    return (StatusCode::INTERNAL_SERVER_ERROR, "down").into_response();
                }
                let data: Vec<Value> = listed.iter().map(|id| json!({"id": id})).collect();
                return axum::Json(json!({"object": "list", "data": data})).into_response();
            }
            if path == "/v1/messages/count_tokens" {
                return axum::Json(json!({"input_tokens": 3})).into_response();
            }
            let request: Value = serde_json::from_slice(&body).unwrap();
            let model = request["model"].clone();
            if request["stream"] == json!(true) {
                let start = json!({
                    "type": "message_start",
                    "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": model},
                });
                let stream = format!(
                    "event: message_start\ndata: {start}\n\n\
                     event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"model\":{model}}}\n\n\
                     event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
                );
                return (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "text/event-stream")],
                    stream,
                )
                    .into_response();
            }
            axum::Json(json!({
                "id": "msg_1", "type": "message", "role": "assistant", "model": model,
                "content": [{"type": "text", "text": "hi"}],
            }))
            .into_response()
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });
    Mock {
        base,
        recorded,
        model_fetches,
        fail_models,
    }
}

/// Upstreams parsed from `entries`, with a key file holding `UPSTREAM_KEY`.
/// `{key}` in an entry's `api_key_file` is replaced by that file.
fn upstreams(dir: &tempfile::TempDir, entries: Value) -> Upstreams {
    let key = dir.path().join("key");
    std::fs::write(&key, format!("{UPSTREAM_KEY}\n")).unwrap();
    let raw = entries
        .to_string()
        .replace("{key}", &key.display().to_string());
    upstream::parse(&serde_json::from_str(&raw).unwrap(), "test upstreams").unwrap()
}

fn igalia(base: &str, models: &[&str], discover: bool) -> Value {
    json!([{
        "name": "igalia",
        "base_url": base,
        "api_key_file": "{key}",
        "models": models,
        "discover_models": discover,
    }])
}

fn app(upstreams: Upstreams, claude_base: Option<&str>) -> Router {
    let registry = Arc::new(Registry::from_providers(
        AliasProvider::Codex,
        [Arc::new(FixedCodex) as Arc<dyn Provider>],
    ));
    let settings = BackendSettings {
        order: BackendOrder::new(vec![Backend::Codex, Backend::Claude]).unwrap(),
        failover: FailoverPolicy::default(),
        upstream: ClaudeUpstream::from_base_url(claude_base).unwrap(),
        upstreams,
    };
    app_with_backend_settings(registry, None, AppFeatures::default(), settings)
}

fn post(path: &str, body: &str, extra: &[(&str, &str)]) -> Request<Body> {
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json");
    for (name, value) in extra {
        builder = builder.header(*name, *value);
    }
    builder.body(Body::from(body.to_string())).unwrap()
}

fn control(method: Method, path: &str, body: &str) -> Request<Body> {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let addr: SocketAddr = LOCAL.parse().unwrap();
    request.extensions_mut().insert(ConnectInfo(addr));
    request
}

fn messages_body(model: &str, stream: bool) -> String {
    format!(
        r#"{{"model":"{model}", "max_tokens": 16, "stream": {stream},
  "messages": [{{"role":"user","content":"hi"}}]}}"#
    )
}

async fn body_bytes(response: Response) -> Bytes {
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap()
}

async fn body_json(response: Response) -> Value {
    serde_json::from_slice(&body_bytes(response).await).unwrap()
}

async fn model_ids(router: &Router) -> Vec<String> {
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    body_json(response).await["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["id"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn prefixed_model_strips_the_prefix_and_restores_it_in_the_stream() {
    let mock = start_mock(&[]).await;
    let dir = tempfile::tempdir().unwrap();
    let router = app(upstreams(&dir, igalia(&mock.base, &[], false)), None);

    let sent = messages_body("igalia/glm-5p3", true);
    let response = router
        .oneshot(post("/v1/messages?beta=true", &sent, &[]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    let text = String::from_utf8(body_bytes(response).await.to_vec()).unwrap();

    let seen = mock.messages();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].path_and_query, "/v1/messages?beta=true");
    let forwarded: Value = serde_json::from_slice(&seen[0].body).unwrap();
    let mut expected: Value = serde_json::from_str(&sent).unwrap();
    expected["model"] = json!("glm-5p3");
    assert_eq!(forwarded, expected, "only the model field changes");

    let events: Vec<&str> = text.split("\n\n").filter(|e| !e.is_empty()).collect();
    assert_eq!(events.len(), 3, "{text}");
    let start: Value = serde_json::from_str(events[0].split_once("data: ").unwrap().1).unwrap();
    assert_eq!(start["type"], "message_start");
    assert_eq!(start["message"]["model"], "igalia/glm-5p3");
    assert_eq!(start["message"]["id"], "msg_1");
    assert_eq!(
        events[1],
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"model\":\"glm-5p3\"}",
        "other events pass through unchanged"
    );
    assert_eq!(
        events[2],
        "event: message_stop\ndata: {\"type\":\"message_stop\"}"
    );
}

#[tokio::test]
async fn prefixed_model_is_restored_in_a_json_response() {
    let mock = start_mock(&[]).await;
    let dir = tempfile::tempdir().unwrap();
    let router = app(upstreams(&dir, igalia(&mock.base, &[], false)), None);

    let response = router
        .oneshot(post(
            "/v1/messages",
            &messages_body("igalia/kimi-k3p1-pro", false),
            &[],
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["model"], "igalia/kimi-k3p1-pro");
    assert_eq!(body["content"][0]["text"], "hi");
    let forwarded: Value = serde_json::from_slice(&mock.messages()[0].body).unwrap();
    assert_eq!(forwarded["model"], "kimi-k3p1-pro");
}

#[tokio::test]
async fn bare_listed_model_is_forwarded_byte_for_byte() {
    let mock = start_mock(&[]).await;
    let dir = tempfile::tempdir().unwrap();
    let router = app(
        upstreams(&dir, igalia(&mock.base, &["glm-5p3"], false)),
        None,
    );

    let sent = messages_body("glm-5p3", false);
    let response = router
        .oneshot(post("/v1/messages", &sent, &[]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_json(response).await["model"], "glm-5p3");
    let seen = mock.messages();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].body.as_ref(), sent.as_bytes());
}

#[tokio::test]
async fn bare_name_served_by_a_proxy_provider_stays_with_the_provider() {
    let mock = start_mock(&[]).await;
    let dir = tempfile::tempdir().unwrap();
    let router = app(
        upstreams(&dir, igalia(&mock.base, &["gpt-5.5"], false)),
        None,
    );

    let response = router
        .clone()
        .oneshot(post("/v1/messages", &messages_body("gpt-5.5", false), &[]))
        .await
        .unwrap();
    assert_eq!(body_bytes(response).await.as_ref(), b"codex-ok");
    assert!(mock.messages().is_empty());

    let response = router
        .clone()
        .oneshot(post(
            "/v1/messages",
            &messages_body("igalia/gpt-5.5", false),
            &[],
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(mock.messages().len(), 1);

    let ids = model_ids(&router).await;
    assert!(ids.contains(&"igalia/gpt-5.5".to_string()));
    assert_eq!(
        ids.iter().filter(|id| *id == "gpt-5.5").count(),
        1,
        "the bare name is listed once, for the provider"
    );
}

#[tokio::test]
async fn client_credentials_never_reach_an_upstream() {
    let mock = start_mock(&[]).await;
    let dir = tempfile::tempdir().unwrap();
    let router = app(upstreams(&dir, igalia(&mock.base, &[], false)), None);

    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        let response = router
            .clone()
            .oneshot(post(
                path,
                &messages_body("igalia/glm-5p3", false),
                &[
                    ("authorization", CLIENT_TOKEN),
                    ("x-api-key", CLIENT_API_KEY),
                    ("anthropic-version", "2023-06-01"),
                    ("cookie", "private=1"),
                ],
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    let seen = mock.messages();
    assert_eq!(seen.len(), 2);
    for request in seen {
        assert_eq!(
            request
                .headers
                .get_all(header::AUTHORIZATION)
                .iter()
                .count(),
            1
        );
        assert_eq!(
            request.headers[header::AUTHORIZATION],
            format!("Bearer {UPSTREAM_KEY}")
        );
        assert_eq!(request.headers.get_all("x-api-key").iter().count(), 1);
        assert_eq!(request.headers["x-api-key"], UPSTREAM_KEY);
        assert_eq!(request.headers["anthropic-version"], "2023-06-01");
        assert!(!request.headers.contains_key(header::COOKIE));
        for (name, value) in &request.headers {
            let value = value.to_str().unwrap_or_default();
            assert!(
                !value.contains("client-oauth-token") && !value.contains(CLIENT_API_KEY),
                "client credential leaked in {name}"
            );
        }
    }
}

#[tokio::test]
async fn count_tokens_routes_to_the_upstream_even_when_claude_is_selected() {
    let mock = start_mock(&[]).await;
    let claude = start_mock(&[]).await;
    let dir = tempfile::tempdir().unwrap();
    let router = app(
        upstreams(&dir, igalia(&mock.base, &[], false)),
        Some(&claude.base),
    );
    let switched = router
        .clone()
        .oneshot(control(
            Method::POST,
            "/control/backend",
            r#"{"backend":"claude"}"#,
        ))
        .await
        .unwrap();
    assert_eq!(switched.status(), StatusCode::OK);

    let response = router
        .clone()
        .oneshot(post(
            "/v1/messages/count_tokens?beta=true",
            &messages_body("igalia/glm-5p3", false),
            &[("authorization", CLIENT_TOKEN)],
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_json(response).await, json!({"input_tokens": 3}));
    let seen = mock.messages();
    assert_eq!(seen.len(), 1);
    assert_eq!(
        seen[0].path_and_query,
        "/v1/messages/count_tokens?beta=true"
    );
    let forwarded: Value = serde_json::from_slice(&seen[0].body).unwrap();
    assert_eq!(forwarded["model"], "glm-5p3");

    let response = router
        .oneshot(post(
            "/v1/messages",
            &messages_body("igalia/glm-5p3", false),
            &[("authorization", CLIENT_TOKEN)],
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(mock.messages().len(), 2);
    assert!(claude.messages().is_empty(), "Claude never saw the request");
}

#[tokio::test]
async fn models_list_upstream_and_discovered_models_after_the_existing_ones() {
    let mock = start_mock(&["disc-1", "glm-5p3"]).await;
    let dir = tempfile::tempdir().unwrap();
    let with_upstreams = app(
        upstreams(&dir, igalia(&mock.base, &["glm-5p3", "kimi-k3"], true)),
        None,
    );
    let without = app(Upstreams::default(), None);

    let ids = model_ids(&with_upstreams).await;
    let existing = model_ids(&without).await;
    assert_eq!(
        ids[..existing.len()],
        existing[..],
        "existing entries first"
    );
    assert_eq!(
        ids[existing.len()..],
        [
            "igalia/glm-5p3",
            "igalia/kimi-k3",
            "igalia/disc-1",
            "glm-5p3",
            "kimi-k3",
            "disc-1",
        ]
    );
    let again = model_ids(&with_upstreams).await;
    assert_eq!(again, ids);
    assert_eq!(
        mock.model_fetches.load(Ordering::SeqCst),
        1,
        "one fetch within the discovery interval"
    );
    let fetch = mock
        .recorded
        .lock()
        .unwrap()
        .iter()
        .find(|request| request.path_and_query == "/v1/models")
        .cloned()
        .unwrap();
    assert_eq!(fetch.headers["x-api-key"], UPSTREAM_KEY);

    let response = with_upstreams
        .oneshot(post("/v1/messages", &messages_body("disc-1", false), &[]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(mock.messages().len(), 1, "a discovered bare name routes");
}

#[tokio::test]
async fn failed_discovery_keeps_the_last_good_list() {
    let mock = start_mock(&["disc-1"]).await;
    let dir = tempfile::tempdir().unwrap();
    let router = app(
        upstreams(&dir, igalia(&mock.base, &[], true)).with_discovery_interval(Duration::ZERO),
        None,
    );
    assert!(
        model_ids(&router)
            .await
            .contains(&"igalia/disc-1".to_string())
    );
    mock.fail_models.store(true, Ordering::SeqCst);
    let before = mock.model_fetches.load(Ordering::SeqCst);
    let ids = model_ids(&router).await;
    assert!(mock.model_fetches.load(Ordering::SeqCst) > before);
    assert!(ids.contains(&"igalia/disc-1".to_string()), "{ids:?}");
    assert!(ids.contains(&"disc-1".to_string()), "{ids:?}");
}

#[tokio::test]
async fn session_records_the_upstream_and_honours_allowed_backends() {
    let mock = start_mock(&[]).await;
    let dir = tempfile::tempdir().unwrap();
    let router = app(upstreams(&dir, igalia(&mock.base, &[], false)), None);
    let session = [("x-claude-code-session-id", "sess-up")];

    let response = router
        .clone()
        .oneshot(post(
            "/v1/messages",
            &messages_body("igalia/glm-5p3", false),
            &session,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let record = body_json(
        router
            .clone()
            .oneshot(control(Method::GET, "/control/sessions/sess-up", ""))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(record["backend"], "igalia");
    assert_eq!(record["model"], "igalia/glm-5p3");
    assert_eq!(record["allowed"], json!(["codex", "claude", "igalia"]));

    let unknown = router
        .clone()
        .oneshot(control(
            Method::POST,
            "/control/sessions/sess-up/allowed",
            r#"{"backends":["nope"]}"#,
        ))
        .await
        .unwrap();
    assert_eq!(unknown.status(), StatusCode::BAD_REQUEST);

    let narrowed = router
        .clone()
        .oneshot(control(
            Method::POST,
            "/control/sessions/sess-up/allowed",
            r#"{"backends":["codex"]}"#,
        ))
        .await
        .unwrap();
    assert_eq!(narrowed.status(), StatusCode::OK);
    let refused = router
        .clone()
        .oneshot(post(
            "/v1/messages",
            &messages_body("igalia/glm-5p3", false),
            &session,
        ))
        .await
        .unwrap();
    assert_eq!(refused.status(), StatusCode::CONFLICT);
    assert_eq!(mock.messages().len(), 1, "the refused request never left");

    let upstream_only = router
        .clone()
        .oneshot(control(
            Method::POST,
            "/control/sessions/sess-up/allowed",
            r#"{"backends":["igalia"]}"#,
        ))
        .await
        .unwrap();
    assert_eq!(upstream_only.status(), StatusCode::OK);
    assert_eq!(body_json(upstream_only).await["allowed"], json!(["igalia"]));
    let response = router
        .oneshot(post(
            "/v1/messages",
            &messages_body("igalia/glm-5p3", false),
            &session,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn without_upstreams_a_prefixed_model_is_still_unknown() {
    let router = app(Upstreams::default(), None);
    let response = router
        .clone()
        .oneshot(post(
            "/v1/messages",
            &messages_body("igalia/glm-5p3", false),
            &[],
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let session = [("x-claude-code-session-id", "sess-plain")];
    router
        .clone()
        .oneshot(post(
            "/v1/messages",
            &messages_body("gpt-5.5", false),
            &session,
        ))
        .await
        .unwrap();
    let record = body_json(
        router
            .oneshot(control(Method::GET, "/control/sessions/sess-plain", ""))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(record["allowed"], json!(["codex", "claude"]));
}
