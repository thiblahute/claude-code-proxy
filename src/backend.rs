//! Backend selection for `/v1/messages`.
//!
//! Claude Code keeps one fixed `ANTHROPIC_BASE_URL` while the proxy decides
//! which backend answers: `codex` (routed through the registry, the default)
//! or `claude` (the request is passed through to Anthropic unchanged). The
//! switch happens by a control API call or automatically when Codex answers
//! 429 before any body byte has been sent.

use std::collections::{HashMap, VecDeque};
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
use std::str::FromStr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{ConnectInfo, Extension, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use futures_util::StreamExt;
use serde_json::{Map, Value, json};
use url::{Host, Url};

use crate::anthropic::json_error;
use crate::logging::{Logger, create_logger};

pub const DEFAULT_CLAUDE_BASE_URL: &str = "https://api.anthropic.com";

/// Most sessions tracked by the control API. The oldest first-seen session is
/// evicted first.
const MAX_TRACKED_SESSIONS: usize = 1024;

/// Request headers that reach Anthropic. Everything else, including Cookie,
/// Host, Origin, the proxy's session header and proxy headers, stays here.
const FORWARDED_HEADERS: &[&str] = &[
    "authorization",
    "anthropic-beta",
    "anthropic-version",
    "anthropic-dangerous-direct-browser-access",
    "content-type",
    "x-app",
    "user-agent",
];
const FORWARDED_HEADER_PREFIX: &str = "x-stainless-";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Codex,
    Claude,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Codex => "codex",
            Backend::Claude => "claude",
        }
    }
}

impl FromStr for Backend {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "codex" => Ok(Backend::Codex),
            "claude" => Ok(Backend::Claude),
            other => Err(format!(
                "unknown backend \"{other}\", expected \"codex\" or \"claude\""
            )),
        }
    }
}

/// Validated, non-empty, duplicate-free list of backends. The first entry is
/// the backend used at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendOrder(Vec<Backend>);

impl BackendOrder {
    pub fn new(backends: Vec<Backend>) -> Result<Self, String> {
        if backends.is_empty() {
            return Err("backend order is empty".to_string());
        }
        for (index, backend) in backends.iter().enumerate() {
            if backends[..index].contains(backend) {
                return Err(format!(
                    "backend \"{}\" is listed more than once",
                    backend.as_str()
                ));
            }
        }
        Ok(Self(backends))
    }

    pub fn codex_only() -> Self {
        Self(vec![Backend::Codex])
    }

    pub fn first(&self) -> Backend {
        self.0[0]
    }

    pub fn contains(&self, backend: Backend) -> bool {
        self.0.contains(&backend)
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.0.iter().map(|backend| backend.as_str()).collect()
    }
}

fn parse_names<'a>(
    source: &str,
    names: impl IntoIterator<Item = &'a str>,
) -> Result<BackendOrder, String> {
    let backends = names
        .into_iter()
        .map(|name| {
            name.trim()
                .parse::<Backend>()
                .map_err(|error| format!("{source}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    BackendOrder::new(backends).map_err(|error| format!("{source}: {error}"))
}

/// Parses `CCP_BACKENDS`, a comma-separated list such as `codex,claude`.
pub fn parse_env_order(raw: &str) -> Result<BackendOrder, String> {
    if raw.trim().is_empty() {
        return Err("CCP_BACKENDS: backend order is empty".to_string());
    }
    parse_names("CCP_BACKENDS", raw.split(','))
}

/// Picks the backend order from the environment first, then `config.json`,
/// then the default of `codex` alone.
pub fn order_from_sources(
    env_value: Option<&str>,
    file_value: Option<&[String]>,
) -> Result<BackendOrder, String> {
    if let Some(raw) = env_value {
        return parse_env_order(raw);
    }
    if let Some(names) = file_value {
        return parse_names("config.json backends", names.iter().map(String::as_str));
    }
    Ok(BackendOrder::codex_only())
}

/// Reason for the current backend. Serialized as an object with a `kind` key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitchReason {
    Config,
    Manual,
    RateLimit429 {
        from: Backend,
        retry_after: Option<String>,
    },
}

impl SwitchReason {
    fn to_json(&self) -> Value {
        match self {
            SwitchReason::Config => json!({ "kind": "config" }),
            SwitchReason::Manual => json!({ "kind": "manual" }),
            SwitchReason::RateLimit429 { from, retry_after } => json!({
                "kind": "rate_limit_429",
                "from": from.as_str(),
                "retry_after": retry_after,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectorState {
    pub current: Backend,
    /// RFC 3339 time the current backend became active.
    pub since: String,
    pub last_switch_reason: SwitchReason,
    /// Reserved for a scheduled re-check of a failed backend. Always `None` for now.
    pub next_check: Option<String>,
}

/// Startup state: the first backend of the configured order, reason `config`.
pub fn startup_state(order: &BackendOrder, now: &str) -> SelectorState {
    SelectorState {
        current: order.first(),
        since: now.to_string(),
        last_switch_reason: SwitchReason::Config,
        next_check: None,
    }
}

/// Manual switch from the control API. `since` only moves when the backend
/// actually changes; the reason is always recorded as `manual`.
pub fn manual_switch(
    state: &SelectorState,
    order: &BackendOrder,
    target: Backend,
    now: &str,
) -> Result<SelectorState, String> {
    if !order.contains(target) {
        return Err(format!(
            "backend \"{}\" is not in the configured order",
            target.as_str()
        ));
    }
    Ok(SelectorState {
        current: target,
        since: if target == state.current {
            state.since.clone()
        } else {
            now.to_string()
        },
        last_switch_reason: SwitchReason::Manual,
        next_check: state.next_check.clone(),
    })
}

/// Codex answered 429 before any body byte was sent. Switches to Claude when
/// Claude is configured, otherwise returns `None` and nothing changes.
pub fn rate_limit_failover(
    state: &SelectorState,
    order: &BackendOrder,
    retry_after: Option<String>,
    now: &str,
) -> Option<SelectorState> {
    if state.current != Backend::Codex || !order.contains(Backend::Claude) {
        return None;
    }
    Some(SelectorState {
        current: Backend::Claude,
        since: now.to_string(),
        last_switch_reason: SwitchReason::RateLimit429 {
            from: Backend::Codex,
            retry_after,
        },
        next_check: state.next_check.clone(),
    })
}

#[derive(Debug, Clone)]
pub struct SessionRecord {
    /// Backend that answered the session's most recent `/v1/messages` request.
    pub backend: Backend,
    pub since: String,
    pub last_switch_reason: SwitchReason,
    pub next_check: Option<String>,
    /// Last `model` string the session sent.
    pub model: Option<String>,
}

impl SessionRecord {
    fn to_json(&self) -> Value {
        json!({
            "backend": self.backend.as_str(),
            "since": self.since,
            "last_switch_reason": self.last_switch_reason.to_json(),
            "next_check": self.next_check,
            "model": self.model,
        })
    }
}

#[derive(Default)]
struct SessionBook {
    records: HashMap<String, SessionRecord>,
    first_seen: VecDeque<String>,
}

impl SessionBook {
    fn upsert(&mut self, session_id: &str, record: SessionRecord) {
        if let Some(existing) = self.records.get_mut(session_id) {
            let model = record.model.or_else(|| existing.model.take());
            *existing = SessionRecord { model, ..record };
            return;
        }
        self.records.insert(session_id.to_string(), record);
        self.first_seen.push_back(session_id.to_string());
        while self.first_seen.len() > MAX_TRACKED_SESSIONS {
            if let Some(oldest) = self.first_seen.pop_front() {
                self.records.remove(&oldest);
            }
        }
    }
}

/// Runtime selector shared by every request handled by one app instance.
pub struct BackendSelector {
    order: BackendOrder,
    state: Mutex<SelectorState>,
    sessions: Mutex<SessionBook>,
}

impl BackendSelector {
    pub fn new(order: BackendOrder) -> Self {
        let state = startup_state(&order, &now_rfc3339());
        Self {
            order,
            state: Mutex::new(state),
            sessions: Mutex::new(SessionBook::default()),
        }
    }

    pub fn order(&self) -> &BackendOrder {
        &self.order
    }

    pub fn snapshot(&self) -> SelectorState {
        lock(&self.state).clone()
    }

    pub fn switch_manual(&self, target: Backend) -> Result<SelectorState, String> {
        let mut state = lock(&self.state);
        let next = manual_switch(&state, &self.order, target, &now_rfc3339())?;
        *state = next.clone();
        Ok(next)
    }

    /// Returns the new state when the selector switched to Claude.
    pub fn fail_over_on_429(&self, retry_after: Option<String>) -> Option<SelectorState> {
        let mut state = lock(&self.state);
        let next = rate_limit_failover(&state, &self.order, retry_after, &now_rfc3339())?;
        *state = next.clone();
        Some(next)
    }

    pub fn record_session(&self, session_id: &str, backend: Backend, model: Option<&str>) {
        let switch = self.snapshot();
        let record = SessionRecord {
            backend,
            since: switch.since,
            last_switch_reason: switch.last_switch_reason,
            next_check: switch.next_check,
            model: model.map(str::to_string),
        };
        lock(&self.sessions).upsert(session_id, record);
    }

    pub fn session(&self, session_id: &str) -> Option<SessionRecord> {
        lock(&self.sessions).records.get(session_id).cloned()
    }

    fn sessions(&self) -> Vec<(String, SessionRecord)> {
        let book = lock(&self.sessions);
        book.first_seen
            .iter()
            .filter_map(|id| {
                book.records
                    .get(id)
                    .map(|record| (id.clone(), record.clone()))
            })
            .collect()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn now_rfc3339() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_default()
}

/// Validates `CCP_CLAUDE_BASE_URL`. Only https is accepted, plus plain http
/// to the loopback addresses so tests can run against a local mock.
pub fn validate_claude_base_url(raw: &str) -> Result<Url, String> {
    let url = Url::parse(raw)
        .map_err(|error| format!("CCP_CLAUDE_BASE_URL is not a valid URL: {error}"))?;
    let loopback = match url.host() {
        Some(Host::Ipv4(ip)) => ip == Ipv4Addr::LOCALHOST,
        Some(Host::Ipv6(ip)) => ip == Ipv6Addr::LOCALHOST,
        _ => false,
    };
    match url.scheme() {
        "https" => Ok(url),
        "http" if loopback => Ok(url),
        other => Err(format!(
            "CCP_CLAUDE_BASE_URL must use https (http is only allowed for 127.0.0.1 or [::1]), got {other}"
        )),
    }
}

/// Upstream used by the Claude passthrough.
pub struct ClaudeUpstream {
    base_url: Url,
    client: reqwest::Client,
}

impl ClaudeUpstream {
    pub fn from_base_url(raw: Option<&str>) -> Result<Self, String> {
        let base_url = validate_claude_base_url(raw.unwrap_or(DEFAULT_CLAUDE_BASE_URL))?;
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .build()
            .map_err(|error| format!("failed to build Claude HTTP client: {error}"))?;
        Ok(Self { base_url, client })
    }

    fn messages_url(&self, query: Option<&str>) -> String {
        let base = self.base_url.as_str().trim_end_matches('/');
        match query {
            Some(query) => format!("{base}/v1/messages?{query}"),
            None => format!("{base}/v1/messages"),
        }
    }
}

/// Everything the server needs to route `/v1/messages`.
pub struct BackendSettings {
    pub order: BackendOrder,
    pub upstream: ClaudeUpstream,
}

impl BackendSettings {
    pub fn defaults() -> Self {
        Self {
            order: BackendOrder::codex_only(),
            upstream: ClaudeUpstream::from_base_url(None)
                .expect("default Claude upstream is valid"),
        }
    }
}

/// Reads `CCP_BACKENDS`, the `backends` key of `config.json`, and
/// `CCP_CLAUDE_BASE_URL`. Any invalid value is an error.
pub fn load_settings() -> Result<BackendSettings, String> {
    let env_order = std::env::var("CCP_BACKENDS").ok();
    let file_order = crate::config::file_backends_in(&crate::paths::config_dir());
    let order = order_from_sources(env_order.as_deref(), file_order.as_deref())?;
    let upstream =
        ClaudeUpstream::from_base_url(std::env::var("CCP_CLAUDE_BASE_URL").ok().as_deref())?;
    Ok(BackendSettings { order, upstream })
}

/// Claude passthrough needs the caller's own credentials. The proxy never
/// supplies a token of its own.
pub fn has_passthrough_credentials(headers: &HeaderMap) -> bool {
    headers.contains_key(header::AUTHORIZATION)
}

fn is_forwarded_header(name: &HeaderName) -> bool {
    FORWARDED_HEADERS.contains(&name.as_str()) || name.as_str().starts_with(FORWARDED_HEADER_PREFIX)
}

pub fn outbound_headers(inbound: &HeaderMap) -> HeaderMap {
    let mut outbound = HeaderMap::new();
    for (name, value) in inbound {
        if is_forwarded_header(name) {
            outbound.append(name.clone(), value.clone());
        }
    }
    outbound
}

/// Logs one line when the passthrough response body ends, or is dropped by
/// the client. Carries metadata only, never headers or bodies.
struct PassthroughSummary {
    log: Logger,
    req_id: String,
    model: Option<String>,
    status: StatusCode,
    started_at: Instant,
    bytes: u64,
}

impl Drop for PassthroughSummary {
    fn drop(&mut self) {
        self.log.info(
            "claude passthrough completed",
            Some(Map::from_iter([
                ("reqId".to_string(), json!(&self.req_id)),
                ("backend".to_string(), json!("claude")),
                ("model".to_string(), json!(&self.model)),
                ("status".to_string(), json!(self.status.as_u16())),
                (
                    "latencyMs".to_string(),
                    json!(self.started_at.elapsed().as_millis() as u64),
                ),
                ("bytes".to_string(), json!(self.bytes)),
            ])),
        );
    }
}

/// Sends the original request bytes to Anthropic and streams the response
/// back with the upstream status and content-type.
pub async fn forward_messages(
    upstream: &ClaudeUpstream,
    req_id: &str,
    model: Option<&str>,
    headers: &HeaderMap,
    query: Option<&str>,
    body: Bytes,
) -> Response {
    if !has_passthrough_credentials(headers) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "Claude passthrough needs the authorization header Claude Code sends; none was received",
        );
    }
    let started_at = Instant::now();
    let sent = upstream
        .client
        .post(upstream.messages_url(query))
        .headers(outbound_headers(headers))
        .body(body)
        .send()
        .await;
    let upstream_response = match sent {
        Ok(response) => response,
        Err(error) => {
            create_logger("backend").warn(
                "claude passthrough failed",
                Some(Map::from_iter([
                    ("reqId".to_string(), json!(req_id)),
                    ("error".to_string(), json!(error.to_string())),
                ])),
            );
            return json_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "Claude upstream request failed",
            );
        }
    };
    let status = upstream_response.status();
    let content_type = upstream_response
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned();
    let summary = PassthroughSummary {
        log: create_logger("backend"),
        req_id: req_id.to_string(),
        model: model.map(str::to_string),
        status,
        started_at,
        bytes: 0,
    };
    let chunks = futures_util::stream::unfold(
        (upstream_response.bytes_stream(), summary),
        |(mut chunks, mut summary)| async move {
            let chunk = chunks.next().await?;
            if let Ok(bytes) = &chunk {
                summary.bytes += bytes.len() as u64;
            }
            Some((chunk, (chunks, summary)))
        },
    );
    let mut response = Response::new(Body::from_stream(chunks));
    *response.status_mut() = status;
    if let Some(content_type) = content_type {
        response
            .headers_mut()
            .insert(header::CONTENT_TYPE, content_type);
    }
    response
}

fn no_store(mut response: Response) -> Response {
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

fn json_no_store(status: StatusCode, value: Value) -> Response {
    no_store((status, Json(value)).into_response())
}

/// Loopback-only, no Origin header. Returns the rejection response, if any.
fn control_rejection(
    peer: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: &HeaderMap,
) -> Option<Response> {
    let local =
        peer.is_some_and(|Extension(ConnectInfo(addr))| addr.ip().to_canonical().is_loopback());
    if !local {
        return Some(no_store(json_error(
            StatusCode::FORBIDDEN,
            "permission_error",
            "Backend control requires a local connection",
        )));
    }
    if headers.contains_key(header::ORIGIN) {
        return Some(no_store(json_error(
            StatusCode::FORBIDDEN,
            "permission_error",
            "Backend control rejects requests with an Origin header",
        )));
    }
    None
}

fn is_json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.split(';').next())
        .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("application/json"))
}

fn selector_json(state: &SelectorState, order: &BackendOrder) -> Value {
    json!({
        "current": state.current.as_str(),
        "since": state.since,
        "last_switch_reason": state.last_switch_reason.to_json(),
        "next_check": state.next_check,
        "order": order.names(),
    })
}

async fn get_backend(
    State(selector): State<Arc<BackendSelector>>,
    peer: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
) -> Response {
    if let Some(rejection) = control_rejection(peer, &headers) {
        return rejection;
    }
    json_no_store(
        StatusCode::OK,
        selector_json(&selector.snapshot(), selector.order()),
    )
}

async fn post_backend(
    State(selector): State<Arc<BackendSelector>>,
    peer: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Some(rejection) = control_rejection(peer, &headers) {
        return rejection;
    }
    if !is_json_content_type(&headers) {
        return no_store(json_error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "Content-Type must be application/json",
        ));
    }
    let requested = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .get("backend")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    let Some(requested) = requested else {
        return no_store(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Body must be a JSON object with a \"backend\" string",
        ));
    };
    let target = match requested.parse::<Backend>() {
        Ok(target) => target,
        Err(message) => {
            return no_store(json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            ));
        }
    };
    match selector.switch_manual(target) {
        Ok(state) => {
            create_logger("backend").info(
                "backend switched",
                Some(Map::from_iter([
                    ("backend".to_string(), json!(target.as_str())),
                    ("reason".to_string(), json!("manual")),
                ])),
            );
            json_no_store(StatusCode::OK, selector_json(&state, selector.order()))
        }
        Err(message) => no_store(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        )),
    }
}

fn session_entry_json(session_id: Option<&str>, record: &SessionRecord) -> Value {
    let mut value = record.to_json();
    if let (Some(session_id), Some(object)) = (session_id, value.as_object_mut()) {
        object.insert("id".to_string(), json!(session_id));
    }
    value
}

async fn list_sessions(
    State(selector): State<Arc<BackendSelector>>,
    peer: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
) -> Response {
    if let Some(rejection) = control_rejection(peer, &headers) {
        return rejection;
    }
    let sessions: Vec<Value> = selector
        .sessions()
        .iter()
        .map(|(id, record)| session_entry_json(Some(id), record))
        .collect();
    json_no_store(StatusCode::OK, json!({ "sessions": sessions }))
}

async fn get_session(
    State(selector): State<Arc<BackendSelector>>,
    axum::extract::Path(session_id): axum::extract::Path<String>,
    peer: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
) -> Response {
    if let Some(rejection) = control_rejection(peer, &headers) {
        return rejection;
    }
    match selector.session(&session_id) {
        Some(record) => json_no_store(StatusCode::OK, session_entry_json(None, &record)),
        None => no_store(json_error(
            StatusCode::NOT_FOUND,
            "not_found_error",
            "Unknown session",
        )),
    }
}

/// Control routes. They carry their own state, so they merge into the main
/// router after `with_state`.
pub fn control_router(selector: Arc<BackendSelector>) -> Router {
    Router::new()
        .route("/control/backend", get(get_backend).post(post_backend))
        .route("/control/sessions", get(list_sessions))
        .route("/control/sessions/{id}", get(get_session))
        .with_state(selector)
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: &str = "2026-11-16T10:00:00Z";
    const T1: &str = "2026-11-16T10:05:00Z";

    fn order(backends: Vec<Backend>) -> BackendOrder {
        BackendOrder::new(backends).unwrap()
    }

    #[test]
    fn env_order_parses_comma_separated_names() {
        assert_eq!(
            parse_env_order("codex, claude").unwrap(),
            order(vec![Backend::Codex, Backend::Claude])
        );
        assert_eq!(
            parse_env_order("claude").unwrap(),
            order(vec![Backend::Claude])
        );
    }

    #[test]
    fn env_order_rejects_empty_duplicate_and_unknown_names() {
        assert!(parse_env_order("  ").unwrap_err().contains("empty"));
        assert!(
            parse_env_order("codex,codex")
                .unwrap_err()
                .contains("more than once")
        );
        assert!(
            parse_env_order("codex,kimi")
                .unwrap_err()
                .contains("unknown backend \"kimi\"")
        );
        assert!(parse_env_order("codex,,claude").is_err());
    }

    #[test]
    fn config_order_is_read_from_backends_key() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"backends": ["claude", "codex"]}"#,
        )
        .unwrap();
        let names = crate::config::file_backends_in(dir.path());
        assert_eq!(
            names.as_deref(),
            Some(["claude".to_string(), "codex".to_string()].as_slice())
        );
        assert_eq!(
            order_from_sources(None, names.as_deref()).unwrap(),
            order(vec![Backend::Claude, Backend::Codex])
        );
    }

    #[test]
    fn config_order_rejects_empty_duplicate_and_unknown_names() {
        assert!(order_from_sources(None, Some(&[])).is_err());
        let duplicate = ["codex".to_string(), "codex".to_string()];
        assert!(order_from_sources(None, Some(&duplicate)).is_err());
        let unknown = ["grok".to_string()];
        assert!(
            order_from_sources(None, Some(&unknown))
                .unwrap_err()
                .contains("unknown backend")
        );
    }

    #[test]
    fn env_order_wins_over_config_and_default_is_codex_only() {
        let file = ["claude".to_string()];
        assert_eq!(
            order_from_sources(Some("codex"), Some(&file)).unwrap(),
            order(vec![Backend::Codex])
        );
        assert_eq!(
            order_from_sources(None, None).unwrap(),
            BackendOrder::codex_only()
        );
    }

    #[test]
    fn startup_uses_first_backend_with_config_reason() {
        let state = startup_state(&order(vec![Backend::Codex, Backend::Claude]), T0);
        assert_eq!(state.current, Backend::Codex);
        assert_eq!(state.since, T0);
        assert_eq!(state.last_switch_reason, SwitchReason::Config);
        assert_eq!(state.next_check, None);
    }

    #[test]
    fn manual_switch_records_manual_reason_and_since() {
        let both = order(vec![Backend::Codex, Backend::Claude]);
        let start = startup_state(&both, T0);
        let switched = manual_switch(&start, &both, Backend::Claude, T1).unwrap();
        assert_eq!(switched.current, Backend::Claude);
        assert_eq!(switched.since, T1);
        assert_eq!(switched.last_switch_reason, SwitchReason::Manual);

        let same = manual_switch(&switched, &both, Backend::Claude, "later").unwrap();
        assert_eq!(same.since, T1, "since only moves when the backend changes");
        assert_eq!(same.last_switch_reason, SwitchReason::Manual);
    }

    #[test]
    fn manual_switch_rejects_backend_outside_the_order() {
        let codex_only = BackendOrder::codex_only();
        let start = startup_state(&codex_only, T0);
        let error = manual_switch(&start, &codex_only, Backend::Claude, T1).unwrap_err();
        assert!(error.contains("not in the configured order"));
    }

    #[test]
    fn rate_limit_429_switches_from_codex_and_keeps_retry_after() {
        let both = order(vec![Backend::Codex, Backend::Claude]);
        let start = startup_state(&both, T0);
        let switched = rate_limit_failover(&start, &both, Some("42".to_string()), T1).unwrap();
        assert_eq!(switched.current, Backend::Claude);
        assert_eq!(switched.since, T1);
        assert_eq!(
            switched.last_switch_reason,
            SwitchReason::RateLimit429 {
                from: Backend::Codex,
                retry_after: Some("42".to_string()),
            }
        );
    }

    #[test]
    fn rate_limit_429_does_not_switch_without_claude_or_when_already_claude() {
        let codex_only = BackendOrder::codex_only();
        let start = startup_state(&codex_only, T0);
        assert!(rate_limit_failover(&start, &codex_only, None, T1).is_none());

        let both = order(vec![Backend::Codex, Backend::Claude]);
        let claude = manual_switch(&start, &both, Backend::Claude, T1).unwrap();
        assert!(rate_limit_failover(&claude, &both, None, T1).is_none());
    }

    #[test]
    fn selector_tracks_state_and_sessions() {
        let selector = BackendSelector::new(order(vec![Backend::Codex, Backend::Claude]));
        assert_eq!(selector.snapshot().current, Backend::Codex);

        selector.record_session("s1", Backend::Codex, Some("gpt-5.5"));
        selector.record_session("s1", Backend::Codex, None);
        let record = selector.session("s1").unwrap();
        assert_eq!(record.model.as_deref(), Some("gpt-5.5"));
        assert_eq!(record.backend, Backend::Codex);

        assert!(selector.fail_over_on_429(Some("5".to_string())).is_some());
        assert_eq!(selector.snapshot().current, Backend::Claude);
        selector.switch_manual(Backend::Codex).unwrap();
        assert_eq!(selector.snapshot().last_switch_reason, SwitchReason::Manual);
    }

    #[test]
    fn session_book_evicts_the_oldest_session() {
        let selector = BackendSelector::new(BackendOrder::codex_only());
        for index in 0..=MAX_TRACKED_SESSIONS {
            selector.record_session(&format!("s{index}"), Backend::Codex, Some("m"));
        }
        assert!(selector.session("s0").is_none());
        assert!(selector.session("s1").is_some());
        assert!(
            selector
                .session(&format!("s{MAX_TRACKED_SESSIONS}"))
                .is_some()
        );
        assert_eq!(selector.sessions().len(), MAX_TRACKED_SESSIONS);
    }

    #[test]
    fn claude_base_url_allows_https_and_loopback_http_only() {
        assert!(validate_claude_base_url("https://api.anthropic.com").is_ok());
        assert!(validate_claude_base_url("http://127.0.0.1:9000").is_ok());
        assert!(validate_claude_base_url("http://[::1]:9000").is_ok());
        assert!(validate_claude_base_url("http://api.anthropic.com").is_err());
        assert!(validate_claude_base_url("http://localhost:9000").is_err());
        assert!(validate_claude_base_url("ftp://127.0.0.1").is_err());
        assert!(validate_claude_base_url("not a url").is_err());
    }

    #[test]
    fn passthrough_forwards_only_allowlisted_headers() {
        let mut inbound = HeaderMap::new();
        for (name, value) in [
            ("authorization", "Bearer secret"),
            ("anthropic-beta", "tools-2025"),
            ("x-stainless-os", "Linux"),
            ("cookie", "session=1"),
            ("origin", "http://evil.example"),
            ("host", "127.0.0.1:18765"),
            ("x-claude-code-session-id", "abc"),
        ] {
            inbound.insert(
                HeaderName::from_static(name),
                HeaderValue::from_static(value),
            );
        }
        let outbound = outbound_headers(&inbound);
        assert!(outbound.contains_key("authorization"));
        assert!(outbound.contains_key("anthropic-beta"));
        assert!(outbound.contains_key("x-stainless-os"));
        assert!(!outbound.contains_key("cookie"));
        assert!(!outbound.contains_key("origin"));
        assert!(!outbound.contains_key("host"));
        assert!(!outbound.contains_key("x-claude-code-session-id"));
    }
}
