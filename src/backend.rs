//! Backend selection for `/v1/messages`.
//!
//! Claude Code keeps one fixed `ANTHROPIC_BASE_URL` while the proxy decides
//! which backend answers: `codex` (routed through the registry, the default)
//! or `claude` (the request is passed through to Anthropic unchanged). The
//! switch happens by a control API call or automatically when Codex answers
//! 429 before any body byte has been sent. Each session may be restricted to a
//! subset of the configured backends, so a session never silently moves to a
//! backend it is not allowed to use.
//!
//! Named Anthropic-compatible upstreams (`crate::upstream`) are routed by the
//! request's model rather than by the selector; they appear here as
//! `Backend::Upstream` so session records and allowed lists can name them.

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
    routing::{get, post},
};
use futures_util::StreamExt;
use serde_json::{Map, Value, json};
use url::{Host, Url};

use crate::anthropic::json_error;
use crate::logging::{Logger, create_logger};

pub const DEFAULT_CLAUDE_BASE_URL: &str = "https://api.anthropic.com";

/// Upper bound for `retry_codex_after_secs` (one day).
pub const MAX_RETRY_CODEX_AFTER_SECS: u64 = 86_400;

const MESSAGES_PATH: &str = "/v1/messages";
const COUNT_TOKENS_PATH: &str = "/v1/messages/count_tokens";

/// Most sessions tracked by the control API. The oldest first-seen session is
/// evicted first.
const MAX_TRACKED_SESSIONS: usize = 1024;

/// Request headers that reach Anthropic. Everything else, including Cookie,
/// Host, Origin, the proxy's session header and proxy headers, stays here.
/// `x-api-key` is forwarded like `authorization`; neither is logged.
const FORWARDED_HEADERS: &[&str] = &[
    "authorization",
    "x-api-key",
    "anthropic-beta",
    "anthropic-version",
    "anthropic-dangerous-direct-browser-access",
    "content-type",
    "x-app",
    "user-agent",
];
const FORWARDED_HEADER_PREFIX: &str = "x-stainless-";

/// A backend that can answer `/v1/messages`. `Codex` and `Claude` are the
/// selector's choices; `Upstream` is a named Anthropic-compatible upstream from
/// the `upstreams` config, only reached by model routing. Each serializes as its
/// name, so the JSON for `codex` and `claude` is unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Backend {
    Codex,
    Claude,
    Upstream(Arc<str>),
}

impl Backend {
    pub fn as_str(&self) -> &str {
        match self {
            Backend::Codex => "codex",
            Backend::Claude => "claude",
            Backend::Upstream(name) => name,
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
        self.0[0].clone()
    }

    pub fn contains(&self, backend: &Backend) -> bool {
        self.0.contains(backend)
    }

    pub fn backends(&self) -> Vec<Backend> {
        self.0.clone()
    }

    pub fn names(&self) -> Vec<&str> {
        names_of(&self.0)
    }
}

fn names_of(backends: &[Backend]) -> Vec<&str> {
    backends.iter().map(|backend| backend.as_str()).collect()
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

/// Failover knobs from `failover_on_429` and `retry_codex_after_secs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FailoverPolicy {
    /// Switch to Claude when Codex answers 429. Off: the 429 reaches the client.
    pub on_429: bool,
    /// Seconds after a 429 switch before the selector moves back to Codex.
    /// `None` keeps Claude until a manual switch.
    pub retry_codex_after_secs: Option<u64>,
}

impl Default for FailoverPolicy {
    fn default() -> Self {
        Self {
            on_429: true,
            retry_codex_after_secs: None,
        }
    }
}

/// Parses `CCP_FAILOVER_ON_429` or the `failover_on_429` config value.
pub fn parse_failover_on_429(raw: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Ok(true),
        "0" | "false" | "off" | "no" => Ok(false),
        _ => Err(format!(
            "CCP_FAILOVER_ON_429 must be true or false, got \"{raw}\""
        )),
    }
}

/// Parses `CCP_RETRY_CODEX_AFTER_SECS`, a whole number of seconds up to
/// `MAX_RETRY_CODEX_AFTER_SECS`.
pub fn parse_retry_codex_after_secs(raw: &str) -> Result<u64, String> {
    let secs = raw.trim().parse::<u64>().map_err(|_| {
        format!("CCP_RETRY_CODEX_AFTER_SECS must be a non-negative whole number, got \"{raw}\"")
    })?;
    check_retry_codex_after_secs(secs, "CCP_RETRY_CODEX_AFTER_SECS")
}

fn check_retry_codex_after_secs(secs: u64, source: &str) -> Result<u64, String> {
    if secs > MAX_RETRY_CODEX_AFTER_SECS {
        return Err(format!(
            "{source} must be at most {MAX_RETRY_CODEX_AFTER_SECS} seconds, got {secs}"
        ));
    }
    Ok(secs)
}

/// Picks the failover policy from the environment first, then the values read
/// from `config.json` (`failover_on_429`, `retry_codex_after_secs`). Values are
/// passed as raw JSON so a wrongly typed key is an error, not a silent skip.
pub fn failover_policy_from_sources(
    env_on_429: Option<&str>,
    file_on_429: Option<&Value>,
    env_retry_secs: Option<&str>,
    file_retry_secs: Option<&Value>,
) -> Result<FailoverPolicy, String> {
    let on_429 = match (env_on_429, file_on_429) {
        (Some(raw), _) => parse_failover_on_429(raw)?,
        (None, Some(Value::Bool(value))) => *value,
        (None, Some(_)) => return Err("config.json failover_on_429 must be a boolean".to_string()),
        (None, None) => true,
    };
    let retry_codex_after_secs = match (env_retry_secs, file_retry_secs) {
        (Some(raw), _) => Some(parse_retry_codex_after_secs(raw)?),
        (None, Some(value)) => {
            let secs = value.as_u64().ok_or_else(|| {
                "config.json retry_codex_after_secs must be a non-negative integer".to_string()
            })?;
            Some(check_retry_codex_after_secs(
                secs,
                "config.json retry_codex_after_secs",
            )?)
        }
        (None, None) => None,
    };
    Ok(FailoverPolicy {
        on_429,
        retry_codex_after_secs,
    })
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
    /// The scheduled retry of Codex after a 429 switch came due.
    RetryCodex,
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
            SwitchReason::RetryCodex => json!({ "kind": "retry_codex" }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectorState {
    pub current: Backend,
    /// RFC 3339 time the current backend became active.
    pub since: String,
    pub last_switch_reason: SwitchReason,
    /// RFC 3339 time the selector will next try Codex again. Set only while a
    /// 429 switch to Claude has a `retry_codex_after_secs` schedule.
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
/// actually changes; the reason is always recorded as `manual` and any
/// scheduled Codex retry is dropped.
pub fn manual_switch(
    state: &SelectorState,
    order: &BackendOrder,
    target: Backend,
    now: &str,
) -> Result<SelectorState, String> {
    if !order.contains(&target) {
        return Err(format!(
            "backend \"{}\" is not in the configured order",
            target.as_str()
        ));
    }
    Ok(SelectorState {
        since: if target == state.current {
            state.since.clone()
        } else {
            now.to_string()
        },
        current: target,
        last_switch_reason: SwitchReason::Manual,
        next_check: None,
    })
}

/// Codex answered 429 before any body byte was sent. Switches to Claude when
/// Claude is configured, otherwise returns `None` and nothing changes. With
/// `retry_codex_after_secs`, `next_check` is the time Codex is tried again.
pub fn rate_limit_failover(
    state: &SelectorState,
    order: &BackendOrder,
    retry_after: Option<String>,
    now: &str,
    retry_codex_after_secs: Option<u64>,
) -> Option<SelectorState> {
    if state.current != Backend::Codex || !order.contains(&Backend::Claude) {
        return None;
    }
    Some(SelectorState {
        current: Backend::Claude,
        since: now.to_string(),
        last_switch_reason: SwitchReason::RateLimit429 {
            from: Backend::Codex,
            retry_after,
        },
        next_check: retry_codex_after_secs.and_then(|secs| add_secs_rfc3339(now, secs)),
    })
}

/// Lazy check run on each request: returns the state moved back to Codex when
/// Claude was chosen by a 429 switch and its `next_check` has passed.
pub fn codex_retry_due(
    state: &SelectorState,
    order: &BackendOrder,
    now: &str,
) -> Option<SelectorState> {
    if state.current != Backend::Claude
        || !matches!(state.last_switch_reason, SwitchReason::RateLimit429 { .. })
        || !order.contains(&Backend::Codex)
    {
        return None;
    }
    let due_at = parse_rfc3339(state.next_check.as_deref()?)?;
    if parse_rfc3339(now)? < due_at {
        return None;
    }
    Some(SelectorState {
        current: Backend::Codex,
        since: now.to_string(),
        last_switch_reason: SwitchReason::RetryCodex,
        next_check: None,
    })
}

fn parse_rfc3339(raw: &str) -> Option<time::OffsetDateTime> {
    time::OffsetDateTime::parse(raw, &time::format_description::well_known::Rfc3339).ok()
}

fn add_secs_rfc3339(now: &str, secs: u64) -> Option<String> {
    let later = parse_rfc3339(now)? + time::Duration::seconds(i64::try_from(secs).ok()?);
    later
        .format(&time::format_description::well_known::Rfc3339)
        .ok()
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
    /// Backends this session may be served by. Defaults to the configured order.
    pub allowed: Vec<Backend>,
}

impl SessionRecord {
    fn to_json(&self) -> Value {
        json!({
            "backend": self.backend.as_str(),
            "since": self.since,
            "last_switch_reason": self.last_switch_reason.to_json(),
            "next_check": self.next_check,
            "model": self.model,
            "allowed": names_of(&self.allowed),
        })
    }
}

/// Why a manual switch was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManualSwitchError {
    /// The target is not in the configured order. Maps to 400.
    NotConfigured(String),
    /// Sessions whose traffic would move to a backend they may not use. Maps to 409.
    SessionsNotAllowed(String),
}

impl ManualSwitchError {
    pub fn message(&self) -> &str {
        match self {
            ManualSwitchError::NotConfigured(message)
            | ManualSwitchError::SessionsNotAllowed(message) => message,
        }
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
    /// Names of the configured upstreams, as `Backend::Upstream`.
    upstreams: Vec<Backend>,
    failover: FailoverPolicy,
    state: Mutex<SelectorState>,
    sessions: Mutex<SessionBook>,
}

impl BackendSelector {
    pub fn new(order: BackendOrder) -> Self {
        Self::with_failover(order, FailoverPolicy::default())
    }

    pub fn with_failover(order: BackendOrder, failover: FailoverPolicy) -> Self {
        Self::with_upstreams(order, failover, Vec::new())
    }

    /// `upstreams` are the configured upstream names. They are never chosen by
    /// the selector but count as backends in each session's allowed list.
    pub fn with_upstreams(
        order: BackendOrder,
        failover: FailoverPolicy,
        upstreams: Vec<Arc<str>>,
    ) -> Self {
        let state = startup_state(&order, &now_rfc3339());
        Self {
            order,
            upstreams: upstreams.into_iter().map(Backend::Upstream).collect(),
            failover,
            state: Mutex::new(state),
            sessions: Mutex::new(SessionBook::default()),
        }
    }

    pub fn order(&self) -> &BackendOrder {
        &self.order
    }

    /// Default allowed list of a session: the configured order, then every
    /// upstream name. Without upstreams this is the configured order.
    fn default_allowed(&self) -> Vec<Backend> {
        let mut allowed = self.order.backends();
        allowed.extend(self.upstreams.iter().cloned());
        allowed
    }

    /// Parses a backend name from the control API: `codex`, `claude` or the
    /// name of a configured upstream.
    pub fn parse_backend(&self, name: &str) -> Result<Backend, String> {
        name.parse::<Backend>().or_else(|error| {
            self.upstreams
                .iter()
                .find(|upstream| upstream.as_str() == name)
                .cloned()
                .ok_or(error)
        })
    }

    pub fn snapshot(&self) -> SelectorState {
        lock(&self.state).clone()
    }

    /// Switches the global backend. Refused when a tracked session would be
    /// moved onto a backend it is not allowed to use.
    pub fn switch_manual(&self, target: Backend) -> Result<SelectorState, ManualSwitchError> {
        let book = lock(&self.sessions);
        let mut blocked: Vec<&str> = book
            .records
            .iter()
            .filter(|(_, record)| record.backend != target && !record.allowed.contains(&target))
            .map(|(id, _)| id.as_str())
            .collect();
        if !blocked.is_empty() {
            blocked.sort_unstable();
            return Err(ManualSwitchError::SessionsNotAllowed(format!(
                "backend \"{}\" is not allowed for session(s): {}",
                target.as_str(),
                blocked.join(", ")
            )));
        }
        let mut state = lock(&self.state);
        let next = manual_switch(&state, &self.order, target.clone(), &now_rfc3339())
            .map_err(ManualSwitchError::NotConfigured)?;
        *state = next.clone();
        Ok(next)
    }

    /// Moves a 429 switch back to Codex once its schedule has passed. Called on
    /// each request; no timer runs in the background.
    pub fn retry_codex_if_due(&self) -> Option<SelectorState> {
        let mut state = lock(&self.state);
        let next = codex_retry_due(&state, &self.order, &now_rfc3339())?;
        *state = next.clone();
        Some(next)
    }

    /// Returns the new state when the selector switched to Claude. Nothing
    /// changes when failover is disabled or the session may not use Claude.
    pub fn fail_over_on_429(
        &self,
        retry_after: Option<String>,
        allowed: &[Backend],
    ) -> Option<SelectorState> {
        if !self.failover.on_429 || !allowed.contains(&Backend::Claude) {
            return None;
        }
        let mut state = lock(&self.state);
        let next = rate_limit_failover(
            &state,
            &self.order,
            retry_after,
            &now_rfc3339(),
            self.failover.retry_codex_after_secs,
        )?;
        *state = next.clone();
        Some(next)
    }

    /// Backends a session may use. Unknown sessions get the configured order.
    pub fn allowed_for(&self, session_id: Option<&str>) -> Vec<Backend> {
        session_id
            .and_then(|id| self.session(id))
            .map(|record| record.allowed)
            .unwrap_or_else(|| self.default_allowed())
    }

    /// Replaces a session's allowed backends. `Ok(None)` when the session is
    /// unknown.
    pub fn set_session_allowed(
        &self,
        session_id: &str,
        allowed: &BackendOrder,
    ) -> Result<Option<SessionRecord>, String> {
        for backend in allowed.backends() {
            if !self.order.contains(&backend) && !self.upstreams.contains(&backend) {
                return Err(format!(
                    "backend \"{}\" is not in the configured order",
                    backend.as_str()
                ));
            }
        }
        let mut book = lock(&self.sessions);
        let Some(record) = book.records.get_mut(session_id) else {
            return Ok(None);
        };
        record.allowed = allowed.backends();
        Ok(Some(record.clone()))
    }

    pub fn record_session(&self, session_id: &str, backend: Backend, model: Option<&str>) {
        let switch = self.snapshot();
        let mut book = lock(&self.sessions);
        let allowed = book
            .records
            .get(session_id)
            .map(|record| record.allowed.clone())
            .unwrap_or_else(|| self.default_allowed());
        let record = SessionRecord {
            backend,
            since: switch.since,
            last_switch_reason: switch.last_switch_reason,
            next_check: switch.next_check,
            model: model.map(str::to_string),
            allowed,
        };
        book.upsert(session_id, record);
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
    validate_base_url(raw, "CCP_CLAUDE_BASE_URL")
}

/// The https-or-loopback rule of `validate_claude_base_url` for any
/// Anthropic-compatible base URL. `source` names the setting in errors.
pub fn validate_base_url(raw: &str, source: &str) -> Result<Url, String> {
    let url = Url::parse(raw).map_err(|error| format!("{source} is not a valid URL: {error}"))?;
    let loopback = match url.host() {
        Some(Host::Ipv4(ip)) => ip == Ipv4Addr::LOCALHOST,
        Some(Host::Ipv6(ip)) => ip == Ipv6Addr::LOCALHOST,
        _ => false,
    };
    match url.scheme() {
        "https" => Ok(url),
        "http" if loopback => Ok(url),
        other => Err(format!(
            "{source} must use https (http is only allowed for 127.0.0.1 or [::1]), got {other}"
        )),
    }
}

/// API key of a named upstream. Never printed: `Debug` hides it and it is only
/// read to build the outbound auth headers.
#[derive(Clone)]
pub struct ApiKey(String);

impl ApiKey {
    /// Trims surrounding whitespace (key files usually end with a newline).
    /// Refuses an empty key or one that cannot be sent as a header value. The
    /// error never contains the key.
    pub fn new(raw: &str) -> Result<Self, String> {
        let key = raw.trim();
        if key.is_empty() {
            return Err("key is empty".to_string());
        }
        HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| "key contains characters not allowed in an HTTP header".to_string())?;
        Ok(Self(key.to_string()))
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ApiKey(<redacted>)")
    }
}

enum UpstreamAuth {
    /// Forward the client's own `authorization` / `x-api-key` (Claude).
    Client,
    /// Drop the client's credentials and send this key instead.
    ApiKey(ApiKey),
}

/// An Anthropic-compatible `/v1/messages` upstream. Either api.anthropic.com
/// for the Claude passthrough, which forwards the client's own credentials, or
/// a named upstream from the `upstreams` config, which never sees the client's
/// credentials and gets its configured key instead.
pub struct AnthropicUpstream {
    /// `claude` or the upstream name; used in logs and error messages.
    label: Arc<str>,
    base_url: Url,
    client: reqwest::Client,
    auth: UpstreamAuth,
}

/// The Claude passthrough is the `AnthropicUpstream` built by `from_base_url`.
pub type ClaudeUpstream = AnthropicUpstream;

impl AnthropicUpstream {
    /// Claude upstream: `CCP_CLAUDE_BASE_URL` or api.anthropic.com, with the
    /// client's credentials passed through.
    pub fn from_base_url(raw: Option<&str>) -> Result<Self, String> {
        let base_url = validate_claude_base_url(raw.unwrap_or(DEFAULT_CLAUDE_BASE_URL))?;
        Ok(Self {
            label: Arc::from("claude"),
            base_url,
            client: build_client("Claude")?,
            auth: UpstreamAuth::Client,
        })
    }

    /// Named upstream authenticated by `key`. `base_url` must already have
    /// passed `validate_base_url`.
    pub fn with_api_key(name: Arc<str>, base_url: Url, key: ApiKey) -> Result<Self, String> {
        let client = build_client(&format!("upstream \"{name}\""))?;
        Ok(Self {
            label: name,
            base_url,
            client,
            auth: UpstreamAuth::ApiKey(key),
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub(crate) fn client(&self) -> &reqwest::Client {
        &self.client
    }

    /// `path` is `/v1/messages`, `/v1/messages/count_tokens` or `/v1/models`;
    /// the query string is passed through unchanged.
    pub(crate) fn endpoint_url(&self, path: &str, query: Option<&str>) -> String {
        let base = self.base_url.as_str().trim_end_matches('/');
        match query {
            Some(query) => format!("{base}{path}?{query}"),
            None => format!("{base}{path}"),
        }
    }

    /// Headers sent upstream. The Claude passthrough forwards the allowlist as
    /// is. A named upstream gets the allowlist without the client's
    /// `authorization` and `x-api-key`, then its own key in both headers: a
    /// client's Claude OAuth token must never reach a third-party gateway.
    pub fn outbound_headers(&self, inbound: &HeaderMap) -> HeaderMap {
        let mut outbound = outbound_headers(inbound);
        if let UpstreamAuth::ApiKey(_) = &self.auth {
            outbound.remove(header::AUTHORIZATION);
            outbound.remove("x-api-key");
            outbound.extend(self.key_headers());
        }
        outbound
    }

    /// `authorization: Bearer <key>` and `x-api-key: <key>`, marked sensitive.
    /// Empty for the Claude passthrough.
    pub(crate) fn key_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let UpstreamAuth::ApiKey(ApiKey(key)) = &self.auth {
            // `ApiKey::new` checked both values are valid header values.
            if let (Ok(mut bearer), Ok(mut raw)) = (
                HeaderValue::from_str(&format!("Bearer {key}")),
                HeaderValue::from_str(key),
            ) {
                bearer.set_sensitive(true);
                raw.set_sensitive(true);
                headers.insert(header::AUTHORIZATION, bearer);
                headers.insert(HeaderName::from_static("x-api-key"), raw);
            }
        }
        headers
    }

    fn is_claude(&self) -> bool {
        matches!(self.auth, UpstreamAuth::Client)
    }
}

fn build_client(what: &str) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .build()
        .map_err(|error| format!("failed to build {what} HTTP client: {error}"))
}

/// Everything the server needs to route `/v1/messages`.
pub struct BackendSettings {
    pub order: BackendOrder,
    pub failover: FailoverPolicy,
    pub upstream: ClaudeUpstream,
    /// Named Anthropic-compatible upstreams, routed by model.
    pub upstreams: crate::upstream::Upstreams,
}

impl BackendSettings {
    pub fn defaults() -> Self {
        Self {
            order: BackendOrder::codex_only(),
            failover: FailoverPolicy::default(),
            upstream: ClaudeUpstream::from_base_url(None)
                .expect("default Claude upstream is valid"),
            upstreams: crate::upstream::Upstreams::default(),
        }
    }
}

/// Reads `CCP_BACKENDS`, the `backends` key of `config.json`, the failover
/// settings (`CCP_FAILOVER_ON_429`, `CCP_RETRY_CODEX_AFTER_SECS` or the matching
/// config keys), `CCP_CLAUDE_BASE_URL`, and the named upstreams
/// (`CCP_UPSTREAMS_FILE` or the `upstreams` key). Any invalid value is an error.
pub fn load_settings() -> Result<BackendSettings, String> {
    let config_dir = crate::paths::config_dir();
    let env_order = std::env::var("CCP_BACKENDS").ok();
    let file_order = crate::config::file_backends_in(&config_dir);
    let order = order_from_sources(env_order.as_deref(), file_order.as_deref())?;
    let env_on_429 = std::env::var("CCP_FAILOVER_ON_429").ok();
    let env_retry = std::env::var("CCP_RETRY_CODEX_AFTER_SECS").ok();
    let file_failover = crate::config::file_failover_in(&config_dir);
    let failover = failover_policy_from_sources(
        env_on_429.as_deref(),
        file_failover.failover_on_429.as_ref(),
        env_retry.as_deref(),
        file_failover.retry_codex_after_secs.as_ref(),
    )?;
    let upstream =
        ClaudeUpstream::from_base_url(std::env::var("CCP_CLAUDE_BASE_URL").ok().as_deref())?;
    let upstreams = crate::upstream::load(
        std::env::var_os("CCP_UPSTREAMS_FILE").as_deref(),
        crate::config::file_upstreams_in(&config_dir).as_ref(),
    )?;
    Ok(BackendSettings {
        order,
        failover,
        upstream,
        upstreams,
    })
}

/// Claude passthrough needs the caller's own credentials, either an
/// `authorization` or an `x-api-key` header. The proxy never supplies a token
/// of its own.
pub fn has_passthrough_credentials(headers: &HeaderMap) -> bool {
    headers.contains_key(header::AUTHORIZATION) || headers.contains_key("x-api-key")
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
    backend: Arc<str>,
    claude: bool,
    model: Option<String>,
    status: StatusCode,
    started_at: Instant,
    bytes: u64,
}

impl Drop for PassthroughSummary {
    fn drop(&mut self) {
        self.log.info(
            if self.claude {
                "claude passthrough completed"
            } else {
                "upstream passthrough completed"
            },
            Some(Map::from_iter([
                ("reqId".to_string(), json!(&self.req_id)),
                ("backend".to_string(), json!(&*self.backend)),
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

/// One request forwarded to an Anthropic-compatible upstream.
pub struct ForwardRequest<'a> {
    pub req_id: &'a str,
    /// Model sent upstream, for the completion log line.
    pub model: Option<&'a str>,
    pub headers: &'a HeaderMap,
    pub query: Option<&'a str>,
    pub body: Bytes,
    /// Model id the client asked for, when it differs from the one sent
    /// upstream (a `<upstream>/` prefix was stripped). A successful response
    /// gets it back in the JSON `model` field, or in `message.model` of the
    /// streamed `message_start` event; nothing else is changed.
    pub restore_model: Option<String>,
}

/// Sends the original `/v1/messages` request bytes to Anthropic and streams the
/// response back with the upstream status and content-type.
///
/// What the Claude path does NOT do, compared with the Codex path in
/// `server::dispatch_request`:
/// - no model rewriting: the body bytes and the `model` field go out verbatim.
///   `registry::normalize_incoming_model` (the `[1m]` suffix strip) is skipped,
///   and so is the auto-review model override (`apply_auto_review_model`).
/// - no Codex-specific translation (`provider.handle_*`, the Codex request
///   builder, session compaction, conversation identity, traffic capture).
/// - no monitor provider label: `provider_selected` and `project_resolved` are
///   never called; the monitor only sees the passthrough status and bytes.
/// - no `thread` beta rejection and no "missing model" check; Anthropic
///   answers those itself.
pub async fn forward_messages(
    upstream: &ClaudeUpstream,
    req_id: &str,
    model: Option<&str>,
    headers: &HeaderMap,
    query: Option<&str>,
    body: Bytes,
) -> Response {
    forward(
        upstream,
        false,
        ForwardRequest {
            req_id,
            model,
            headers,
            query,
            body,
            restore_model: None,
        },
    )
    .await
}

/// Same passthrough as `forward_messages`, to `/v1/messages/count_tokens`.
pub async fn forward_count_tokens(
    upstream: &ClaudeUpstream,
    req_id: &str,
    model: Option<&str>,
    headers: &HeaderMap,
    query: Option<&str>,
    body: Bytes,
) -> Response {
    forward(
        upstream,
        true,
        ForwardRequest {
            req_id,
            model,
            headers,
            query,
            body,
            restore_model: None,
        },
    )
    .await
}

/// Forwards to `/v1/messages` or, with `count_tokens`, to
/// `/v1/messages/count_tokens` of any Anthropic-compatible upstream, streaming
/// the response back as it arrives.
pub async fn forward(
    upstream: &AnthropicUpstream,
    count_tokens: bool,
    request: ForwardRequest<'_>,
) -> Response {
    let path = if count_tokens {
        COUNT_TOKENS_PATH
    } else {
        MESSAGES_PATH
    };
    if upstream.is_claude() && !has_passthrough_credentials(request.headers) {
        return json_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "Claude passthrough needs the authorization or x-api-key header Claude Code sends; none was received",
        );
    }
    let started_at = Instant::now();
    let sent = upstream
        .client
        .post(upstream.endpoint_url(path, request.query))
        .headers(upstream.outbound_headers(request.headers))
        .body(request.body)
        .send()
        .await;
    let upstream_response = match sent {
        Ok(response) => response,
        Err(error) => {
            create_logger("backend").warn(
                if upstream.is_claude() {
                    "claude passthrough failed"
                } else {
                    "upstream passthrough failed"
                },
                Some(Map::from_iter([
                    ("reqId".to_string(), json!(request.req_id)),
                    ("backend".to_string(), json!(upstream.label())),
                    ("error".to_string(), json!(error.to_string())),
                ])),
            );
            let message = if upstream.is_claude() {
                "Claude upstream request failed".to_string()
            } else {
                format!("Upstream \"{}\" request failed", upstream.label())
            };
            return json_error(StatusCode::BAD_GATEWAY, "api_error", message);
        }
    };
    let status = upstream_response.status();
    let content_type = upstream_response
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned();
    let mut summary = PassthroughSummary {
        log: create_logger("backend"),
        req_id: request.req_id.to_string(),
        backend: upstream.label.clone(),
        claude: upstream.is_claude(),
        model: request.model.map(str::to_string),
        status,
        started_at,
        bytes: 0,
    };
    let media_type = content_type
        .as_ref()
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.split(';').next())
        .map(|media_type| media_type.trim().to_ascii_lowercase());
    let restore_model = request.restore_model.filter(|_| status.is_success());
    let mut rewriter = None;
    let body = match (restore_model, media_type.as_deref()) {
        (Some(model), Some("application/json")) => {
            // A non-streamed answer is one JSON document: read it whole.
            let bytes = match upstream_response.bytes().await {
                Ok(bytes) => bytes,
                Err(error) => {
                    create_logger("backend").warn(
                        "upstream response read failed",
                        Some(Map::from_iter([
                            ("reqId".to_string(), json!(request.req_id)),
                            ("backend".to_string(), json!(upstream.label())),
                            ("error".to_string(), json!(error.to_string())),
                        ])),
                    );
                    return json_error(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        format!("Upstream \"{}\" response failed", upstream.label()),
                    );
                }
            };
            summary.bytes = bytes.len() as u64;
            Body::from(crate::upstream::restore_model_in_json(&bytes, &model).unwrap_or(bytes))
        }
        (restore_model, media_type) => {
            if media_type == Some("text/event-stream") {
                rewriter = restore_model.map(crate::upstream::SseModelRewriter::new);
            }
            Body::from_stream(stream_body(
                upstream_response.bytes_stream(),
                summary,
                rewriter,
            ))
        }
    };
    let mut response = Response::new(body);
    *response.status_mut() = status;
    if let Some(content_type) = content_type {
        response
            .headers_mut()
            .insert(header::CONTENT_TYPE, content_type);
    }
    response
}

/// Streams the upstream body through, counting bytes for the summary line.
/// With a rewriter, each chunk goes through it so `message_start` carries the
/// requested model; every other byte passes through as soon as its event ends.
fn stream_body<S>(
    chunks: S,
    summary: PassthroughSummary,
    rewriter: Option<crate::upstream::SseModelRewriter>,
) -> impl futures_util::Stream<Item = Result<Bytes, reqwest::Error>>
where
    S: futures_util::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    futures_util::stream::unfold(
        (chunks, summary, rewriter, false),
        |(mut chunks, mut summary, mut rewriter, ended)| async move {
            if ended {
                return None;
            }
            loop {
                match chunks.next().await {
                    Some(Ok(bytes)) => {
                        summary.bytes += bytes.len() as u64;
                        let out = match rewriter.as_mut() {
                            Some(rewriter) => Bytes::from(rewriter.push(&bytes)),
                            None => bytes,
                        };
                        if out.is_empty() {
                            continue;
                        }
                        return Some((Ok(out), (chunks, summary, rewriter, false)));
                    }
                    Some(Err(error)) => {
                        return Some((Err(error), (chunks, summary, rewriter, false)));
                    }
                    None => {
                        let rest = rewriter
                            .as_mut()
                            .map(crate::upstream::SseModelRewriter::finish)
                            .unwrap_or_default();
                        if rest.is_empty() {
                            return None;
                        }
                        return Some((Ok(Bytes::from(rest)), (chunks, summary, rewriter, true)));
                    }
                }
            }
        },
    )
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
    match selector.switch_manual(target.clone()) {
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
        Err(ManualSwitchError::NotConfigured(message)) => no_store(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        )),
        Err(ManualSwitchError::SessionsNotAllowed(message)) => no_store(json_error(
            StatusCode::CONFLICT,
            "permission_error",
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

/// `POST /control/sessions/{id}/allowed` with `{"backends": ["claude"]}`.
/// Replaces the session's allowed backends and answers with the session entry.
async fn post_session_allowed(
    State(selector): State<Arc<BackendSelector>>,
    axum::extract::Path(session_id): axum::extract::Path<String>,
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
    let bad_body = || {
        no_store(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Body must be a JSON object with a \"backends\" array of backend names",
        ))
    };
    let Some(names) = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|value| value.get("backends").and_then(Value::as_array).cloned())
    else {
        return bad_body();
    };
    if names.is_empty() {
        return no_store(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "\"backends\" must name at least one backend",
        ));
    }
    let mut backends = Vec::with_capacity(names.len());
    for name in &names {
        let Some(name) = name.as_str() else {
            return bad_body();
        };
        match selector.parse_backend(name) {
            Ok(backend) => backends.push(backend),
            Err(message) => {
                return no_store(json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    message,
                ));
            }
        }
    }
    let allowed = match BackendOrder::new(backends) {
        Ok(allowed) => allowed,
        Err(message) => {
            return no_store(json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                message,
            ));
        }
    };
    match selector.set_session_allowed(&session_id, &allowed) {
        Ok(Some(record)) => {
            create_logger("backend").info(
                "session allowed backends set",
                Some(Map::from_iter([(
                    "allowed".to_string(),
                    json!(allowed.names()),
                )])),
            );
            json_no_store(
                StatusCode::OK,
                session_entry_json(Some(&session_id), &record),
            )
        }
        Ok(None) => no_store(json_error(
            StatusCode::NOT_FOUND,
            "not_found_error",
            "Unknown session",
        )),
        Err(message) => no_store(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
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
        .route("/control/sessions/{id}/allowed", post(post_session_allowed))
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
        let switched =
            rate_limit_failover(&start, &both, Some("42".to_string()), T1, None).unwrap();
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
        assert!(rate_limit_failover(&start, &codex_only, None, T1, None).is_none());

        let both = order(vec![Backend::Codex, Backend::Claude]);
        let claude = manual_switch(&start, &both, Backend::Claude, T1).unwrap();
        assert!(rate_limit_failover(&claude, &both, None, T1, None).is_none());
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

        let both_backends = [Backend::Codex, Backend::Claude];
        assert!(
            selector
                .fail_over_on_429(Some("5".to_string()), &both_backends)
                .is_some()
        );
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
    fn failover_settings_come_from_env_then_config_and_reject_garbage() {
        let defaults = failover_policy_from_sources(None, None, None, None).unwrap();
        assert_eq!(defaults, FailoverPolicy::default());
        assert!(defaults.on_429);
        assert_eq!(defaults.retry_codex_after_secs, None);

        let from_env = failover_policy_from_sources(Some("off"), None, Some("300"), None).unwrap();
        assert!(!from_env.on_429);
        assert_eq!(from_env.retry_codex_after_secs, Some(300));

        let file_off = Value::Bool(false);
        let file_secs = json!(60);
        let from_file =
            failover_policy_from_sources(None, Some(&file_off), None, Some(&file_secs)).unwrap();
        assert!(!from_file.on_429);
        assert_eq!(from_file.retry_codex_after_secs, Some(60));

        let env_wins = failover_policy_from_sources(Some("true"), Some(&file_off), None, None);
        assert!(env_wins.unwrap().on_429);

        assert!(failover_policy_from_sources(Some("maybe"), None, None, None).is_err());
        assert!(failover_policy_from_sources(None, None, Some("-1"), None).is_err());
        assert!(failover_policy_from_sources(None, None, Some("soon"), None).is_err());
        assert!(failover_policy_from_sources(None, None, Some("86401"), None).is_err());
        assert!(failover_policy_from_sources(None, None, None, Some(&json!(-5))).is_err());
        assert!(failover_policy_from_sources(None, None, None, Some(&json!("60"))).is_err());
        assert!(failover_policy_from_sources(None, Some(&json!("yes")), None, None).is_err());
        assert_eq!(
            failover_policy_from_sources(None, None, Some("86400"), None)
                .unwrap()
                .retry_codex_after_secs,
            Some(MAX_RETRY_CODEX_AFTER_SECS)
        );
    }

    #[test]
    fn codex_retry_is_scheduled_on_429_and_due_only_after_the_delay() {
        let both = order(vec![Backend::Codex, Backend::Claude]);
        let start = startup_state(&both, T0);
        let switched = rate_limit_failover(&start, &both, None, T1, Some(600)).unwrap();
        assert_eq!(
            switched.next_check.as_deref(),
            Some("2026-11-16T10:15:00Z"),
            "next_check is the switch time plus the delay"
        );
        assert!(codex_retry_due(&switched, &both, "2026-11-16T10:14:59Z").is_none());

        let back = codex_retry_due(&switched, &both, "2026-11-16T10:15:00Z").unwrap();
        assert_eq!(back.current, Backend::Codex);
        assert_eq!(back.last_switch_reason, SwitchReason::RetryCodex);
        assert_eq!(back.next_check, None);
    }

    #[test]
    fn no_codex_retry_without_a_schedule_or_for_manual_switches() {
        let both = order(vec![Backend::Codex, Backend::Claude]);
        let start = startup_state(&both, T0);
        let unscheduled = rate_limit_failover(&start, &both, None, T1, None).unwrap();
        assert_eq!(unscheduled.next_check, None);
        assert!(codex_retry_due(&unscheduled, &both, "2030-01-01T00:00:00Z").is_none());

        let scheduled = rate_limit_failover(&start, &both, None, T1, Some(0)).unwrap();
        let manual = manual_switch(&scheduled, &both, Backend::Claude, T1).unwrap();
        assert_eq!(
            manual.next_check, None,
            "a manual switch drops the schedule"
        );
        assert!(codex_retry_due(&manual, &both, "2030-01-01T00:00:00Z").is_none());
    }

    #[test]
    fn selector_retries_codex_once_the_scheduled_delay_has_passed() {
        let selector = BackendSelector::with_failover(
            order(vec![Backend::Codex, Backend::Claude]),
            FailoverPolicy {
                on_429: true,
                retry_codex_after_secs: Some(0),
            },
        );
        let both = [Backend::Codex, Backend::Claude];
        assert!(selector.fail_over_on_429(None, &both).is_some());
        assert!(selector.snapshot().next_check.is_some());
        let back = selector.retry_codex_if_due().unwrap();
        assert_eq!(back.current, Backend::Codex);
        assert_eq!(back.last_switch_reason, SwitchReason::RetryCodex);
        assert!(selector.retry_codex_if_due().is_none());
    }

    #[test]
    fn session_allowed_defaults_to_order_and_can_be_narrowed() {
        let both = order(vec![Backend::Codex, Backend::Claude]);
        let selector = BackendSelector::new(both.clone());
        assert_eq!(
            selector.allowed_for(Some("s1")),
            vec![Backend::Codex, Backend::Claude]
        );

        selector.record_session("s1", Backend::Codex, Some("gpt-5.5"));
        assert_eq!(
            selector.session("s1").unwrap().allowed,
            vec![Backend::Codex, Backend::Claude]
        );

        let claude_only = BackendOrder::new(vec![Backend::Claude]).unwrap();
        let updated = selector
            .set_session_allowed("s1", &claude_only)
            .unwrap()
            .unwrap();
        assert_eq!(updated.allowed, vec![Backend::Claude]);
        selector.record_session("s1", Backend::Claude, None);
        assert_eq!(selector.allowed_for(Some("s1")), vec![Backend::Claude]);
        assert!(
            selector
                .set_session_allowed("nope", &claude_only)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn session_allowed_rejects_backends_outside_the_order() {
        let selector = BackendSelector::new(BackendOrder::codex_only());
        selector.record_session("s1", Backend::Codex, None);
        let claude = BackendOrder::new(vec![Backend::Claude]).unwrap();
        assert!(selector.set_session_allowed("s1", &claude).is_err());
    }

    #[test]
    fn manual_switch_is_refused_when_a_session_may_not_use_the_target() {
        let both = order(vec![Backend::Codex, Backend::Claude]);
        let selector = BackendSelector::new(both);
        selector.record_session("codex-only", Backend::Codex, None);
        let codex = BackendOrder::new(vec![Backend::Codex]).unwrap();
        selector.set_session_allowed("codex-only", &codex).unwrap();

        match selector.switch_manual(Backend::Claude) {
            Err(ManualSwitchError::SessionsNotAllowed(message)) => {
                assert!(message.contains("codex-only"));
            }
            other => panic!("expected a refused switch, got {other:?}"),
        }
        assert_eq!(selector.snapshot().current, Backend::Codex);

        selector.record_session("free", Backend::Codex, None);
        assert!(selector.switch_manual(Backend::Claude).is_err());
    }

    #[test]
    fn x_api_key_counts_as_passthrough_credentials() {
        let mut headers = HeaderMap::new();
        assert!(!has_passthrough_credentials(&headers));
        headers.insert("x-api-key", HeaderValue::from_static("key"));
        assert!(has_passthrough_credentials(&headers));
        assert!(outbound_headers(&headers).contains_key("x-api-key"));
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
