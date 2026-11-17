//! Named Anthropic-compatible upstreams.
//!
//! An upstream is any gateway that speaks the Anthropic Messages API (for
//! example a LiteLLM deployment). Each one is configured with a name, a base
//! URL, a key file and, optionally, the models it serves. Requests reach an
//! upstream by their `model` field, so Claude Code switches to one with
//! `/model` and no restart:
//!
//! - `<name>/<model>` routes to upstream `<name>`; the prefix is stripped from
//!   the forwarded `model` field and put back in the response.
//! - a bare model listed by exactly one upstream, and not served by a built-in
//!   provider, routes to that upstream unchanged.
//!
//! The client's `authorization` and `x-api-key` never reach an upstream; the
//! upstream's own key is sent instead (see `AnthropicUpstream::outbound_headers`).

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::backend::{AnthropicUpstream, ApiKey, Backend, validate_base_url};
use crate::logging::create_logger;

/// Shortest time between two `/v1/models` fetches of one upstream.
pub const DISCOVERY_INTERVAL: Duration = Duration::from_secs(600);

/// Bound for one discovery request, connection included.
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(30);

/// Bytes an SSE stream may hold back while looking for `message_start`. Past
/// this the stream is passed through untouched.
const MAX_SSE_LOOKAHEAD: usize = 1 << 20;

/// Names an upstream may not take: the selector's backends and the prefix the
/// opencode provider already claims.
const RESERVED_NAMES: &[&str] = &["claude", "codex", "opencode-go"];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpstreamConfig {
    name: String,
    base_url: String,
    api_key_file: PathBuf,
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    discover_models: bool,
}

/// One configured upstream with its model lists.
pub struct NamedUpstream {
    name: Arc<str>,
    http: AnthropicUpstream,
    models: Vec<String>,
    discover: bool,
    /// Last good `/v1/models` answer.
    discovered: RwLock<Option<Vec<String>>>,
    /// Time of the last discovery attempt, good or not. The async lock also
    /// keeps two fetches of one upstream from running at once.
    last_attempt: tokio::sync::Mutex<Option<Instant>>,
}

impl NamedUpstream {
    pub fn name(&self) -> &Arc<str> {
        &self.name
    }

    pub fn backend(&self) -> Backend {
        Backend::Upstream(self.name.clone())
    }

    pub fn http(&self) -> &AnthropicUpstream {
        &self.http
    }

    /// Configured models, then discovered ones, without duplicates.
    pub fn known_models(&self) -> Vec<String> {
        let mut models = self.models.clone();
        let discovered = self
            .discovered
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for model in discovered.iter().flatten() {
            if !models.contains(model) {
                models.push(model.clone());
            }
        }
        models
    }

    fn serves(&self, model: &str) -> bool {
        self.models.iter().any(|candidate| candidate == model)
            || self
                .discovered
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .flatten()
                .any(|candidate| candidate == model)
    }

    /// Fetches `/v1/models` when discovery is on and the last attempt is at
    /// least `interval` old. A failure keeps the last good list.
    async fn refresh_if_stale(&self, interval: Duration) {
        if !self.discover {
            return;
        }
        let mut last_attempt = self.last_attempt.lock().await;
        if last_attempt.is_some_and(|at| at.elapsed() < interval) {
            return;
        }
        *last_attempt = Some(Instant::now());
        let log = create_logger("upstream");
        match self.fetch_models().await {
            Ok(models) => {
                log.info(
                    "upstream models discovered",
                    Some(Map::from_iter([
                        ("upstream".to_string(), json!(&*self.name)),
                        ("count".to_string(), json!(models.len())),
                    ])),
                );
                *self
                    .discovered
                    .write()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(models);
            }
            Err(error) => log.warn(
                "upstream model discovery failed; keeping the last good list",
                Some(Map::from_iter([
                    ("upstream".to_string(), json!(&*self.name)),
                    ("error".to_string(), json!(error)),
                ])),
            ),
        }
    }

    async fn fetch_models(&self) -> Result<Vec<String>, String> {
        let response = self
            .http
            .client()
            .get(self.http.endpoint_url("/v1/models", None))
            .headers(self.http.key_headers())
            .header("anthropic-version", "2023-06-01")
            .timeout(DISCOVERY_TIMEOUT)
            .send()
            .await
            .map_err(|error| format!("request failed: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("HTTP {}", status.as_u16()));
        }
        let value: Value = response
            .json()
            .await
            .map_err(|error| format!("invalid JSON: {error}"))?;
        let data = value
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| "response has no \"data\" array".to_string())?;
        let mut models = Vec::new();
        for id in data
            .iter()
            .filter_map(|entry| entry.get("id").and_then(Value::as_str))
        {
            if !id.is_empty() && !models.iter().any(|model| model == id) {
                models.push(id.to_string());
            }
        }
        Ok(models)
    }
}

/// Where a request goes when it names an upstream model.
pub struct UpstreamRoute {
    pub upstream: Arc<NamedUpstream>,
    /// Model to send when the request used `<name>/<model>`: `<model>`. `None`
    /// for a bare model, which is forwarded unchanged.
    pub forwarded_model: Option<String>,
}

/// Every configured upstream, in config order. Empty by default.
#[derive(Clone)]
pub struct Upstreams {
    list: Vec<Arc<NamedUpstream>>,
    interval: Duration,
}

impl Default for Upstreams {
    fn default() -> Self {
        Self {
            list: Vec::new(),
            interval: DISCOVERY_INTERVAL,
        }
    }
}

impl Upstreams {
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn names(&self) -> Vec<Arc<str>> {
        self.list
            .iter()
            .map(|upstream| upstream.name.clone())
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<&Arc<NamedUpstream>> {
        self.list.iter().find(|upstream| &*upstream.name == name)
    }

    /// Replaces the ten-minute minimum between discovery fetches (tests).
    pub fn with_discovery_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Routes `model` to an upstream. `claimed` says whether a built-in proxy
    /// provider serves a bare model name; such a name stays with the provider
    /// and needs the `<name>/` prefix to reach an upstream.
    pub fn route(&self, model: &str, claimed: impl Fn(&str) -> bool) -> Option<UpstreamRoute> {
        if let Some((prefix, rest)) = model.split_once('/')
            && !rest.is_empty()
            && let Some(upstream) = self.get(prefix)
        {
            return Some(UpstreamRoute {
                upstream: upstream.clone(),
                forwarded_model: Some(rest.to_string()),
            });
        }
        self.bare_owner(model, &claimed)
            .map(|upstream| UpstreamRoute {
                upstream: upstream.clone(),
                forwarded_model: None,
            })
    }

    fn bare_owner(
        &self,
        model: &str,
        claimed: &impl Fn(&str) -> bool,
    ) -> Option<&Arc<NamedUpstream>> {
        let mut owners = self.list.iter().filter(|upstream| upstream.serves(model));
        let owner = owners.next()?;
        if owners.next().is_some() || claimed(model) {
            return None;
        }
        Some(owner)
    }

    /// `/v1/models` entries as `(id, upstream name)`: every known model as
    /// `<name>/<model>`, then the bare names that `route` sends to an upstream.
    pub fn model_entries(&self, claimed: impl Fn(&str) -> bool) -> Vec<(String, Arc<str>)> {
        let mut entries = Vec::new();
        let mut bare = Vec::new();
        for upstream in &self.list {
            for model in upstream.known_models() {
                entries.push((format!("{}/{model}", upstream.name), upstream.name.clone()));
                let owned_here = self
                    .bare_owner(&model, &claimed)
                    .is_some_and(|owner| Arc::ptr_eq(owner, upstream));
                if owned_here {
                    bare.push((model, upstream.name.clone()));
                }
            }
        }
        entries.extend(bare);
        entries
    }

    /// Refreshes every discovering upstream whose list is older than the
    /// discovery interval. Called lazily from `/v1/models`.
    pub async fn refresh_stale(&self) {
        let refreshes = self
            .list
            .iter()
            .map(|upstream| upstream.refresh_if_stale(self.interval));
        futures_util::future::join_all(refreshes).await;
    }

    /// Starts the startup discovery fetch in the background. Does nothing
    /// outside a Tokio runtime; `/v1/models` fetches lazily then.
    pub fn spawn_discovery(&self) {
        if !self.list.iter().any(|upstream| upstream.discover) {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let upstreams = self.clone();
        runtime.spawn(async move { upstreams.refresh_stale().await });
    }
}

/// Reads the upstreams from the file named by `CCP_UPSTREAMS_FILE` when set,
/// else from the `upstreams` key of `config.json`. Neither gives an empty set.
pub fn load(env_file: Option<&OsStr>, file_value: Option<&Value>) -> Result<Upstreams, String> {
    if let Some(path) = env_file {
        let path = Path::new(path);
        let raw = std::fs::read_to_string(path).map_err(|error| {
            format!(
                "CCP_UPSTREAMS_FILE {}: cannot be read: {error}",
                path.display()
            )
        })?;
        let value: Value = serde_json::from_str(&raw).map_err(|error| {
            format!(
                "CCP_UPSTREAMS_FILE {}: invalid JSON: {error}",
                path.display()
            )
        })?;
        return parse(&value, "CCP_UPSTREAMS_FILE");
    }
    match file_value {
        Some(value) => parse(value, "config.json upstreams"),
        None => Ok(Upstreams::default()),
    }
}

/// Validates an `upstreams` array and reads every key file. `source` names the
/// setting in errors. Errors name files, never key contents.
pub fn parse(value: &Value, source: &str) -> Result<Upstreams, String> {
    let entries = value
        .as_array()
        .ok_or_else(|| format!("{source} must be an array"))?;
    let mut list: Vec<Arc<NamedUpstream>> = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let config: UpstreamConfig = serde_json::from_value(entry.clone())
            .map_err(|error| format!("{source}[{index}]: {error}"))?;
        let at = format!("{source}[{index}]");
        check_name(&config.name).map_err(|error| format!("{at}: {error}"))?;
        if list.iter().any(|upstream| *upstream.name == *config.name) {
            return Err(format!(
                "{at}: upstream name \"{}\" is used more than once",
                config.name
            ));
        }
        let at = format!("{at} ({})", config.name);
        let base_url = validate_base_url(&config.base_url, &format!("{at} base_url"))?;
        for model in &config.models {
            if model.trim().is_empty() {
                return Err(format!("{at}: models must be non-empty strings"));
            }
        }
        let key = read_key(&config.name, &config.api_key_file)
            .map_err(|error| format!("{at} api_key_file: {error}"))?;
        let name: Arc<str> = Arc::from(config.name.as_str());
        let http = AnthropicUpstream::with_api_key(name.clone(), base_url, key)?;
        list.push(Arc::new(NamedUpstream {
            name,
            http,
            models: config.models,
            discover: config.discover_models,
            discovered: RwLock::new(None),
            last_attempt: tokio::sync::Mutex::new(None),
        }));
    }
    Ok(Upstreams {
        list,
        interval: DISCOVERY_INTERVAL,
    })
}

fn check_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("upstream name is empty".to_string());
    }
    if !name
        .bytes()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(format!(
            "upstream name \"{name}\" must only use a-z, 0-9 and -"
        ));
    }
    if RESERVED_NAMES.contains(&name) || crate::registry::BUILTIN_PROVIDERS.contains(&name) {
        return Err(format!(
            "upstream name \"{name}\" is already a backend or provider name"
        ));
    }
    Ok(())
}

fn read_key(name: &str, path: &Path) -> Result<ApiKey, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|error| format!("{} cannot be read: {error}", path.display()))?;
    warn_if_shared(name, path);
    ApiKey::new(&raw).map_err(|error| format!("{}: {error}", path.display()))
}

#[cfg(unix)]
fn warn_if_shared(name: &str, path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let Ok(metadata) = std::fs::metadata(path) else {
        return;
    };
    if metadata.permissions().mode() & 0o044 != 0 {
        create_logger("upstream").warn(
            "upstream key file is readable by group or others",
            Some(Map::from_iter([
                ("upstream".to_string(), json!(name)),
                ("path".to_string(), json!(path.display().to_string())),
            ])),
        );
    }
}

#[cfg(not(unix))]
fn warn_if_shared(_name: &str, _path: &Path) {}

/// Request body with only `model` replaced. The body is parsed and serialized
/// again, so the JSON keeps its values but not its exact bytes: whitespace goes
/// and object keys come out sorted (serde_json is built without
/// `preserve_order`). `None` when the body is not a JSON object.
pub fn rewrite_model_field(body: &[u8], model: &str) -> Option<Bytes> {
    let mut value: Value = serde_json::from_slice(body).ok()?;
    value
        .as_object_mut()?
        .insert("model".to_string(), json!(model));
    serde_json::to_vec(&value).ok().map(Bytes::from)
}

/// Non-streamed response with its top-level `model` set to `model`. `None`
/// when the body is not a JSON object with a `model` key, so it is passed on
/// untouched.
pub fn restore_model_in_json(body: &[u8], model: &str) -> Option<Bytes> {
    let mut value: Value = serde_json::from_slice(body).ok()?;
    let object = value.as_object_mut()?;
    if !object.contains_key("model") {
        return None;
    }
    object.insert("model".to_string(), json!(model));
    serde_json::to_vec(&value).ok().map(Bytes::from)
}

/// Puts the requested model back into `message.model` of the streamed
/// `message_start` event. Holds back bytes only until each event is complete;
/// once `message_start` has gone through (or the lookahead limit is hit), every
/// byte passes through as it arrives. Other events are never changed.
pub struct SseModelRewriter {
    model: String,
    pending: Vec<u8>,
    done: bool,
}

impl SseModelRewriter {
    pub fn new(model: String) -> Self {
        Self {
            model,
            pending: Vec::new(),
            done: false,
        }
    }

    /// Bytes ready to send after `chunk`.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<u8> {
        if self.done {
            return chunk.to_vec();
        }
        self.pending.extend_from_slice(chunk);
        let mut out = Vec::new();
        while !self.done {
            let Some(end) = event_end(&self.pending) else {
                break;
            };
            let event: Vec<u8> = self.pending.drain(..end).collect();
            out.extend(self.rewrite_event(event));
        }
        if self.done || self.pending.len() > MAX_SSE_LOOKAHEAD {
            self.done = true;
            out.append(&mut self.pending);
        }
        out
    }

    /// Whatever is still held back when the stream ends.
    pub fn finish(&mut self) -> Vec<u8> {
        self.done = true;
        std::mem::take(&mut self.pending)
    }

    fn rewrite_event(&mut self, event: Vec<u8>) -> Vec<u8> {
        let Ok(text) = std::str::from_utf8(&event) else {
            return event;
        };
        let mut out = String::with_capacity(text.len() + self.model.len());
        let mut changed = false;
        for line in text.split_inclusive('\n') {
            let content = line.trim_end_matches(['\r', '\n']);
            let ending = &line[content.len()..];
            if !changed
                && let Some(payload) = content.strip_prefix("data:")
                && let Ok(mut value) = serde_json::from_str::<Value>(payload.trim_start())
                && value.get("type").and_then(Value::as_str) == Some("message_start")
            {
                self.done = true;
                if let Some(message) = value.get_mut("message").and_then(Value::as_object_mut)
                    && message.contains_key("model")
                {
                    message.insert("model".to_string(), json!(self.model));
                    out.push_str("data: ");
                    out.push_str(&value.to_string());
                    out.push_str(ending);
                    changed = true;
                    continue;
                }
            }
            out.push_str(line);
        }
        if changed { out.into_bytes() } else { event }
    }
}

/// End of the first complete SSE event (blank line included), if any.
fn event_end(buffer: &[u8]) -> Option<usize> {
    let lf = find(buffer, b"\n\n").map(|at| at + 2);
    let crlf = find(buffer, b"\r\n\r\n").map(|at| at + 4);
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AliasProvider;
    use crate::registry::{Registry, normalize_incoming_model};

    fn key_file(dir: &Path) -> String {
        let path = dir.join("key");
        std::fs::write(&path, "test-key\n").unwrap();
        path.display().to_string()
    }

    fn parse_one(entry: Value) -> Result<Upstreams, String> {
        parse(&json!([entry]), "upstreams")
    }

    fn claimed_by(registry: &Registry) -> impl Fn(&str) -> bool + '_ {
        |model| {
            registry
                .provider_for_model(&normalize_incoming_model(model), None)
                .is_some()
        }
    }

    #[test]
    fn rejects_bad_names_collisions_and_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let key = key_file(dir.path());
        for name in [
            "",
            "Igalia",
            "ig alia",
            "ig/alia",
            "ig_alia",
            "claude",
            "codex",
            "kimi",
            "opencode",
            "opencode-go",
            "cursor",
            "grok",
        ] {
            let result = parse_one(json!({
                "name": name, "base_url": "https://ai.example", "api_key_file": key,
            }));
            assert!(result.is_err(), "name {name:?} must be rejected");
        }
        let entry =
            json!({"name": "igalia", "base_url": "https://ai.example", "api_key_file": key});
        let error = parse(&json!([entry.clone(), entry]), "upstreams")
            .err()
            .unwrap();
        assert!(error.contains("more than once"), "{error}");
        assert!(parse(&json!({"name": "x"}), "upstreams").is_err());
        assert!(
            parse_one(
                json!({"name": "a", "base_url": "https://a", "api_key_file": key, "extra": 1})
            )
            .is_err(),
            "unknown keys are rejected"
        );
    }

    #[test]
    fn requires_https_or_loopback_http_and_a_readable_key() {
        let dir = tempfile::tempdir().unwrap();
        let key = key_file(dir.path());
        for base_url in [
            "http://ai.example",
            "http://localhost:9000",
            "ftp://127.0.0.1",
        ] {
            let result = parse_one(json!({"name": "a", "base_url": base_url, "api_key_file": key}));
            assert!(result.is_err(), "{base_url} must be rejected");
        }
        for base_url in [
            "https://ai.example",
            "http://127.0.0.1:9000",
            "http://[::1]:9000",
        ] {
            let result = parse_one(json!({"name": "a", "base_url": base_url, "api_key_file": key}));
            assert!(result.is_ok(), "{base_url} must be accepted");
        }
        let missing = dir.path().join("missing");
        let error = parse_one(json!({
            "name": "a", "base_url": "https://ai.example", "api_key_file": missing,
        }))
        .err()
        .unwrap();
        assert!(error.contains("api_key_file"), "{error}");

        let empty = dir.path().join("empty");
        std::fs::write(&empty, "\n").unwrap();
        assert!(
            parse_one(
                json!({"name": "a", "base_url": "https://ai.example", "api_key_file": empty})
            )
            .is_err()
        );
    }

    #[test]
    fn errors_and_debug_never_show_the_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key");
        std::fs::write(&path, "very-secret\u{7f}value").unwrap();
        let error = parse_one(json!({
            "name": "a", "base_url": "https://ai.example", "api_key_file": path,
        }))
        .err()
        .unwrap();
        assert!(!error.contains("very-secret"), "{error}");
        let key = ApiKey::new("very-secret").unwrap();
        assert!(!format!("{key:?}").contains("very-secret"));
    }

    #[test]
    fn env_file_wins_over_config_key() {
        let dir = tempfile::tempdir().unwrap();
        let key = key_file(dir.path());
        let env_path = dir.path().join("upstreams.json");
        std::fs::write(
            &env_path,
            json!([{"name": "from-env", "base_url": "https://a", "api_key_file": key}]).to_string(),
        )
        .unwrap();
        let file_value =
            json!([{"name": "from-file", "base_url": "https://b", "api_key_file": key}]);
        let loaded = load(Some(env_path.as_os_str()), Some(&file_value)).unwrap();
        assert_eq!(loaded.names(), vec![Arc::<str>::from("from-env")]);
        let loaded = load(None, Some(&file_value)).unwrap();
        assert_eq!(loaded.names(), vec![Arc::<str>::from("from-file")]);
        assert!(load(None, None).unwrap().is_empty());
        assert!(load(Some(OsStr::new("/nonexistent/upstreams.json")), None).is_err());
    }

    #[test]
    fn bare_name_served_by_a_builtin_provider_keeps_the_provider() {
        let dir = tempfile::tempdir().unwrap();
        let key = key_file(dir.path());
        let upstreams = parse_one(json!({
            "name": "igalia", "base_url": "https://ai.example", "api_key_file": key,
            "models": ["kimi-k3", "glm-5p3"],
        }))
        .unwrap();
        let registry = Registry::new(AliasProvider::Codex);
        let claimed = claimed_by(&registry);

        assert!(
            upstreams.route("kimi-k3", &claimed).is_none(),
            "kimi-k3 stays with the built-in kimi provider"
        );
        assert_eq!(
            registry.provider_for_model("kimi-k3", None).unwrap().name(),
            "kimi"
        );
        let prefixed = upstreams.route("igalia/kimi-k3", &claimed).unwrap();
        assert_eq!(&**prefixed.upstream.name(), "igalia");
        assert_eq!(prefixed.forwarded_model.as_deref(), Some("kimi-k3"));

        let bare = upstreams.route("glm-5p3", &claimed).unwrap();
        assert_eq!(bare.forwarded_model, None);
        assert!(upstreams.route("claude-opus-5", &claimed).is_none());
        assert!(upstreams.route("other/glm-5p3", &claimed).is_none());
        assert!(upstreams.route("igalia/", &claimed).is_none());

        let ids: Vec<String> = upstreams
            .model_entries(&claimed)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(ids, ["igalia/kimi-k3", "igalia/glm-5p3", "glm-5p3"]);
    }

    #[test]
    fn bare_name_listed_by_two_upstreams_needs_a_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let key = key_file(dir.path());
        let upstreams = parse(
            &json!([
                {"name": "one", "base_url": "https://a", "api_key_file": key, "models": ["shared"]},
                {"name": "two", "base_url": "https://b", "api_key_file": key, "models": ["shared"]},
            ]),
            "upstreams",
        )
        .unwrap();
        assert!(upstreams.route("shared", |_| false).is_none());
        assert_eq!(
            &**upstreams
                .route("two/shared", |_| false)
                .unwrap()
                .upstream
                .name(),
            "two"
        );
    }

    #[test]
    fn rewrite_model_field_changes_only_the_model() {
        let body = br#"{"model":"igalia/glm-5p3","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#;
        let rewritten: Value =
            serde_json::from_slice(&rewrite_model_field(body, "glm-5p3").unwrap()).unwrap();
        let mut expected: Value = serde_json::from_slice(body).unwrap();
        expected["model"] = json!("glm-5p3");
        assert_eq!(rewritten, expected);
        assert!(rewrite_model_field(b"[1]", "m").is_none());
    }

    #[test]
    fn sse_rewriter_changes_only_message_start_across_split_chunks() {
        let stream = concat!(
            "event: message_start\r\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"glm-5p3\"}}\r\n\r\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"model\":\"glm-5p3\"}\n\n",
        );
        let bytes = stream.as_bytes();
        let mut rewriter = SseModelRewriter::new("igalia/glm-5p3".to_string());
        let mut out = Vec::new();
        for chunk in bytes.chunks(7) {
            out.extend(rewriter.push(chunk));
        }
        out.extend(rewriter.finish());
        let text = String::from_utf8(out).unwrap();
        let (first, rest) = text.split_once("\r\n\r\n").unwrap();
        assert!(first.starts_with("event: message_start\r\ndata: "));
        let data: Value = serde_json::from_str(first.split_once("data: ").unwrap().1).unwrap();
        assert_eq!(data["message"]["model"], "igalia/glm-5p3");
        assert_eq!(data["message"]["id"], "m1");
        assert_eq!(
            rest,
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"model\":\"glm-5p3\"}\n\n"
        );
    }

    #[test]
    fn sse_rewriter_passes_events_before_message_start_and_after_it_untouched() {
        let mut rewriter = SseModelRewriter::new("x/m".to_string());
        assert_eq!(
            rewriter.push(b"event: ping\ndata: {}\n\n"),
            b"event: ping\ndata: {}\n\n"
        );
        assert_eq!(rewriter.push(b"partial"), b"");
        assert_eq!(rewriter.finish(), b"partial");
        assert_eq!(rewriter.push(b"after"), b"after");
    }

    #[test]
    fn restores_model_in_json_response_only_when_present() {
        let restored = restore_model_in_json(
            br#"{"id":"m","model":"glm-5p3","content":[]}"#,
            "igalia/glm-5p3",
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&restored).unwrap();
        assert_eq!(value["model"], "igalia/glm-5p3");
        assert_eq!(value["id"], "m");
        assert!(restore_model_in_json(br#"{"input_tokens":3}"#, "x/y").is_none());
        assert!(restore_model_in_json(b"not json", "x/y").is_none());
    }
}
