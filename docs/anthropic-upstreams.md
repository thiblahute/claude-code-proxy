# Anthropic-compatible upstreams

Claude Code talks to the proxy through one fixed `ANTHROPIC_BASE_URL`. Named
upstreams let it also reach any gateway that speaks the Anthropic Messages API
(for example a LiteLLM deployment) by picking a model with `/model`. Routing is
per request, by the request's `model` field, so switching between Claude, Codex
and an upstream model needs no restart.

## Configuration

Add an `upstreams` array to `config.json`, or point `CCP_UPSTREAMS_FILE` at a
JSON file holding the same array (the environment variable wins):

```json
{
  "upstreams": [
    {
      "name": "igalia",
      "base_url": "https://ai.igalia.com",
      "api_key_file": "/home/me/.config/claude-code-proxy/igalia.key",
      "models": ["glm-5p3", "glm-5p3-flash", "kimi-k3", "minimax-m3"],
      "discover_models": true
    }
  ]
}
```

- `name`: required, unique, made of `a-z`, `0-9` and `-`. It may not be
  `claude`, `codex`, `opencode-go` or a built-in provider name (`codex`,
  `cursor`, `grok`, `kimi`, `opencode`).
- `base_url`: https only. Plain http is accepted for `127.0.0.1` and `[::1]`
  only (tests), the same rule as `CCP_CLAUDE_BASE_URL`.
- `api_key_file`: must exist and be readable at startup. The key is read once
  and kept in memory; surrounding whitespace is trimmed. A key file readable by
  group or others is accepted with a warning in the log.
- `models` (optional): bare model names the upstream serves.
- `discover_models` (optional, default false): fetch `<base_url>/v1/models`
  with the upstream key at startup, then again on a proxy `/v1/models` request
  when the last fetch is at least 10 minutes old. A failed fetch keeps the
  last good list and logs a warning.

Any invalid entry stops the proxy at startup.

## Routing

For `/v1/messages` and `/v1/messages/count_tokens`, in this order:

1. `<name>/<model>` goes to upstream `<name>`. The prefix is stripped from the
   forwarded `model` field, and the response gets the requested id back: the
   `model` of a JSON response, or `message.model` of the streamed
   `message_start` event. Other events pass through unchanged and the stream
   stays incremental. To change the field the request JSON is parsed and
   serialized again, so values are kept but whitespace and key order are not.
2. A bare model listed (configured or discovered) by exactly one upstream, and
   not served by a built-in proxy provider, goes to that upstream with the body
   bytes unchanged.
3. Everything else keeps the existing behaviour: the backend selector (Claude
   passthrough or Codex), then the provider registry.

Example of a collision: `kimi-k3` is served both by the built-in `kimi`
provider and by the `igalia` upstream. A bare `kimi-k3` keeps going to the
built-in provider; use `igalia/kimi-k3` to reach the upstream.

`/v1/models` lists the existing entries first, then `<name>/<model>` for every
upstream model, then the bare names that rule 2 routes to an upstream.

## Sessions

`/control/sessions` records the upstream name as the session's `backend`. An
upstream name counts as a backend in a session's allowed list; the default
allowed list is the configured backend order followed by every upstream name,
and `POST /control/sessions/{id}/allowed` accepts upstream names. The global
backend switch (`/control/backend`) stays `claude` or `codex`. Upstream
requests never fail over to another backend.

## Security

A request routed to an upstream never carries the client's `authorization` or
`x-api-key` header: Claude Code may be sending a Claude OAuth token, which must
not reach a third-party gateway. The proxy drops both and sends
`Authorization: Bearer <key>` and `x-api-key: <key>` from the upstream's key
file instead. The rest of the Claude passthrough header allowlist applies
unchanged. The key never appears in logs, errors or the control API.

## Known limitation

Switching model mid-conversation sends the existing history, including Claude
or Codex thinking blocks, to a different model family. An upstream may reject
such a request.
