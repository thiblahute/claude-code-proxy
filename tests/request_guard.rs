//! Browser-abuse guard on `/v1/*`: Host, Origin and Content-Type checks.
//! Runs against in-process apps; no real network.

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use claude_code_proxy::{
    backend::BackendSettings,
    registry::Registry,
    server::{AppFeatures, app_with_backend_settings},
};
use serde_json::Value;
use std::sync::Arc;
use tower::util::ServiceExt;

const JSON_BODY: &str =
    r#"{"model":"gpt-5.5","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#;
// Malformed on purpose: once past the guard the handler answers 400 locally,
// so these tests never reach an upstream.
const INVALID_JSON_BODY: &str = "{";

fn app(features: AppFeatures) -> axum::Router {
    app_with_backend_settings(
        Arc::new(Registry::with_default_alias()),
        None,
        features,
        BackendSettings::defaults(),
    )
}

fn plain_features() -> AppFeatures {
    AppFeatures {
        responses_api: false,
        images_api: false,
        transcriptions_api: false,
    }
}

async fn send(app: axum::Router, request: Request<Body>) -> StatusCode {
    app.oneshot(request).await.unwrap().status()
}

async fn send_json_error(app: axum::Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

fn post_json(uri: &str, host: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(host) = host {
        builder = builder.header(header::HOST, host);
    }
    builder.body(Body::from(INVALID_JSON_BODY)).unwrap()
}

#[tokio::test]
async fn foreign_host_is_forbidden_with_json_error() {
    let request = Request::builder()
        .method(Method::GET)
        .uri("/v1/models")
        .header(header::HOST, "evil.example")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send_json_error(app(plain_features()), request).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "permission_error");
    assert_eq!(body["error"]["message"], "Host not allowed");
}

#[tokio::test]
async fn foreign_host_is_forbidden_on_post() {
    let status = send(
        app(plain_features()),
        post_json("/v1/messages", Some("evil.example:18765")),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn local_hosts_are_not_forbidden() {
    for host in [
        "127.0.0.1:18765",
        "127.0.0.1",
        "localhost",
        "localhost:18765",
        "[::1]:18765",
        "[::1]",
    ] {
        let status = send(app(plain_features()), post_json("/v1/messages", Some(host))).await;
        assert_ne!(status, StatusCode::FORBIDDEN, "host {host} was rejected");
    }
}

#[tokio::test]
async fn missing_host_is_allowed() {
    let status = send(app(plain_features()), post_json("/v1/messages", None)).await;
    assert_ne!(status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn origin_header_is_forbidden() {
    let request = Request::builder()
        .method(Method::GET)
        .uri("/v1/models")
        .header(header::ORIGIN, "https://evil.example")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send_json_error(app(plain_features()), request).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["error"]["message"], "Origin not allowed");
}

#[tokio::test]
async fn text_plain_post_is_unsupported_media_type() {
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header(header::HOST, "127.0.0.1:18765")
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::from(JSON_BODY))
        .unwrap();
    let (status, body) = send_json_error(app(plain_features()), request).await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert_eq!(
        body["error"]["message"],
        "Content-Type must be application/json"
    );
}

#[tokio::test]
async fn post_without_content_type_is_unsupported_media_type() {
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .body(Body::from(JSON_BODY))
        .unwrap();
    assert_eq!(
        send(app(plain_features()), request).await,
        StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
}

#[tokio::test]
async fn json_content_type_matches_case_insensitively_with_parameters() {
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "Application/JSON; charset=utf-8")
        .body(Body::from(INVALID_JSON_BODY))
        .unwrap();
    assert_ne!(
        send(app(plain_features()), request).await,
        StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
}

#[tokio::test]
async fn json_post_is_not_unsupported_media_type() {
    let status = send(app(plain_features()), post_json("/v1/messages", None)).await;
    assert_ne!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[tokio::test]
async fn get_is_not_checked_for_content_type() {
    let request = Request::builder()
        .method(Method::GET)
        .uri("/v1/models")
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::empty())
        .unwrap();
    assert_eq!(send(app(plain_features()), request).await, StatusCode::OK);
}

#[tokio::test]
async fn bad_host_on_get_models_is_forbidden() {
    let request = Request::builder()
        .method(Method::GET)
        .uri("/v1/models")
        .header(header::HOST, "evil.example")
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        send(app(plain_features()), request).await,
        StatusCode::FORBIDDEN
    );
}

#[tokio::test]
async fn healthz_is_not_guarded() {
    let request = Request::builder()
        .method(Method::GET)
        .uri("/healthz")
        .header(header::HOST, "evil.example")
        .header(header::ORIGIN, "https://evil.example")
        .body(Body::empty())
        .unwrap();
    assert_eq!(send(app(plain_features()), request).await, StatusCode::OK);
}

#[tokio::test]
async fn multipart_transcription_is_accepted_by_content_type_rule() {
    let features = AppFeatures {
        transcriptions_api: true,
        ..plain_features()
    };
    let boundary = "ccp-request-guard-test";
    // No `file` part: the handler rejects it locally with a 400, never upstream.
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n--{boundary}--\r\n"
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/audio/transcriptions")
        .header(header::HOST, "127.0.0.1:18765")
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap();
    assert_ne!(
        send(app(features), request).await,
        StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
}

#[tokio::test]
async fn monitor_with_foreign_host_is_forbidden() {
    let request = Request::builder()
        .method(Method::GET)
        .uri("/monitor")
        .header(header::HOST, "evil.example")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send_json_error(app(plain_features()), request).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body["error"]["message"], "Host not allowed");
}

#[tokio::test]
async fn control_with_origin_is_forbidden() {
    let request = Request::builder()
        .method(Method::GET)
        .uri("/control/backend")
        .header(header::HOST, "127.0.0.1:18765")
        .header(header::ORIGIN, "https://evil.example")
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        send(app(plain_features()), request).await,
        StatusCode::FORBIDDEN
    );
}

#[tokio::test]
async fn multipart_content_type_is_rejected_on_json_routes() {
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "multipart/form-data; boundary=x")
        .body(Body::from(JSON_BODY))
        .unwrap();
    assert_eq!(
        send(app(plain_features()), request).await,
        StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
}
