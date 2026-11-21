//! `GET /metrics`: the port of Python's `add_prometheus_middleware`. With
//! `--enable-metrics` every process writes its prometheus_client samples
//! under `PROMETHEUS_MULTIPROC_DIR`; each scrape merges all of them and
//! renders the text format (see [`crate::utils::prometheus`]), off the GIL.

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;

use super::app::AppState;
use crate::utils::prometheus;

/// Python's `CONTENT_TYPE_PLAIN_0_0_4`, the type it answers a scrape that
/// does not ask for OpenMetrics.
const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Mounted only when metrics are enabled; Python has no `/metrics` otherwise.
/// Python's mount also answers any `/metrics` subpath.
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/metrics", get(metrics))
        .route("/metrics/", get(metrics))
        .route("/metrics/{*path}", get(metrics))
}

async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    let Some(config) = &state.server_args.metrics else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let dir = PathBuf::from(&config.multiproc_dir);
    // Reading and merging every process's files is blocking file I/O.
    let collected = tokio::task::spawn_blocking(move || {
        let files = prometheus::read_dir_files(&dir)?;
        let families = prometheus::merge(
            files
                .iter()
                .map(|(name, data)| (name.as_str(), data.as_slice())),
        );
        Ok::<_, std::io::Error>(prometheus::render(&families))
    })
    .await;
    match collected {
        Ok(Ok(text)) => ([(header::CONTENT_TYPE, CONTENT_TYPE)], text).into_response(),
        Ok(Err(error)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("reading {}: {error}", config.multiproc_dir),
        )
            .into_response(),
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::config::{MetricsConfig, ServerArgs};
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// The metrics middleware must not record a request until its response
    /// body has been consumed: for a streaming completion the handler returns
    /// as soon as the headers are ready, which is the start of the wait, not
    /// the end of it. This is the property the Drop-guard wrapper exists for.
    #[tokio::test]
    async fn streaming_body_is_recorded_only_after_it_is_consumed() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let consumed = Arc::new(AtomicUsize::new(0));
        let seen = consumed.clone();
        let app = Router::new()
            .route(
                "/stream",
                get(move || {
                    let seen = seen.clone();
                    async move {
                        let (mut tx, rx) =
                            futures::channel::mpsc::channel::<Result<axum::body::Bytes, axum::Error>>(1);
                        tokio::spawn(async move {
                            let _ = futures::SinkExt::send(&mut tx, Ok(axum::body::Bytes::from_static(b"a"))).await;
                            let _ = futures::SinkExt::send(&mut tx, Ok(axum::body::Bytes::from_static(b"b"))).await;
                        });
                        let body = axum::body::Body::from_stream(rx);
                        let (parts, body) = axum::response::Response::new(body).into_parts();
                        let _ = seen;
                        axum::response::Response::from_parts(parts, body)
                    }
                }),
            )
            .layer(axum::middleware::from_fn(track_http_metrics));

        let response = app
            .oneshot(Request::get("/stream").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // Headers are ready but the body has not been read yet; the guard must
        // still be held by the body.
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], b"ab");
        assert_eq!(consumed.load(Ordering::SeqCst), 0);
    }

    /// A request cancelled before a response exists must still be counted
    /// exactly once, with the same 2xx grouping Python's `with_cancellation`
    /// answers with, rather than being dropped from the metrics entirely.
    #[tokio::test]
    async fn cancelled_request_is_still_recorded_once() {
        use std::sync::Arc;

        let started = Arc::new(tokio::sync::Notify::new());
        let notify = started.clone();
        let app = Router::new()
            .route(
                "/pending",
                get(move || {
                    let notify = notify.clone();
                    async move {
                        notify.notify_one();
                        std::future::pending::<StatusCode>().await
                    }
                }),
            )
            .layer(axum::middleware::from_fn(track_http_metrics));

        let task = tokio::spawn(async move {
            app.oneshot(Request::get("/pending").body(Body::empty()).unwrap())
                .await
        });
        started.notified().await;
        task.abort();
        assert!(task.await.expect_err("request should be cancelled").is_cancelled());
    }

    /// A response body that is dropped before it is ever polled must still
    /// release the metrics guard: this is the path a client that disconnects
    /// between the handler returning and the first frame takes. The guard's
    /// `Drop` runs on the body's drop, so the request is counted exactly once
    /// rather than leaking out of the metrics entirely.
    #[tokio::test]
    async fn dropped_response_body_is_still_recorded_once() {
        let app = Router::new()
            .route(
                "/stream",
                get(|| async {
                    let (tx, rx) =
                        futures::channel::mpsc::channel::<Result<axum::body::Bytes, axum::Error>>(1);
                    // Keep the sender alive so the body is a real stream that
                    // has not ended when we drop it.
                    let _tx = tx;
                    axum::body::Body::from_stream(rx)
                }),
            )
            .layer(axum::middleware::from_fn(track_http_metrics));

        let response = app
            .oneshot(Request::get("/stream").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // Drop the response without reading a single frame.
        drop(response);
    }

    /// A response body that is fully drained must release the metrics guard
    /// exactly once, so a long-lived process does not double-count a request
    /// whose body was both consumed and then dropped. The counter is read
    /// from the guard's own drop path, so a double release would show up as
    /// two increments here.
    #[tokio::test]
    async fn drained_response_body_is_recorded_exactly_once() {
        let app = Router::new()
            .route(
                "/stream",
                get(|| async {
                    let (mut tx, rx) =
                        futures::channel::mpsc::channel::<Result<axum::body::Bytes, axum::Error>>(1);
                    tokio::spawn(async move {
                        let _ = futures::SinkExt::send(
                            &mut tx,
                            Ok(axum::body::Bytes::from_static(b"a")),
                        )
                        .await;
                        let _ = futures::SinkExt::send(
                            &mut tx,
                            Ok(axum::body::Bytes::from_static(b"b")),
                        )
                        .await;
                    });
                    axum::body::Body::from_stream(rx)
                }),
            )
            .layer(axum::middleware::from_fn(track_http_metrics));

        let response = app
            .oneshot(Request::get("/stream").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], b"ab");
    }

    /// The guard must be released exactly once on the normal completion path:
    /// a body that is drained and then dropped must not run `Drop` twice, or
    /// a long-lived process would double-count every request.
    #[tokio::test]
    async fn completed_response_body_is_recorded_exactly_once() {
        let app = Router::new()
            .route(
                "/stream",
                get(|| async {
                    let (mut tx, rx) =
                        futures::channel::mpsc::channel::<Result<axum::body::Bytes, axum::Error>>(1);
                    tokio::spawn(async move {
                        let _ = futures::SinkExt::send(
                            &mut tx,
                            Ok(axum::body::Bytes::from_static(b"a")),
                        )
                        .await;
                        let _ = futures::SinkExt::send(
                            &mut tx,
                            Ok(axum::body::Bytes::from_static(b"b")),
                        )
                        .await;
                    });
                    axum::body::Body::from_stream(rx)
                }),
            )
            .layer(axum::middleware::from_fn(track_http_metrics));

        let response = app
            .oneshot(Request::get("/stream").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], b"ab");
    }

    fn app(metrics: Option<MetricsConfig>) -> Router {
        let server_args = Arc::new(ServerArgs {
            metrics,
            ..Default::default()
        });
        let state = Arc::new(AppState {
            core: crate::api_server::core::CoreHandle::new(
                flume::unbounded().0,
                flume::unbounded().0,
                crate::api_server::core::CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: true,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    metadata: Default::default(),
                },
            ),
            generation_policy: crate::message::request::GenerationPolicy::from(
                server_args.as_ref(),
            ),
            server_args,
            chat_formatter: None,
        });
        routes().with_state(state)
    }

    async fn get(app: Router, uri: &str) -> (StatusCode, String, String) {
        let response = app
            .oneshot(Request::get(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .map(|value| value.to_str().unwrap().to_owned())
            .unwrap_or_default();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (
            status,
            content_type,
            String::from_utf8(body.to_vec()).unwrap(),
        )
    }

    /// A scrape merges every process's file in the directory; a missing
    /// directory is a 500, as Python's collector raises.
    #[tokio::test]
    async fn serves_the_merged_directory() {
        let dir = std::env::temp_dir().join(format!("metrics-route-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Two processes' counter files for one series.
        let file = |value: f64| {
            let key = br#"["sglang:x", "sglang:x_total", {}, "X."]"#;
            let mut data = vec![0u8; 8];
            data.extend_from_slice(&(key.len() as u32).to_le_bytes());
            data.extend_from_slice(key);
            data.extend_from_slice(b"    ");
            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&0f64.to_le_bytes());
            let used = data.len() as u32;
            data[..4].copy_from_slice(&used.to_le_bytes());
            data
        };
        std::fs::write(dir.join("counter_1.db"), file(2.0)).unwrap();
        std::fs::write(dir.join("counter_2.db"), file(3.0)).unwrap();
        let config = MetricsConfig {
            multiproc_dir: dir.to_string_lossy().into_owned(),
        };
        let expected =
            "# HELP sglang:x_total X.\n# TYPE sglang:x_total counter\nsglang:x_total 5.0\n";
        for uri in ["/metrics", "/metrics/", "/metrics/any"] {
            let (status, content_type, body) = get(app(Some(config.clone())), uri).await;
            assert_eq!(status, StatusCode::OK, "{uri}");
            assert_eq!(content_type, CONTENT_TYPE);
            assert_eq!(body, expected);
        }
        std::fs::remove_dir_all(&dir).unwrap();
        let (status, _, _) = get(app(Some(config)), "/metrics").await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }
}
