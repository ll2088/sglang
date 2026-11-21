// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Hold a guard until a response body is finished.
//!
//! Several middlewares need to keep some state alive for exactly as long as
//! the client is still being written to: an in-flight counter, a load gauge,
//! an HTTP-metrics timer. The handler returning is *not* that moment — for a
//! streaming completion it only means the headers are ready, and the body may
//! still be emitting tokens for minutes afterwards. A client that goes away
//! mid-stream drops the body instead of draining it, and that path must
//! release the state too.
//!
//! [`hold_until_body_dropped`] wraps a [`Response`] so that `guard` is moved
//! into the body and dropped when the body is dropped: after the last frame is
//! sent, or earlier when the client disconnects. The guard's own `Drop` impl
//! does the recording, so it runs exactly once on every path.

use axum::body::{Body, Bytes, HttpBody};
use axum::response::Response;
use http_body::{Frame, SizeHint};
use std::pin::Pin;
use std::task::{Context, Poll};

/// Wrap `response` so that `guard` lives until its body is dropped.
///
/// The response's status, headers and extensions are preserved; only the body
/// is replaced by a transparent wrapper. `G` must be `Send + Unpin + 'static`
/// because the body is moved across tasks by the server.
///
/// Shared by the in-flight counter and any future HTTP-metrics middleware so
/// both release their state on the same paths: last frame sent, or body
/// dropped early when the client goes away mid-stream.
pub(crate) fn hold_until_body_dropped<G>(response: Response, guard: G) -> Response
where
    G: Send + Unpin + 'static,
{
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(GuardedBody {
            inner: body,
            _guard: guard,
        }),
    )
}

/// A response body that owns a guard, dropping it when the body is dropped.
struct GuardedBody<G> {
    inner: Body,
    _guard: G,
}

// Delegate every `HttpBody` method to the inner body; the wrapper adds no
// framing, buffering or error mapping of its own.
impl<G: Unpin> HttpBody for GuardedBody<G> {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{StatusCode, header};
    use http_body_util::BodyExt;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A guard that counts how many times it was dropped, so a test can tell
    /// "not yet" from "exactly once".
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// The wrapper must not disturb the response it wraps: status, headers and
    /// body bytes all pass through unchanged, and the guard runs exactly once.
    #[tokio::test]
    async fn a_unit_body_passes_through_unchanged() {
        let drops = Arc::new(AtomicUsize::new(0));
        let response = Response::builder()
            .status(StatusCode::CREATED)
            .header(header::CONTENT_TYPE, "text/plain")
            .body(Body::from("payload"))
            .unwrap();
        let response = hold_until_body_dropped(response, DropCounter(drops.clone()));

        assert_eq!(response.status(), StatusCode::CREATED);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/plain",
        );
        assert_eq!(drops.load(Ordering::SeqCst), 0, "not dropped before reading");

        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("collecting a guarded body must not error")
            .to_bytes();
        assert_eq!(&bytes[..], b"payload");
        assert_eq!(drops.load(Ordering::SeqCst), 1, "dropped exactly once");
    }

    /// The property the helper exists for: a body with frames still to come
    /// keeps the guard alive, and only the end of the stream releases it.
    #[tokio::test]
    async fn a_streaming_body_holds_the_guard_until_its_last_frame() {
        let drops = Arc::new(AtomicUsize::new(0));
        let (mut tx, rx) = futures::channel::mpsc::channel::<Result<Bytes, axum::Error>>(1);
        let response = Response::new(Body::from_stream(rx));
        let response = hold_until_body_dropped(response, DropCounter(drops.clone()));

        let mut stream = response.into_body().into_data_stream();
        futures::SinkExt::send(&mut tx, Ok(Bytes::from_static(b"chunk")))
            .await
            .expect("the stream must accept a chunk");
        let frame = futures::StreamExt::next(&mut stream).await;
        assert!(frame.is_some(), "the wrapper must pass frames through");
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "a body with frames still to come must hold the guard",
        );

        drop(tx);
        assert!(
            futures::StreamExt::next(&mut stream).await.is_none(),
            "the wrapper must end when the inner body does",
        );
        drop(stream);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "the finished body releases the guard",
        );
    }

    /// A client that disconnects mid-stream drops the body rather than
    /// draining it. The guard must still run, exactly once, or the state it
    /// protects ratchets up over the life of the process.
    #[tokio::test]
    async fn an_abandoned_body_releases_the_guard_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let (_tx, rx) = futures::channel::mpsc::channel::<Result<Bytes, axum::Error>>(1);
        let response = Response::new(Body::from_stream(rx));
        let response = hold_until_body_dropped(response, DropCounter(drops.clone()));
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        drop(response);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "an abandoned body must release the guard",
        );
    }

    /// Dropping the response *before* the body is ever polled must also run
    /// the guard exactly once: this is the path a cancelled request takes when
    /// the client goes away between the handler returning and the first frame
    /// being written. A guard that only ran on the last frame would leak here.
    #[tokio::test]
    async fn a_never_polled_body_releases_the_guard_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let response = Response::new(Body::from("never read"));
        let response = hold_until_body_dropped(response, DropCounter(drops.clone()));
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        drop(response);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "a body dropped before its first poll must release the guard",
        );
    }

    /// The guard must run exactly once even when the body is fully drained and
    /// then dropped: draining ends the stream, and the subsequent drop of the
    /// wrapper must not run `Drop` a second time. The counter is asserted to
    /// be exactly 1 (not merely non-zero) so a double release fails here.
    #[tokio::test]
    async fn a_drained_body_releases_the_guard_exactly_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let response = Response::new(Body::from("drained"));
        let response = hold_until_body_dropped(response, DropCounter(drops.clone()));

        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("collecting a guarded body must not error")
            .to_bytes();
        assert_eq!(&bytes[..], b"drained");
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "a drained body must release the guard exactly once",
        );
    }

    /// The two paths the helper exists to unify must agree: a body that is
    /// fully consumed and a body that is dropped early each release the guard
    /// exactly once, so a caller cannot tell them apart by the drop count.
    #[tokio::test]
    async fn completed_and_abandoned_bodies_release_the_guard_once_each() {
        let completed = Arc::new(AtomicUsize::new(0));
        let response = Response::new(Body::from("done"));
        let response = hold_until_body_dropped(response, DropCounter(completed.clone()));
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("collecting a guarded body must not error")
            .to_bytes();
        assert_eq!(&bytes[..], b"done");
        assert_eq!(completed.load(Ordering::SeqCst), 1);

        let abandoned = Arc::new(AtomicUsize::new(0));
        let response = Response::new(Body::from("never read"));
        let response = hold_until_body_dropped(response, DropCounter(abandoned.clone()));
        drop(response);
        assert_eq!(abandoned.load(Ordering::SeqCst), 1);
    }
}
