//! Failures reported by the Dynamo wrappers.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A rendering, tokenization, or output-parsing failure. Hosts map it onto
/// their own transport and engine error types.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum ProcessorError {
    /// The request cannot be rendered or constrained as given.
    #[error("{0}")]
    InvalidRequest(String),
    #[error("tokenize failed: {0}")]
    Tokenize(String),
    /// Output processing reached a state the request could not have caused.
    #[error("{0}")]
    Internal(String),
    /// A media source the request got wrong (bad base64, undecodable bytes,
    /// a URL the connector refuses). Kept as its own variant so hosts can map
    /// it onto a client-error status instead of a generic server error.
    #[error("invalid media: {0}")]
    Media(String),

    /// An `input_audio` part whose `format` is neither `wav` nor `mp3`.
    /// A dedicated variant keeps the field-level validation error on the
    /// client-error path instead of the generic multimodal one.
    #[error("unsupported input_audio format `{format}`; expected `wav` or `mp3`")]
    UnsupportedInputAudioFormat { format: String },
}

impl ProcessorError {
    /// Whether this failure is the request's fault rather than the server's.
    ///
    /// Hosts map a request-fault error onto their client-error status (HTTP
    /// 400, gRPC `INVALID_ARGUMENT`) and everything else onto a server error,
    /// so a malformed request is never reported as a server defect.
    pub fn is_request_validation_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidRequest(_)
                | Self::Media(_)
                | Self::UnsupportedInputAudioFormat { .. }
        )
    }

    /// Build a media error from a source the request got wrong. The message
    /// keeps the underlying cause so the client can see what was rejected.
    pub fn media(message: impl Into<String>) -> Self {
        Self::Media(message.into())
    }
}

impl From<String> for ProcessorError {
    fn from(message: String) -> Self {
        Self::InvalidRequest(message)
    }
}

impl From<&str> for ProcessorError {
    fn from(message: &str) -> Self {
        Self::InvalidRequest(message.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only a request the caller got wrong is a validation error; a tokenize
    /// or internal failure stays a server error.
    #[test]
    fn only_invalid_request_is_a_request_validation_error() {
        assert!(ProcessorError::InvalidRequest("bad media".into()).is_request_validation_error());
        assert!(ProcessorError::from("bad media").is_request_validation_error());
        assert!(!ProcessorError::Tokenize("boom".into()).is_request_validation_error());
        assert!(!ProcessorError::Internal("bug".into()).is_request_validation_error());
    }

    /// Media the request got wrong is a client fault (400 / INVALID_ARGUMENT),
    /// while a tokenize or internal failure stays a server fault (5xx).
    #[test]
    fn media_error_is_a_request_validation_error() {
        let error = ProcessorError::media("base64 decode error: Invalid symbol 32");
        assert!(error.is_request_validation_error());
        assert!(error.to_string().contains("base64 decode error"));
        assert!(!ProcessorError::Tokenize("boom".into()).is_request_validation_error());
        assert!(!ProcessorError::Internal("bug".into()).is_request_validation_error());
    }

    /// A media error built from a fetch failure keeps the underlying cause in
    /// its message, so the client sees what was rejected rather than a generic
    /// server error.
    #[test]
    fn media_error_preserves_the_underlying_cause() {
        let error = ProcessorError::media("media fetch: malformed data: URL");
        assert!(error.is_request_validation_error());
        assert!(error.to_string().contains("malformed data: URL"));
    }

    /// An unsupported `input_audio` format is the request's fault, so hosts
    /// answer 400 / `INVALID_ARGUMENT` rather than a server error.
    #[test]
    fn unsupported_input_audio_format_is_a_request_validation_error() {
        let error = ProcessorError::UnsupportedInputAudioFormat {
            format: "flac".into(),
        };
        assert!(error.is_request_validation_error());
        assert!(error.to_string().contains("flac"), "{error}");
        assert!(error.to_string().contains("wav"), "{error}");
        assert!(error.to_string().contains("mp3"), "{error}");
    }

    /// The classification boundary: a request-fault error is a validation
    /// error, a server-side failure is not.
    #[test]
    fn request_fault_and_server_fault_are_distinguished() {
        let client = [
            ProcessorError::InvalidRequest("bad".into()),
            ProcessorError::media("base64 decode error"),
            ProcessorError::UnsupportedInputAudioFormat {
                format: "flac".into(),
            },
        ];
        for error in client {
            assert!(error.is_request_validation_error(), "{error}");
        }
        let server = [
            ProcessorError::Tokenize("boom".into()),
            ProcessorError::Internal("bug".into()),
        ];
        for error in server {
            assert!(!error.is_request_validation_error(), "{error}");
        }
    }
}
