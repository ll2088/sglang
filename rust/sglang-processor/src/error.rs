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

impl ProcessorError {
    /// Whether this failure is the request's fault rather than the server's.
    ///
    /// Hosts map a request-fault error onto their client-error status (HTTP
    /// 400, gRPC `INVALID_ARGUMENT`) and everything else onto a server error,
    /// so a malformed request is never reported as a server defect.
    pub fn is_request_validation_error(&self) -> bool {
        matches!(self, Self::InvalidRequest(_))
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
}
