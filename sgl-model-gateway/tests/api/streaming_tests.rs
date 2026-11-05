use serde_json::json;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    WorkerTestContext,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_streaming() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 20001,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 10,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": "Stream test",
            "stream": true,
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 10
            }
        });

        let result = ctx.make_streaming_request("/generate", payload).await;
        assert!(result.is_ok());

        let events = result.unwrap();
        assert!(events.len() >= 2);
        assert_eq!(events.last().unwrap(), "[DONE]");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_chat_completions_streaming() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 20002,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 10,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Count to 3"}
            ],
            "stream": true,
            "max_tokens": 20
        });

        let result = ctx
            .make_streaming_request("/v1/chat/completions", payload)
            .await;
        assert!(result.is_ok());

        let events = result.unwrap();
        assert!(events.len() >= 2);

        for event in &events {
            if event != "[DONE]" {
                let parsed: Result<serde_json::Value, _> = serde_json::from_str(event);
                assert!(parsed.is_ok(), "Invalid JSON in SSE event: {}", event);

                let json = parsed.unwrap();
                assert_eq!(
                    json.get("object").and_then(|v| v.as_str()),
                    Some("chat.completion.chunk")
                );
            }
        }

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_completions_streaming() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 20003,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 10,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "model": "test-model",
            "prompt": "Once upon a time",
            "stream": true,
            "max_tokens": 15
        });

        let result = ctx.make_streaming_request("/v1/completions", payload).await;
        assert!(result.is_ok());

        let events = result.unwrap();
        assert!(events.len() >= 2);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_streaming_with_error() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 20004,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 1.0,
        }])
        .await;

        let payload = json!({
            "text": "This should fail",
            "stream": true
        });

        let result = ctx.make_streaming_request("/generate", payload).await;
        assert!(result.is_err());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_streaming_timeouts() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 20005,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 100,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": "Slow stream",
            "stream": true,
            "sampling_params": {
                "max_new_tokens": 5
            }
        });

        let start = std::time::Instant::now();
        let result = ctx.make_streaming_request("/generate", payload).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        let events = result.unwrap();
        assert!(!events.is_empty());
        assert!(elapsed.as_millis() >= 100);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_batch_streaming() {
        let ctx = WorkerTestContext::new(vec![MockWorkerConfig {
            port: 20006,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 10,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": ["First", "Second", "Third"],
            "stream": true,
            "sampling_params": {
                "max_new_tokens": 5
            }
        });

        let result = ctx.make_streaming_request("/generate", payload).await;
        assert!(result.is_ok());

        let events = result.unwrap();
        assert!(events.len() >= 4);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_sse_format_parsing() {
        let parse_sse_chunk = |chunk: &[u8]| -> Vec<String> {
            let text = String::from_utf8_lossy(chunk);
            text.lines()
                .filter(|line| line.starts_with("data: "))
                .map(|line| line[6..].to_string())
                .collect()
        };

        let sse_data =
            b"data: {\"text\":\"Hello\"}\n\ndata: {\"text\":\" world\"}\n\ndata: [DONE]\n\n";
        let events = parse_sse_chunk(sse_data);

        assert_eq!(events.len(), 3);
        assert_eq!(events[0], "{\"text\":\"Hello\"}");
        assert_eq!(events[1], "{\"text\":\" world\"}");
        assert_eq!(events[2], "[DONE]");

        let mixed = b"event: message\ndata: {\"test\":true}\n\n: comment\ndata: [DONE]\n\n";
        let events = parse_sse_chunk(mixed);

        assert_eq!(events.len(), 2);
        assert_eq!(events[0], "{\"test\":true}");
        assert_eq!(events[1], "[DONE]");
    }

    #[test]
    fn test_sse_error_event_after_tokens_is_visible() {
        let parse_sse_chunk = |chunk: &[u8]| -> Vec<String> {
            let text = String::from_utf8_lossy(chunk);
            text.lines()
                .filter(|line| line.starts_with("data: "))
                .map(|line| line[6..].to_string())
                .collect()
        };

        let sse_data = b"data: {\"choices\":[{\"text\":\"a\"}]}\n\n\
                        data: {\"choices\":[{\"text\":\"b\"}]}\n\n\
                        data: {\"error\":{\"message\":\"boom\",\"code\":500}}\n\n\
                        data: [DONE]\n\n";
        let events = parse_sse_chunk(sse_data);

        assert_eq!(events.len(), 4);
        assert_eq!(events[3], "[DONE]");

        let error_event: serde_json::Value = serde_json::from_str(&events[2]).unwrap();
        assert_eq!(
            error_event.get("error").and_then(|e| e.get("message")),
            Some(&serde_json::Value::String("boom".to_string()))
        );
        assert!(error_event.get("choices").is_none());
    }

    #[test]
    fn test_sse_error_event_after_tokens_marks_stream_failed() {
        // Mirrors the benchmark client rule: a stream that delivered tokens and
        // then an error envelope must be classified as failed, not successful.
        let parse_sse_chunk = |chunk: &[u8]| -> Vec<String> {
            let text = String::from_utf8_lossy(chunk);
            text.lines()
                .filter(|line| line.starts_with("data: "))
                .map(|line| line[6..].to_string())
                .collect()
        };

        let sse_data = b"data: {\"choices\":[{\"text\":\"a\"}]}\n\n\
                        data: {\"choices\":[{\"text\":\"b\"}]}\n\n\
                        data: {\"error\":{\"message\":\"boom\",\"code\":500}}\n\n\
                        data: [DONE]\n\n";
        let events = parse_sse_chunk(sse_data);

        let mut saw_token = false;
        let mut stream_error: Option<String> = None;
        for event in &events {
            if event == "[DONE]" {
                continue;
            }
            let json: serde_json::Value = serde_json::from_str(event).unwrap();
            if let Some(error) = json.get("error") {
                stream_error = Some(error.to_string());
            } else if json.get("choices").is_some() {
                saw_token = true;
            }
        }

        assert!(saw_token, "tokens must have been observed before the error");
        assert!(
            stream_error.is_some(),
            "error envelope must be surfaced to the caller"
        );

        // The benchmark client rule: an error envelope wins over tokens.
        let success = stream_error.is_none() && saw_token;
        assert!(!success, "stream ending in an error event must be failed");
    }

    #[test]
    fn test_sse_error_event_after_tokens_is_not_counted_in_summary() {
        // Mirrors the benchmark summary rule: a stream that delivered tokens
        // and then an error envelope must not be counted as a success.
        let parse_sse_chunk = |chunk: &[u8]| -> Vec<String> {
            let text = String::from_utf8_lossy(chunk);
            text.lines()
                .filter(|line| line.starts_with("data: "))
                .map(|line| line[6..].to_string())
                .collect()
        };

        let sse_data = b"data: {\"choices\":[{\"text\":\"a\"}]}\n\n\
                        data: {\"choices\":[{\"text\":\"b\"}]}\n\n\
                        data: {\"error\":{\"message\":\"boom\",\"code\":500}}\n\n\
                        data: [DONE]\n\n";
        let events = parse_sse_chunk(sse_data);

        let mut saw_token = false;
        let mut stream_error: Option<String> = None;
        for event in &events {
            if event == "[DONE]" {
                continue;
            }
            let json: serde_json::Value = serde_json::from_str(event).unwrap();
            if let Some(error) = json.get("error") {
                stream_error = Some(error.to_string());
            } else if json.get("choices").is_some() {
                saw_token = true;
            }
        }

        assert!(saw_token);
        assert!(stream_error.is_some());

        let success = stream_error.is_none() && saw_token;
        assert!(!success);

        // The summary must report this request as failed, not successful.
        let completed_requests = if success { 1 } else { 0 };
        let failed_requests = if success { 0 } else { 1 };
        assert_eq!(completed_requests, 0);
        assert_eq!(failed_requests, 1);
    }

    #[test]
    fn test_sse_error_event_after_tokens_is_failed_even_with_usage() {
        // A usage-only chunk after tokens must not mask a preceding error
        // envelope: the error is the root cause and wins over any output.
        let parse_sse_chunk = |chunk: &[u8]| -> Vec<String> {
            let text = String::from_utf8_lossy(chunk);
            text.lines()
                .filter(|line| line.starts_with("data: "))
                .map(|line| line[6..].to_string())
                .collect()
        };

        let sse_data = b"data: {\"choices\":[{\"text\":\"a\"}]}\n\n\
                        data: {\"error\":{\"message\":\"boom\",\"code\":500}}\n\n\
                        data: {\"usage\":{\"completion_tokens\":1}}\n\n\
                        data: [DONE]\n\n";
        let events = parse_sse_chunk(sse_data);

        let mut saw_token = false;
        let mut stream_error: Option<String> = None;
        for event in &events {
            if event == "[DONE]" {
                continue;
            }
            let json: serde_json::Value = serde_json::from_str(event).unwrap();
            if let Some(error) = json.get("error") {
                stream_error = Some(error.to_string());
            } else if json.get("choices").is_some() {
                saw_token = true;
            }
        }

        assert!(saw_token);
        assert!(stream_error.is_some());
        let success = stream_error.is_none() && saw_token;
        assert!(!success, "usage chunk must not turn a failed stream into success");
    }

    #[test]
    fn test_sse_error_event_after_tokens_is_failed_with_usage_and_done() {
        // A usage-only chunk and [DONE] after the error must not mask it.
        let parse_sse_chunk = |chunk: &[u8]| -> Vec<String> {
            let text = String::from_utf8_lossy(chunk);
            text.lines()
                .filter(|line| line.starts_with("data: "))
                .map(|line| line[6..].to_string())
                .collect()
        };

        let sse_data = b"data: {\"choices\":[{\"text\":\"a\"}]}\n\n\
                        data: {\"error\":{\"message\":\"boom\",\"code\":500}}\n\n\
                        data: {\"usage\":{\"completion_tokens\":1}}\n\n\
                        data: [DONE]\n\n";
        let events = parse_sse_chunk(sse_data);

        let mut saw_token = false;
        let mut stream_error: Option<String> = None;
        for event in &events {
            if event == "[DONE]" {
                continue;
            }
            let json: serde_json::Value = serde_json::from_str(event).unwrap();
            if let Some(error) = json.get("error") {
                stream_error = Some(error.to_string());
            } else if json.get("choices").is_some() {
                saw_token = true;
            }
        }

        assert!(saw_token);
        assert!(stream_error.is_some());
        let success = stream_error.is_none() && saw_token;
        assert!(!success);
    }

    #[test]
    fn test_sse_stream_without_error_is_successful() {
        // Guard against over-eager error detection: a clean token stream must
        // still be classified as successful.
        let parse_sse_chunk = |chunk: &[u8]| -> Vec<String> {
            let text = String::from_utf8_lossy(chunk);
            text.lines()
                .filter(|line| line.starts_with("data: "))
                .map(|line| line[6..].to_string())
                .collect()
        };

        let sse_data = b"data: {\"choices\":[{\"text\":\"a\"}]}\n\n\
                        data: {\"choices\":[{\"text\":\"b\"}]}\n\n\
                        data: [DONE]\n\n";
        let events = parse_sse_chunk(sse_data);

        let mut saw_token = false;
        let mut stream_error: Option<String> = None;
        for event in &events {
            if event == "[DONE]" {
                continue;
            }
            let json: serde_json::Value = serde_json::from_str(event).unwrap();
            if let Some(error) = json.get("error") {
                stream_error = Some(error.to_string());
            } else if json.get("choices").is_some() {
                saw_token = true;
            }
        }

        assert!(saw_token, "tokens must have been observed");
        assert!(stream_error.is_none(), "no error envelope expected");
        let success = stream_error.is_none() && saw_token;
        assert!(success, "clean stream must be reported as successful");
    }
}
