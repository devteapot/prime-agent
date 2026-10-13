use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn large_4_sends_high_reasoning_effort_without_prompt_mode() {
    for id in [
        "mistral-large-4",
        "mistral-large-4-0",
        "mistral-large-4-unrestricted",
    ] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            let body_start;
            let content_length;
            loop {
                let read = socket.read(&mut chunk).await.unwrap();
                assert_ne!(read, 0, "request ended before its headers");
                request.extend_from_slice(&chunk[..read]);
                if let Some(pos) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                    body_start = pos + 4;
                    content_length = String::from_utf8_lossy(&request[..pos])
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .expect("POST has a content length");
                    break;
                }
            }
            while request.len() < body_start + content_length {
                let read = socket.read(&mut chunk).await.unwrap();
                assert_ne!(read, 0, "request ended before its body");
                request.extend_from_slice(&chunk[..read]);
            }
            let body: Value =
                serde_json::from_slice(&request[body_start..body_start + content_length]).unwrap();
            let response_body = concat!(
                "data: {\"id\":\"ml4\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"OK\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n",
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                response_body.len(),
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            body
        });
        let model: Model = serde_json::from_value(json!({
            "id": id, "name": id,
            "api": API_MISTRAL_CONVERSATIONS, "provider": "mistral",
            "baseUrl": format!("http://{addr}"), "reasoning": true,
            "thinkingLevelMap": {
                "off": null, "minimal": null, "low": null, "medium": null,
                "high": "high", "xhigh": null, "max": null,
            },
            "input": ["text", "image"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 1_048_576, "maxTokens": 128,
        }))
        .unwrap();
        let options = SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("test".into()),
                ..Default::default()
            },
            reasoning: Some(ModelThinkingLevel::High),
            ..Default::default()
        };
        let context = Context {
            system_prompt: None,
            messages: vec![],
            tools: None,
        };
        let result = stream_simple_mistral(&model, &context, Some(&options))
            .result()
            .await;
        assert_eq!(result.stop_reason, StopReason::Stop);
        assert_eq!(
            server.await.unwrap(),
            json!({
                "model": id, "stream": true, "messages": [], "max_tokens": 128,
                "reasoning_effort": "high",
            })
        );
    }
}
