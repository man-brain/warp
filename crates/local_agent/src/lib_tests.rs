//! Wire-level tests: a mockito server plays an OpenAI-compatible endpoint
//! returning canned `text/event-stream` bodies, and we assert the ordered
//! [`AgentEvent`]s produced by [`run_turn`]. No internals are mocked — the
//! whole HTTP/SSE path is exercised.

use futures::StreamExt;
use mockito::{Matcher, Server, ServerGuard};
use serde_json::json;

use super::*;

fn config_for(server: &ServerGuard) -> LocalEndpointConfig {
    LocalEndpointConfig {
        base_url: server.url(),
        api_key: "test-key".to_string(),
        model_slug: "test-model".to_string(),
    }
}

fn user_only_transcript() -> Vec<ChatMessage> {
    vec![
        ChatMessage::System("You are a coding agent.".to_string()),
        ChatMessage::User("hi".to_string()),
    ]
}

fn shell_tool() -> ToolDefinition {
    ToolDefinition {
        name: "run_shell_command".to_string(),
        description: "Run a shell command".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "command": { "type": "string" } },
            "required": ["command"],
        }),
    }
}

/// One `chat.completion.chunk` SSE event with the given choice delta.
fn delta_chunk(delta: serde_json::Value, finish_reason: Option<&str>) -> serde_json::Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
    })
}

fn usage_chunk(prompt_tokens: u64, completion_tokens: u64) -> serde_json::Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })
}

fn sse_body(chunks: &[serde_json::Value], terminated: bool) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    if terminated {
        body.push_str("data: [DONE]\n\n");
    }
    body
}

async fn mock_completion(server: &mut ServerGuard, body: String) -> mockito::Mock {
    server
        .mock("POST", "/chat/completions")
        .match_header("authorization", "Bearer test-key")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(body)
        .create_async()
        .await
}

async fn collect_events(
    transcript: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
    config: LocalEndpointConfig,
) -> Vec<Result<AgentEvent, LocalAgentError>> {
    run_turn(transcript, tools, config).collect().await
}

/// Unwraps a fully successful run into its events, panicking on any error.
fn expect_ok(results: Vec<Result<AgentEvent, LocalAgentError>>) -> Vec<AgentEvent> {
    results
        .into_iter()
        .map(|r| r.expect("expected only Ok events"))
        .collect()
}

#[tokio::test]
async fn complete_text_collects_the_full_reply() {
    let mut server = Server::new_async().await;
    let body = sse_body(
        &[
            delta_chunk(json!({ "role": "assistant", "content": "git " }), None),
            delta_chunk(json!({ "content": "status" }), None),
            delta_chunk(json!({}), Some("stop")),
        ],
        true,
    );
    let _mock = mock_completion(&mut server, body).await;

    let text = complete_text(user_only_transcript(), config_for(&server))
        .await
        .expect("completion should succeed");

    assert_eq!(text, "git status");
}

#[tokio::test]
async fn complete_text_propagates_errors() {
    let mut server = Server::new_async().await;
    let _mock = server
        .mock("POST", "/chat/completions")
        .with_status(401)
        .with_body(r#"{"error":{"message":"bad key"}}"#)
        .create_async()
        .await;

    let result = complete_text(user_only_transcript(), config_for(&server)).await;

    assert!(
        result.is_err(),
        "a 401 must surface as an error, not empty text"
    );
}

#[tokio::test]
async fn streams_text_deltas_then_usage_and_done() {
    let mut server = Server::new_async().await;
    let body = sse_body(
        &[
            delta_chunk(json!({ "role": "assistant", "content": "Hel" }), None),
            delta_chunk(json!({ "content": "lo" }), None),
            delta_chunk(json!({}), Some("stop")),
            usage_chunk(42, 7),
        ],
        true,
    );
    let _mock = mock_completion(&mut server, body).await;

    let events =
        expect_ok(collect_events(user_only_transcript(), vec![], config_for(&server)).await);

    assert_eq!(
        events,
        vec![
            AgentEvent::TextDelta("Hel".to_string()),
            AgentEvent::TextDelta("lo".to_string()),
            AgentEvent::Usage {
                input_tokens: 42,
                output_tokens: 7
            },
            AgentEvent::Done,
        ]
    );
}

#[tokio::test]
async fn accumulates_tool_call_arguments_fragmented_across_chunks() {
    let mut server = Server::new_async().await;
    // The id and name arrive only on the first fragment; later fragments
    // carry argument pieces under the same index, as OpenAI-compatible
    // servers commonly stream them.
    let body = sse_body(
        &[
            delta_chunk(
                json!({
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": { "name": "run_shell_command", "arguments": "" },
                    }],
                }),
                None,
            ),
            delta_chunk(
                json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "{\"comm" } }] }),
                None,
            ),
            delta_chunk(
                json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "and\":\"ls -la\"}" } }] }),
                None,
            ),
            delta_chunk(json!({}), Some("tool_calls")),
        ],
        true,
    );
    let _mock = mock_completion(&mut server, body).await;

    let events = expect_ok(
        collect_events(
            user_only_transcript(),
            vec![shell_tool()],
            config_for(&server),
        )
        .await,
    );

    assert_eq!(
        events,
        vec![
            AgentEvent::ToolCall(ToolCallRequest {
                id: "call_abc".to_string(),
                name: "run_shell_command".to_string(),
                arguments: json!({ "command": "ls -la" }),
            }),
            AgentEvent::Done,
        ]
    );
}

#[tokio::test]
async fn emits_parallel_tool_calls_in_index_order() {
    let mut server = Server::new_async().await;
    let body = sse_body(
        &[
            delta_chunk(
                json!({
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_one",
                        "type": "function",
                        "function": { "name": "run_shell_command", "arguments": "{\"command\":\"pwd\"}" },
                    }],
                }),
                None,
            ),
            delta_chunk(
                json!({
                    "tool_calls": [{
                        "index": 1,
                        "id": "call_two",
                        "type": "function",
                        "function": { "name": "run_shell_command", "arguments": "{\"comm" },
                    }],
                }),
                None,
            ),
            delta_chunk(
                json!({ "tool_calls": [{ "index": 1, "function": { "arguments": "and\":\"ls\"}" } }] }),
                None,
            ),
            delta_chunk(json!({}), Some("tool_calls")),
        ],
        true,
    );
    let _mock = mock_completion(&mut server, body).await;

    let events = expect_ok(
        collect_events(
            user_only_transcript(),
            vec![shell_tool()],
            config_for(&server),
        )
        .await,
    );

    assert_eq!(
        events,
        vec![
            AgentEvent::ToolCall(ToolCallRequest {
                id: "call_one".to_string(),
                name: "run_shell_command".to_string(),
                arguments: json!({ "command": "pwd" }),
            }),
            AgentEvent::ToolCall(ToolCallRequest {
                id: "call_two".to_string(),
                name: "run_shell_command".to_string(),
                arguments: json!({ "command": "ls" }),
            }),
            AgentEvent::Done,
        ]
    );
}

#[tokio::test]
async fn emits_text_before_tool_calls_when_model_streams_both() {
    let mut server = Server::new_async().await;
    let body = sse_body(
        &[
            delta_chunk(
                json!({ "role": "assistant", "content": "Let me check." }),
                None,
            ),
            delta_chunk(
                json!({
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": { "name": "run_shell_command", "arguments": "{\"command\":\"ls\"}" },
                    }],
                }),
                None,
            ),
            delta_chunk(json!({}), Some("tool_calls")),
        ],
        true,
    );
    let _mock = mock_completion(&mut server, body).await;

    let events = expect_ok(
        collect_events(
            user_only_transcript(),
            vec![shell_tool()],
            config_for(&server),
        )
        .await,
    );

    assert_eq!(
        events,
        vec![
            AgentEvent::TextDelta("Let me check.".to_string()),
            AgentEvent::ToolCall(ToolCallRequest {
                id: "call_abc".to_string(),
                name: "run_shell_command".to_string(),
                arguments: json!({ "command": "ls" }),
            }),
            AgentEvent::Done,
        ]
    );
}

#[tokio::test]
async fn emits_reasoning_deltas_from_reasoning_content_field() {
    let mut server = Server::new_async().await;
    // `reasoning_content` is the de-facto field used by DeepSeek-style
    // OpenAI-compatible servers for thinking tokens.
    let body = sse_body(
        &[
            delta_chunk(
                json!({ "role": "assistant", "reasoning_content": "thinking..." }),
                None,
            ),
            delta_chunk(json!({ "content": "answer" }), None),
            delta_chunk(json!({}), Some("stop")),
        ],
        true,
    );
    let _mock = mock_completion(&mut server, body).await;

    let events =
        expect_ok(collect_events(user_only_transcript(), vec![], config_for(&server)).await);

    assert_eq!(
        events,
        vec![
            AgentEvent::ReasoningDelta("thinking...".to_string()),
            AgentEvent::TextDelta("answer".to_string()),
            AgentEvent::Done,
        ]
    );
}

#[tokio::test]
async fn sends_chat_completions_request_with_expected_shape() {
    let mut server = Server::new_async().await;
    let body = sse_body(
        &[delta_chunk(json!({ "content": "ok" }), Some("stop"))],
        true,
    );
    let mock = server
        .mock("POST", "/chat/completions")
        .match_header("authorization", "Bearer test-key")
        .match_header("content-type", "application/json")
        .match_body(Matcher::AllOf(vec![
            Matcher::PartialJson(json!({
                "model": "test-model",
                "stream": true,
                "stream_options": { "include_usage": true },
                "messages": [
                    { "role": "system", "content": "You are a coding agent." },
                    { "role": "user", "content": "hi" },
                    {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_prev",
                            "type": "function",
                            "function": {
                                "name": "run_shell_command",
                                "arguments": "{\"command\":\"ls\"}",
                            },
                        }],
                    },
                    { "role": "tool", "tool_call_id": "call_prev", "content": "file.txt" },
                ],
            })),
            Matcher::PartialJson(json!({
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "run_shell_command",
                        "description": "Run a shell command",
                    },
                }],
            })),
        ]))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(body)
        .create_async()
        .await;

    let transcript = vec![
        ChatMessage::System("You are a coding agent.".to_string()),
        ChatMessage::User("hi".to_string()),
        ChatMessage::Assistant {
            text: String::new(),
            tool_calls: vec![ToolCallRequest {
                id: "call_prev".to_string(),
                name: "run_shell_command".to_string(),
                arguments: json!({ "command": "ls" }),
            }],
        },
        ChatMessage::Tool {
            tool_call_id: "call_prev".to_string(),
            content: "file.txt".to_string(),
        },
    ];

    let events =
        expect_ok(collect_events(transcript, vec![shell_tool()], config_for(&server)).await);

    mock.assert_async().await;
    assert_eq!(
        events,
        vec![AgentEvent::TextDelta("ok".to_string()), AgentEvent::Done]
    );
}

#[tokio::test]
async fn base_url_already_ending_in_chat_completions_is_used_as_is() {
    let mut server = Server::new_async().await;
    let body = sse_body(
        &[delta_chunk(json!({ "content": "ok" }), Some("stop"))],
        true,
    );
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(body)
        .create_async()
        .await;

    let config = LocalEndpointConfig {
        base_url: format!("{}/chat/completions", server.url()),
        api_key: "test-key".to_string(),
        model_slug: "test-model".to_string(),
    };

    let events = expect_ok(collect_events(user_only_transcript(), vec![], config).await);

    mock.assert_async().await;
    assert_eq!(
        events,
        vec![AgentEvent::TextDelta("ok".to_string()), AgentEvent::Done]
    );
}

#[tokio::test]
async fn maps_401_to_invalid_api_key() {
    let mut server = Server::new_async().await;
    let _mock = server
        .mock("POST", "/chat/completions")
        .with_status(401)
        .with_body(r#"{"error":{"message":"Incorrect API key provided"}}"#)
        .create_async()
        .await;

    let results = collect_events(user_only_transcript(), vec![], config_for(&server)).await;

    assert_eq!(results.len(), 1);
    assert!(
        matches!(
            &results[0],
            Err(LocalAgentError::InvalidApiKey { model_slug }) if model_slug == "test-model"
        ),
        "expected InvalidApiKey, got {:?}",
        results[0]
    );
}

#[tokio::test]
async fn maps_context_overflow_400_to_context_window_exceeded() {
    let mut server = Server::new_async().await;
    let _mock = server
        .mock("POST", "/chat/completions")
        .with_status(400)
        .with_body(
            r#"{"error":{"message":"This model's maximum context length is 8192 tokens.","code":"context_length_exceeded"}}"#,
        )
        .create_async()
        .await;

    let results = collect_events(user_only_transcript(), vec![], config_for(&server)).await;

    assert_eq!(results.len(), 1);
    assert!(
        matches!(
            &results[0],
            Err(LocalAgentError::ContextWindowExceeded { .. })
        ),
        "expected ContextWindowExceeded, got {:?}",
        results[0]
    );
}

#[tokio::test]
async fn maps_server_errors_to_error_status() {
    let mut server = Server::new_async().await;
    let _mock = server
        .mock("POST", "/chat/completions")
        .with_status(503)
        .with_body("upstream unavailable")
        .create_async()
        .await;

    let results = collect_events(user_only_transcript(), vec![], config_for(&server)).await;

    assert_eq!(results.len(), 1);
    assert!(
        matches!(
            &results[0],
            Err(LocalAgentError::ErrorStatus { status: 503, .. })
        ),
        "expected ErrorStatus(503), got {:?}",
        results[0]
    );
}

#[tokio::test]
async fn stream_ending_without_finish_reason_is_a_transport_error() {
    let mut server = Server::new_async().await;
    // The connection closes after one delta: no finish_reason, no [DONE].
    let body = sse_body(&[delta_chunk(json!({ "content": "partial" }), None)], false);
    let _mock = mock_completion(&mut server, body).await;

    let results = collect_events(user_only_transcript(), vec![], config_for(&server)).await;

    assert_eq!(
        results[0]
            .as_ref()
            .expect("first event should be the delta"),
        &AgentEvent::TextDelta("partial".to_string())
    );
    assert!(
        matches!(results.last(), Some(Err(LocalAgentError::Transport(_)))),
        "expected trailing Transport error, got {:?}",
        results.last()
    );
    assert!(
        !results.iter().any(|r| matches!(r, Ok(AgentEvent::Done))),
        "an interrupted stream must not report Done"
    );
}

#[tokio::test]
async fn build_suggestion_system_prompt_pushes_a_single_actionable_suggestion() {
    let env = EnvironmentInfo {
        pwd: Some("/home/user/project".to_string()),
        ..Default::default()
    };

    let prompt = build_suggestion_system_prompt(&env);

    // Still environment-aware.
    assert!(prompt.contains("/home/user/project"));
    // Uses the suggestion tool.
    assert!(prompt.contains("suggest_prompt"));
    // Pushes a terse, runnable suggestion rather than an open-ended one.
    assert!(
        prompt.contains("actionable"),
        "suggestion prompt should ask for an actionable suggestion:\n{prompt}"
    );
    // Discourages the agent from over-working the accepted suggestion.
    assert!(
        prompt.to_lowercase().contains("do not"),
        "suggestion prompt should discourage extra work:\n{prompt}"
    );
}

#[tokio::test]
async fn build_system_prompt_includes_environment_details() {
    let env = EnvironmentInfo {
        pwd: Some("/home/user/project".to_string()),
        operating_system: Some("linux".to_string()),
        shell: Some("zsh".to_string()),
        git_branch: Some("feature/x".to_string()),
        project_rules: vec!["Always run tests before committing.".to_string()],
        ..Default::default()
    };

    let prompt = build_system_prompt(&env);

    for needle in [
        "/home/user/project",
        "linux",
        "zsh",
        "feature/x",
        "Always run tests before committing.",
    ] {
        assert!(
            prompt.contains(needle),
            "prompt is missing {needle:?}:\n{prompt}"
        );
    }
}
