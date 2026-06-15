//! End-to-end tests for the local agent loop.
//!
//! A mockito server plays an OpenAI-compatible Chat Completions endpoint.
//! The user's custom endpoint settings point at it, so the agent loop runs
//! entirely client-side: the app talks straight to the stub and never
//! contacts the Warp API server (which isn't reachable from the hermetic
//! test environment in the first place — a green run is itself proof that
//! requests were routed locally).

use std::time::Duration;

use warp::features::FeatureFlag;
use warp::integration_testing::agent_mode::{
    ConversationTarget, add_custom_model_endpoint, assert_any_exchange_text,
    assert_exchange_text_contains, assert_turned_finish_with_success, enter_agent_view,
    set_preferred_agent_mode_custom_llm, submit_ai_query, submit_ai_query_and_wait_until_done,
    user_defaults_map_with_active_ai,
};
use warp::integration_testing::step::new_step_with_default_assertions;
use warp::integration_testing::terminal::{
    clear_blocklist_to_remove_bootstrapped_blocks, execute_echo_str,
    wait_until_bootstrapped_single_pane_for_tab,
};
use warp::integration_testing::view_getters::single_input_view_for_tab;
use warpui_core::async_assert;
use warpui_core::integration::TestStep;

use super::new_builder;
use crate::Builder;

const API_KEY: &str = "integration-test-key";
const MODEL_NAME: &str = "stub-model";
const CONFIG_KEY: &str = "11111111-2222-3333-4444-555555555555";
const AGENT_TIMEOUT: Duration = Duration::from_secs(60);
const AUTO_EXECUTE_PROFILE: &str = r#"
[agents.execution_profiles.default]
name = "Integration Test"
execute_commands = "always_allow"
"#;

fn write_auto_execute_profile() {
    let path = warp::settings::user_preferences_toml_file_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("should create settings directory");
    }
    std::fs::write(path, AUTO_EXECUTE_PROFILE).expect("should write execution profile settings");
}

/// One `chat.completion.chunk` SSE event with the given choice delta.
fn delta_chunk(delta: serde_json::Value, finish_reason: Option<&str>) -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-integration",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL_NAME,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
    })
}

/// Renders chunks as a complete `text/event-stream` body ending in `[DONE]`.
fn sse_body(chunks: &[serde_json::Value]) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

/// A streamed plain-text assistant reply.
fn text_reply_sse(text: &str) -> String {
    sse_body(&[
        delta_chunk(
            serde_json::json!({ "role": "assistant", "content": text }),
            None,
        ),
        delta_chunk(serde_json::json!({}), Some("stop")),
    ])
}

/// A reply that calls the `run_shell_command` tool with the given command.
fn shell_tool_call_sse(command: &str) -> String {
    let arguments = serde_json::json!({ "command": command, "is_read_only": true }).to_string();
    sse_body(&[
        delta_chunk(
            serde_json::json!({
                "role": "assistant",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_integration_1",
                    "type": "function",
                    "function": { "name": "run_shell_command", "arguments": arguments },
                }],
            }),
            None,
        ),
        delta_chunk(serde_json::json!({}), Some("tool_calls")),
    ])
}

/// Leaks a mockito server so it outlives the `Builder` factory and serves
/// requests for the whole test run. The hermetic test process exits when the
/// test is over, so the leak is inconsequential.
fn leaked_stub_server() -> &'static mut mockito::ServerGuard {
    Box::leak(Box::new(mockito::Server::new()))
}

/// Shared opening steps: point a custom endpoint at the stub server and make
/// its model the preferred agent mode LLM.
///
/// Note: passive prompt suggestions run through the local path by default, so
/// once a block completes they fire one extra request to the same stub. These
/// tests therefore assert "at least N" rather than exact request counts; the
/// passive flow has its own dedicated test.
fn builder_routing_to(server_url: &str) -> Builder {
    FeatureFlag::CustomInferenceEndpoints.set_enabled(true);

    new_builder()
        .with_step(wait_until_bootstrapped_single_pane_for_tab(0))
        .with_step(add_custom_model_endpoint(
            "Stub OpenAI endpoint",
            server_url,
            API_KEY,
            MODEL_NAME,
            CONFIG_KEY,
        ))
        .with_step(set_preferred_agent_mode_custom_llm(CONFIG_KEY))
        .with_step(enter_agent_view())
}

/// The stub streams a plain text reply; the agent view must render it and
/// finish the task. Covers routing, the Chat Completions request shape, SSE
/// text accumulation, and the Init/CreateTask/AddMessages/append event chain.
pub fn test_local_agent_loop_streams_text_reply() -> Builder {
    let server = leaked_stub_server();
    let mock: &'static mockito::Mock = Box::leak(Box::new(
        server
            .mock("POST", "/chat/completions")
            .match_header("authorization", &*format!("Bearer {API_KEY}"))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(text_reply_sse(
                "Hello from the local agent loop integration stub!",
            ))
            .expect_at_least(1)
            .create(),
    ));

    builder_routing_to(&server.url())
        .with_step(submit_ai_query_and_wait_until_done(
            "Say hello please",
            AGENT_TIMEOUT,
        ))
        .with_step(
            new_step_with_default_assertions("Verify the stub reply rendered")
                .add_named_assertion(
                    "Agent output contains the stub text",
                    assert_exchange_text_contains(
                        0,
                        "Hello from the local agent loop integration stub!",
                    ),
                )
                .add_named_assertion("Stub endpoint served the agent request", |_, _| {
                    async_assert!(
                        mock.matched(),
                        "Expected at least one request to the stub /chat/completions"
                    )
                }),
        )
}

/// The stub first requests a `run_shell_command` tool call; the client
/// executes it (default profile auto-executes) and sends the result back as
/// a new request, to which the stub streams the final text. Covers the
/// tool-call delta accumulation, proto tool-call mapping, executor round-trip
/// of `tool_call_id`, and transcript replay on the follow-up turn.
pub fn test_local_agent_loop_shell_tool_round_trip() -> Builder {
    FeatureFlag::SettingsFile.set_enabled(true);
    FeatureFlag::FileBackedExecutionProfiles.set_enabled(true);

    let server = leaked_stub_server();
    let tool_call_reply = shell_tool_call_sse("echo local-loop-roundtrip");
    let final_reply = text_reply_sse("Round trip complete, the command ran fine.");
    let mock: &'static mockito::Mock = Box::leak(Box::new(
        server
            .mock("POST", "/chat/completions")
            .match_header("authorization", &*format!("Bearer {API_KEY}"))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body_from_request(move |request| {
                let body = request
                    .body()
                    .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                    .unwrap_or_default();
                // The follow-up turn replays the transcript with the tool
                // result attached; the first turn has no tool messages yet.
                if body.contains("\"role\":\"tool\"") {
                    final_reply.clone().into_bytes()
                } else {
                    tool_call_reply.clone().into_bytes()
                }
            })
            .expect_at_least(2)
            .create(),
    ));

    builder_routing_to(&server.url())
        // Runtime profile edits are deferred while the initial cloud-backed
        // profile migration is pending. Seed an explicit file-backed profile
        // before app initialization so the hermetic test can auto-execute.
        .with_setup(|_| write_auto_execute_profile())
        .with_step(
            // An auto-executed terminal action can retain focus on its rendered
            // action block. Focus is unrelated to the local agent-loop contract,
            // so wait for task completion here and verify the round trip below.
            submit_ai_query("Run a quick echo for me", AGENT_TIMEOUT).add_named_assertion(
                "Assert the local agent task is complete",
                assert_turned_finish_with_success(ConversationTarget::Active, None),
            ),
        )
        .with_step(
            new_step_with_default_assertions("Verify the tool round trip")
                .add_named_assertion(
                    "Final agent output contains the stub text",
                    assert_any_exchange_text(|text| {
                        text.contains("Round trip complete, the command ran fine.")
                    }),
                )
                .add_named_assertion("Stub endpoint served the tool round trip", |_, _| {
                    async_assert!(
                        mock.matched(),
                        "Expected at least two requests to the stub /chat/completions"
                    )
                }),
        )
}

/// A reply that calls the `suggest_prompt` tool — what the model returns for a
/// passive prompt-suggestion turn.
fn suggest_prompt_tool_call_sse(prompt: &str, label: &str) -> String {
    let arguments = serde_json::json!({ "prompt": prompt, "label": label }).to_string();
    sse_body(&[
        delta_chunk(
            serde_json::json!({
                "role": "assistant",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_suggest_1",
                    "type": "function",
                    "function": { "name": "suggest_prompt", "arguments": arguments },
                }],
            }),
            None,
        ),
        delta_chunk(serde_json::json!({}), Some("tool_calls")),
    ])
}

/// End-to-end test of passive prompt suggestions on the local path. After a
/// shell command completes, the passive-suggestions trigger must go through the
/// MAA (local) path — not the server-backed legacy path — reach the custom
/// endpoint, and surface the model's `suggest_prompt` as a prompt-suggestion
/// banner. This is the path that the unit tests bypass (they drive the adapter
/// directly), so it guards the MAA-vs-legacy selection and the wiring.
pub fn test_local_agent_loop_passive_prompt_suggestion() -> Builder {
    FeatureFlag::CustomInferenceEndpoints.set_enabled(true);
    // Route passive suggestions through the local multi-agent path.
    FeatureFlag::PromptSuggestionsViaMAA.set_enabled(true);

    const SUGGESTED_PROMPT: &str = "Run `fd -e go .` to find Go files and show the output";
    const SUGGESTED_LABEL: &str = "Fix fd usage";

    let server = leaked_stub_server();
    let mock: &'static mockito::Mock = Box::leak(Box::new(
        server
            .mock("POST", "/chat/completions")
            .match_header("authorization", &*format!("Bearer {API_KEY}"))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(suggest_prompt_tool_call_sse(
                SUGGESTED_PROMPT,
                SUGGESTED_LABEL,
            ))
            .expect_at_least(1)
            .create(),
    ));

    new_builder()
        // Enables active AI + prompt suggestions (AgentModeQuerySuggestionsEnabled).
        .with_user_defaults(user_defaults_map_with_active_ai(true))
        .with_step(wait_until_bootstrapped_single_pane_for_tab(0))
        .with_step(clear_blocklist_to_remove_bootstrapped_blocks())
        .with_step(add_custom_model_endpoint(
            "Stub OpenAI endpoint",
            &server.url(),
            API_KEY,
            MODEL_NAME,
            CONFIG_KEY,
        ))
        .with_step(set_preferred_agent_mode_custom_llm(CONFIG_KEY))
        // Completing a user block triggers passive prompt suggestions.
        .with_step(execute_echo_str(0, "hi"))
        .with_step(
            TestStep::new("Wait for the local passive prompt suggestion banner")
                .set_timeout(Duration::from_secs(60))
                .add_named_assertion(
                    "prompt-suggestion banner shows the stub's suggested prompt",
                    move |app, window_id| {
                        let input = single_input_view_for_tab(app, window_id, 0);
                        input.read(app, |input, _| {
                            let shown = input.prompt_suggestion_banner_prompt();
                            async_assert!(
                                shown.as_deref() == Some(SUGGESTED_PROMPT),
                                "expected local passive prompt-suggestion banner; got {shown:?}"
                            )
                        })
                    },
                )
                .add_named_assertion("the custom endpoint served the suggestion", move |_, _| {
                    async_assert!(
                        mock.matched(),
                        "passive suggestion must hit the local endpoint, not the server"
                    )
                }),
        )
}
