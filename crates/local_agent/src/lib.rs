//! A local agent loop that talks directly to an OpenAI-compatible API
//! (Chat Completions), independent of the Warp API server.
//!
//! This crate is deliberately free of `warp_multi_agent_api` (protobuf) types:
//! it operates on its own neutral model. The conversion between Warp's
//! protobuf request/response types and this model lives in a thin adapter in
//! the `app` crate (`app/src/ai/agent/api/local_adapter.rs`).
//!
//! Like the server-side loop, `run_turn` performs a single LLM call per
//! request: the host executes any emitted tool calls and continues the
//! conversation with a follow-up `run_turn` whose transcript includes the
//! tool results.

mod engine;
mod openai;
mod prompt;

use futures::stream::BoxStream;
pub use prompt::{EnvironmentInfo, build_suggestion_system_prompt, build_system_prompt};

/// Connection parameters for one OpenAI-compatible endpoint, resolved by the
/// host from the user's custom endpoint configuration.
#[derive(Debug, Clone)]
pub struct LocalEndpointConfig {
    /// Base URL of the endpoint, e.g. `https://api.example.com/v1`. A URL
    /// already ending in `/chat/completions` is used as-is.
    pub base_url: String,
    pub api_key: String,
    /// Model name as the provider knows it (e.g. `llama-3.3-70b`).
    pub model_slug: String,
}

/// One entry of the LLM transcript, in provider-neutral form.
#[derive(Debug, Clone, PartialEq)]
pub enum ChatMessage {
    System(String),
    User(String),
    Assistant {
        text: String,
        tool_calls: Vec<ToolCallRequest>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

/// A tool exposed to the model, with a JSON-schema parameter description.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// A completed tool call requested by the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallRequest {
    /// Provider-issued id, round-tripped back as the `tool_call_id` of the
    /// corresponding [`ChatMessage::Tool`] result.
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Events produced by a single agent turn, in stream order.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentEvent {
    TextDelta(String),
    ReasoningDelta(String),
    /// Emitted once a streamed tool call is fully accumulated.
    ToolCall(ToolCallRequest),
    Usage {
        input_tokens: u64,
        output_tokens: u64,
    },
    /// Always the final event of a successful turn.
    Done,
}

#[derive(Debug, thiserror::Error)]
pub enum LocalAgentError {
    /// The endpoint rejected our credentials (HTTP 401/403).
    #[error("invalid API key for {model_slug}")]
    InvalidApiKey { model_slug: String },
    /// The request exceeded the model's context window.
    #[error("context window exceeded: {message}")]
    ContextWindowExceeded { message: String },
    /// Any other non-success HTTP status. Retryable statuses (429/5xx) are
    /// mapped by the host onto its existing retry machinery.
    #[error("endpoint returned {status}: {body}")]
    ErrorStatus { status: u16, body: String },
    /// Network/transport failure, including mid-stream disconnects.
    #[error("transport error: {0}")]
    Transport(String),
    /// The endpoint returned something we could not parse.
    #[error("malformed response: {0}")]
    Parse(String),
    #[error("not implemented: {0}")]
    Unimplemented(&'static str),
}

/// Runs one turn of the agent loop against the configured endpoint, streaming
/// neutral [`AgentEvent`]s. Dropping the stream cancels the request.
pub fn run_turn(
    transcript: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
    config: LocalEndpointConfig,
) -> BoxStream<'static, Result<AgentEvent, LocalAgentError>> {
    engine::run_turn(transcript, tools, config)
}

/// Runs a single, tool-less completion against the endpoint and returns the
/// model's full text reply. Used by lightweight features (e.g. next-command
/// prediction) that want one answer rather than an agent loop.
pub async fn complete_text(
    transcript: Vec<ChatMessage>,
    config: LocalEndpointConfig,
) -> Result<String, LocalAgentError> {
    use futures::StreamExt;

    let mut stream = engine::run_turn(transcript, vec![], config);
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::TextDelta(delta) => text.push_str(&delta),
            AgentEvent::Done => break,
            // Reasoning, usage, and (absent) tool calls are irrelevant here.
            _ => {}
        }
    }
    Ok(text)
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
