//! Chat Completions wire types and the SSE delta accumulator.
//!
//! Implemented against the OpenAI Chat Completions streaming format
//! (`stream: true`, `stream_options: {"include_usage": true}`), tolerating
//! the quirks of common OpenAI-compatible servers (vLLM, llama.cpp, Ollama,
//! LiteLLM, OpenRouter): tool-call arguments fragmented across chunks,
//! continuation chunks without an `id`, and nonstandard reasoning fields.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    AgentEvent, ChatMessage, LocalAgentError, LocalEndpointConfig, ToolCallRequest, ToolDefinition,
};

/// Resolves the Chat Completions URL for an endpoint. Users configure either
/// a base URL (e.g. `https://host/v1`) or the full completions URL; a URL
/// already ending in `/chat/completions` is used as-is.
pub(crate) fn endpoint_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat/completions")
    }
}

#[derive(Serialize)]
pub(crate) struct ChatRequest {
    model: String,
    messages: Vec<OutgoingMessage>,
    stream: bool,
    stream_options: StreamOptions,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OutgoingTool>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct OutgoingMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OutgoingToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct OutgoingToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OutgoingFunctionCall,
}

#[derive(Serialize)]
struct OutgoingFunctionCall {
    name: String,
    /// JSON-encoded arguments, as the wire format requires.
    arguments: String,
}

#[derive(Serialize)]
struct OutgoingTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OutgoingFunctionDef,
}

#[derive(Serialize)]
struct OutgoingFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

pub(crate) fn build_request(
    transcript: &[ChatMessage],
    tools: &[ToolDefinition],
    config: &LocalEndpointConfig,
) -> ChatRequest {
    let messages = transcript
        .iter()
        .map(|message| match message {
            ChatMessage::System(text) => OutgoingMessage {
                role: "system",
                content: Some(text.clone()),
                tool_calls: vec![],
                tool_call_id: None,
            },
            ChatMessage::User(text) => OutgoingMessage {
                role: "user",
                content: Some(text.clone()),
                tool_calls: vec![],
                tool_call_id: None,
            },
            ChatMessage::Assistant { text, tool_calls } => OutgoingMessage {
                role: "assistant",
                content: (!text.is_empty()).then(|| text.clone()),
                tool_calls: tool_calls
                    .iter()
                    .map(|call| OutgoingToolCall {
                        id: call.id.clone(),
                        kind: "function",
                        function: OutgoingFunctionCall {
                            name: call.name.clone(),
                            arguments: call.arguments.to_string(),
                        },
                    })
                    .collect(),
                tool_call_id: None,
            },
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => OutgoingMessage {
                role: "tool",
                content: Some(content.clone()),
                tool_calls: vec![],
                tool_call_id: Some(tool_call_id.clone()),
            },
        })
        .collect();

    ChatRequest {
        model: config.model_slug.clone(),
        messages,
        stream: true,
        stream_options: StreamOptions {
            include_usage: true,
        },
        tools: tools
            .iter()
            .map(|tool| OutgoingTool {
                kind: "function",
                function: OutgoingFunctionDef {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: tool.parameters.clone(),
                },
            })
            .collect(),
    }
}

#[derive(Deserialize)]
pub(crate) struct ChunkResponse {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    usage: Option<ChunkUsage>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: ChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct ChunkDelta {
    content: Option<String>,
    // Reasoning arrives under different names depending on the server
    // (llama.cpp/DeepSeek: `reasoning_content`, others: `reasoning` or
    // `reasoning_text`); some servers populate several with the same text.
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    reasoning_text: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallDelta>,
}

impl ChunkDelta {
    fn reasoning_delta(&self) -> Option<&str> {
        [
            &self.reasoning_content,
            &self.reasoning,
            &self.reasoning_text,
        ]
        .into_iter()
        .find_map(|field| field.as_deref().filter(|text| !text.is_empty()))
    }
}

#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: u32,
    id: Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct ChunkUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

/// Folds streamed chunks into [`AgentEvent`]s. Text and reasoning deltas pass
/// through immediately; tool calls accumulate fragmented arguments per
/// `delta.tool_calls[*].index` and are emitted together once the choice
/// reports a `finish_reason`.
#[derive(Default)]
pub(crate) struct StreamAccumulator {
    pending_tool_calls: BTreeMap<u32, PartialToolCall>,
    finish_reason_seen: bool,
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl StreamAccumulator {
    pub(crate) fn ingest(
        &mut self,
        chunk: ChunkResponse,
    ) -> Result<Vec<AgentEvent>, LocalAgentError> {
        let mut events = Vec::new();
        for choice in chunk.choices {
            if let Some(reasoning) = choice.delta.reasoning_delta() {
                events.push(AgentEvent::ReasoningDelta(reasoning.to_string()));
            }
            if let Some(text) = choice.delta.content.filter(|text| !text.is_empty()) {
                events.push(AgentEvent::TextDelta(text));
            }
            for delta in choice.delta.tool_calls {
                let partial = self.pending_tool_calls.entry(delta.index).or_default();
                if let Some(id) = delta.id.filter(|id| !id.is_empty()) {
                    partial.id = id;
                }
                if let Some(function) = delta.function {
                    if let Some(name) = function.name.filter(|name| !name.is_empty()) {
                        partial.name = name;
                    }
                    if let Some(arguments) = function.arguments {
                        partial.arguments.push_str(&arguments);
                    }
                }
            }
            if choice.finish_reason.is_some() {
                self.finish_reason_seen = true;
                events.extend(self.flush_tool_calls()?);
            }
        }
        if let Some(usage) = chunk.usage {
            events.push(AgentEvent::Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
            });
        }
        Ok(events)
    }

    /// Whether the model reported a `finish_reason`; streams that end without
    /// one were cut off mid-response.
    pub(crate) fn finished(&self) -> bool {
        self.finish_reason_seen
    }

    pub(crate) fn flush_tool_calls(&mut self) -> Result<Vec<AgentEvent>, LocalAgentError> {
        std::mem::take(&mut self.pending_tool_calls)
            .into_values()
            .map(|partial| {
                let arguments = if partial.arguments.trim().is_empty() {
                    serde_json::Value::Object(Default::default())
                } else {
                    serde_json::from_str(&partial.arguments).map_err(|err| {
                        LocalAgentError::Parse(format!(
                            "tool call {} has malformed arguments: {err}",
                            partial.name
                        ))
                    })?
                };
                Ok(AgentEvent::ToolCall(ToolCallRequest {
                    id: partial.id,
                    name: partial.name,
                    arguments,
                }))
            })
            .collect()
    }
}

pub(crate) fn parse_chunk(data: &str) -> Result<ChunkResponse, LocalAgentError> {
    serde_json::from_str(data)
        .map_err(|err| LocalAgentError::Parse(format!("malformed chunk: {err}: {data}")))
}

fn looks_like_context_overflow(body: &str) -> bool {
    let body = body.to_ascii_lowercase();
    [
        "context_length_exceeded",
        "maximum context length",
        "context window",
        "too many tokens",
        "prompt is too long",
    ]
    .iter()
    .any(|marker| body.contains(marker))
}

/// Maps an [`EventSource`](reqwest_eventsource::EventSource) failure onto
/// [`LocalAgentError`]. `StreamEnded` is not an error and is handled by the
/// engine before calling this.
pub(crate) async fn map_event_source_error(
    err: reqwest_eventsource::Error,
    config: &LocalEndpointConfig,
) -> LocalAgentError {
    use reqwest_eventsource::Error;
    match err {
        Error::InvalidStatusCode(status, response) => {
            let body = response.text().await.unwrap_or_default();
            match status.as_u16() {
                401 | 403 => LocalAgentError::InvalidApiKey {
                    model_slug: config.model_slug.clone(),
                },
                400 if looks_like_context_overflow(&body) => {
                    LocalAgentError::ContextWindowExceeded { message: body }
                }
                status => LocalAgentError::ErrorStatus { status, body },
            }
        }
        Error::InvalidContentType(content_type, response) => {
            let body = response.text().await.unwrap_or_default();
            LocalAgentError::Parse(format!(
                "endpoint did not return an event stream (content-type {content_type:?}): {body}"
            ))
        }
        Error::Transport(err) => LocalAgentError::Transport(err.to_string()),
        Error::Utf8(err) => LocalAgentError::Parse(err.to_string()),
        Error::Parser(err) => LocalAgentError::Parse(err.to_string()),
        Error::InvalidLastEventId(id) => {
            LocalAgentError::Parse(format!("invalid last-event-id: {id}"))
        }
        Error::StreamEnded => {
            LocalAgentError::Transport("stream ended before completion".to_string())
        }
    }
}

/// The host owns retries (Warp's `ResponseStream` already retries retryable
/// failures); the SSE layer must surface errors instead of reconnecting.
pub(crate) struct NeverRetry;

impl reqwest_eventsource::retry::RetryPolicy for NeverRetry {
    fn retry(
        &self,
        _error: &reqwest_eventsource::Error,
        _last_retry: Option<(usize, std::time::Duration)>,
    ) -> Option<std::time::Duration> {
        None
    }

    fn set_reconnection_time(&mut self, _duration: std::time::Duration) {}
}
