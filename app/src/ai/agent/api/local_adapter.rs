//! The single place where the proto request/response model of the agent API
//! meets the neutral model of `crates/local_agent`.
//!
//! Responsibilities (mirroring what `convert_to.rs`/`convert_from.rs` do for
//! the server path):
//! - rebuild the LLM transcript from `request.task_context` + `request.input`,
//! - expose builtin + MCP tools as JSON-schema [`local_agent::ToolDefinition`]s
//!   and map completed tool calls back to proto `ToolCall` messages,
//! - turn the engine's [`local_agent::AgentEvent`] stream into the
//!   `ResponseEvent` sequence the existing client machinery consumes
//!   (StreamInit, transactions, CreateTask/AddMessagesToTask/
//!   AppendToMessageContent, Finished).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::anyhow;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;
use warp_multi_agent_api as api;

use super::{Event, ResponseStream};
use crate::ai::agent::api::local_route::LocalRoute;
use crate::server::server_api::AIApiError;

#[derive(Debug, thiserror::Error)]
pub(crate) enum AdapterError {
    #[error("unsupported input for local agent loop: {0}")]
    UnsupportedInput(&'static str),
    #[error("unknown tool call from model: {0}")]
    UnknownTool(String),
    #[error("malformed tool arguments for {tool}: {message}")]
    BadToolArguments { tool: String, message: String },
}

/// Everything derived from one proto request that the engine and the event
/// mapper need: the rebuilt transcript, the tool registry, and turn identity
/// (conversation id, root task id, whether this is the first turn).
pub(crate) struct TurnPlan {
    pub transcript: Vec<local_agent::ChatMessage>,
    pub tools: ToolRegistry,
    pub conversation_id: String,
    pub request_id: String,
    pub root_task_id: String,
    pub is_first_turn: bool,
    /// Input messages to echo into the task tree so the next turn's
    /// transcript rebuild sees them (user query, tool call results).
    pub input_echo_messages: Vec<api::Message>,
    pub config_key: String,
}

/// Identity of one turn shared by every turn kind: which conversation and
/// task it belongs to, and whether the task tree already exists.
struct TurnIdentity {
    conversation_id: String,
    root_task_id: String,
    is_first_turn: bool,
}

fn turn_identity(request: &api::Request) -> TurnIdentity {
    let tasks = request_tasks(request);
    TurnIdentity {
        is_first_turn: tasks.is_empty(),
        root_task_id: tasks
            .first()
            .map(|task| task.id.clone())
            .unwrap_or_else(|| Uuid::new_v4().to_string()),
        conversation_id: request
            .metadata
            .as_ref()
            .map(|metadata| metadata.conversation_id.clone())
            .filter(|conversation_id| !conversation_id.is_empty())
            // The client parses conversation tokens as UUIDs, so mint one.
            .unwrap_or_else(|| Uuid::new_v4().to_string()),
    }
}

fn request_tasks(request: &api::Request) -> &[api::Task] {
    request
        .task_context
        .as_ref()
        .map(|task_context| task_context.tasks.as_slice())
        .unwrap_or_default()
}

/// Derives the [`TurnPlan`] from a fully-built proto request.
pub(crate) fn plan_turn(
    request: &api::Request,
    route: &LocalRoute,
) -> Result<TurnPlan, AdapterError> {
    let input = request
        .input
        .as_ref()
        .ok_or(AdapterError::UnsupportedInput("request has no input"))?;
    match input.r#type.as_ref() {
        Some(api::request::input::Type::UserInputs(user_inputs)) => {
            plan_agent_turn(request, input, &user_inputs.inputs, route)
        }
        Some(api::request::input::Type::GeneratePassiveSuggestions(generate)) => {
            plan_suggestion_turn(request, input, generate, route)
        }
        Some(_) => Err(AdapterError::UnsupportedInput(
            "system-driven inputs are served by the server path",
        )),
        None => Err(AdapterError::UnsupportedInput("input has no type")),
    }
}

/// Plans a regular agent-loop turn driven by user queries and tool results.
fn plan_agent_turn(
    request: &api::Request,
    input: &api::request::Input,
    user_inputs: &[api::request::input::user_inputs::UserInput],
    route: &LocalRoute,
) -> Result<TurnPlan, AdapterError> {
    use api::request::input::user_inputs::user_input::Input;

    let tasks = request_tasks(request);
    let TurnIdentity {
        conversation_id,
        root_task_id,
        is_first_turn,
    } = turn_identity(request);

    let tools = ToolRegistry::new(request.mcp_context.as_ref());

    let environment = environment_from_context(input.context.as_ref());
    let mut transcript = vec![local_agent::ChatMessage::System(
        local_agent::build_system_prompt(&environment),
    )];
    let mut known_call_ids = Vec::new();
    for task in tasks {
        for message in &task.messages {
            append_history_message(&mut transcript, &mut known_call_ids, message, &tools);
        }
    }

    let mut input_echo_messages = Vec::new();
    for user_input in user_inputs {
        match user_input.input.as_ref() {
            Some(Input::UserQuery(query)) => {
                transcript.push(local_agent::ChatMessage::User(query.query.clone()));
                input_echo_messages.push(new_task_message(
                    &root_task_id,
                    api::message::Message::UserQuery(api::message::UserQuery {
                        query: query.query.clone(),
                        context: input.context.clone(),
                        referenced_attachments: query.referenced_attachments.clone(),
                        mode: query.mode,
                        intended_agent: query.intended_agent,
                    }),
                ));
            }
            Some(Input::ToolCallResult(input_result)) => {
                let message_result = api::message::ToolCallResult {
                    tool_call_id: input_result.tool_call_id.clone(),
                    context: None,
                    result: convert_input_tool_result(input_result.result.as_ref()),
                };
                // If the task tree somehow lacks the originating tool call
                // (e.g. a forked or truncated conversation), synthesize one:
                // OpenAI-compatible servers reject `role:"tool"` messages
                // without a preceding assistant tool call.
                if !known_call_ids.contains(&input_result.tool_call_id) {
                    transcript.push(local_agent::ChatMessage::Assistant {
                        text: String::new(),
                        tool_calls: vec![local_agent::ToolCallRequest {
                            id: input_result.tool_call_id.clone(),
                            name: "tool".to_string(),
                            arguments: json!({}),
                        }],
                    });
                    known_call_ids.push(input_result.tool_call_id.clone());
                }
                transcript.push(local_agent::ChatMessage::Tool {
                    tool_call_id: input_result.tool_call_id.clone(),
                    content: render_tool_result_text(&message_result),
                });
                input_echo_messages.push(new_task_message(
                    &root_task_id,
                    api::message::Message::ToolCallResult(message_result),
                ));
            }
            Some(Input::PassiveSuggestionResult(result_input)) => {
                // The client sends an accepted passive suggestion only as a
                // PassiveSuggestionResult, with no accompanying UserQuery. Bring
                // the trigger (what the user just did) into context for this
                // turn, then treat the suggested prompt as the durable user
                // query: push it for this turn AND echo it into the task tree.
                // Without the echo, a follow-up turn (e.g. after a tool
                // round-trip) rebuilds a transcript with no query for this turn,
                // so the model answers the *previous* turn's question.
                if let Some(result) = result_input.result.as_ref() {
                    if let Some(trigger) = render_passive_suggestion_trigger(result) {
                        transcript.push(local_agent::ChatMessage::User(trigger));
                    }
                    if let Some(api::passive_suggestion_result_type::Suggestion::Prompt(prompt)) =
                        result.suggestion.as_ref()
                    {
                        transcript.push(local_agent::ChatMessage::User(prompt.prompt.clone()));
                        input_echo_messages.push(new_task_message(
                            &root_task_id,
                            api::message::Message::UserQuery(api::message::UserQuery {
                                query: prompt.prompt.clone(),
                                context: input.context.clone(),
                                ..Default::default()
                            }),
                        ));
                    }
                }
            }
            _ => {
                return Err(AdapterError::UnsupportedInput(
                    "only user queries, tool call results, and passive suggestion \
                     results are supported",
                ));
            }
        }
    }

    Ok(TurnPlan {
        transcript,
        tools,
        conversation_id,
        request_id: Uuid::new_v4().to_string(),
        root_task_id,
        is_first_turn,
        input_echo_messages,
        config_key: route.config_key.clone(),
    })
}

/// Plans a passive prompt suggestion turn: a one-shot, read-only request
/// whose only tool is `suggest_prompt`. The conversation history (if any) is
/// replayed for context, then the trigger is rendered as the final user
/// instruction.
fn plan_suggestion_turn(
    request: &api::Request,
    input: &api::request::Input,
    generate: &api::request::input::GeneratePassiveSuggestions,
    route: &LocalRoute,
) -> Result<TurnPlan, AdapterError> {
    use api::request::input::generate_passive_suggestions::Trigger;

    let trigger = generate
        .trigger
        .as_ref()
        .ok_or(AdapterError::UnsupportedInput(
            "passive suggestion request has no trigger",
        ))?;

    let tasks = request_tasks(request);
    let TurnIdentity {
        conversation_id,
        root_task_id,
        is_first_turn,
    } = turn_identity(request);

    let tools = ToolRegistry::for_passive_suggestions();

    let environment = environment_from_context(input.context.as_ref());
    let mut transcript = vec![local_agent::ChatMessage::System(
        local_agent::build_suggestion_system_prompt(&environment),
    )];
    let mut known_call_ids = Vec::new();
    for task in tasks {
        for message in &task.messages {
            append_history_message(&mut transcript, &mut known_call_ids, message, &tools);
        }
    }

    let instruction = match trigger {
        Trigger::ShellCommandCompleted(completed) => {
            let mut instruction = String::from(
                "The user just ran a shell command in their terminal. If a useful follow-up \
                 exists, propose it via the `suggest_prompt` tool.\n",
            );
            if let Some(command) = completed.executed_shell_command.as_ref() {
                instruction.push_str(&format!(
                    "\nCommand: {}\nExit code: {}\nOutput:\n{}\n",
                    command.command, command.exit_code, command.output
                ));
            }
            for file in &completed.relevant_files {
                if let Some(api::any_file_content::Content::TextContent(content)) =
                    file.content.as_ref()
                {
                    instruction.push_str(&format!(
                        "\n=== {} ===\n{}\n",
                        content.file_path, content.content
                    ));
                }
            }
            instruction
        }
        Trigger::AgentResponseCompleted(_) => String::from(
            "The agent just completed the response above. If a useful follow-up prompt \
             exists for this conversation, propose it via the `suggest_prompt` tool.",
        ),
        // Deprecated pre-MAA triggers; nothing sends them anymore.
        Trigger::FilesChanged(()) | Trigger::CommandRun(()) => {
            return Err(AdapterError::UnsupportedInput(
                "deprecated passive suggestion trigger",
            ));
        }
    };
    transcript.push(local_agent::ChatMessage::User(instruction));

    let input_echo_messages = vec![new_task_message(
        &root_task_id,
        api::message::Message::SystemQuery(api::message::SystemQuery {
            context: input.context.clone(),
            r#type: Some(
                api::message::system_query::Type::GeneratePassiveSuggestions(
                    api::message::GeneratePassiveSuggestions {
                        attachments: generate.attachments.clone(),
                        trigger: Some(message_trigger_from_request(trigger)),
                    },
                ),
            ),
        }),
    )];

    Ok(TurnPlan {
        transcript,
        tools,
        conversation_id,
        request_id: Uuid::new_v4().to_string(),
        root_task_id,
        is_first_turn,
        input_echo_messages,
        config_key: route.config_key.clone(),
    })
}

/// Renders the trigger that led to an accepted passive suggestion (what the
/// user just did) into a short context note for the transcript. This is
/// transient per-turn context only — the accepted prompt itself is pushed as
/// the durable user query. Returns `None` when there is no trigger.
fn render_passive_suggestion_trigger(result: &api::PassiveSuggestionResultType) -> Option<String> {
    use api::passive_suggestion_result_type::Trigger;

    match result.trigger.as_ref()? {
        Trigger::ExecutedShellCommand(cmd) => Some(format!(
            "Context: this request follows a passive suggestion.\n\
             Triggered by shell command: {}\nExit code: {}\nOutput:\n{}",
            cmd.command, cmd.exit_code, cmd.output
        )),
        Trigger::AgentResponseCompleted(_) => Some(String::from(
            "Context: this request follows a passive suggestion.\n\
             Triggered after the agent completed its previous response.",
        )),
    }
}

/// The request-level and message-level `GeneratePassiveSuggestions` triggers
/// are structurally identical but distinct generated types; convert for the
/// task-tree echo.
fn message_trigger_from_request(
    trigger: &api::request::input::generate_passive_suggestions::Trigger,
) -> api::message::generate_passive_suggestions::Trigger {
    use api::message::generate_passive_suggestions as message_gps;
    use api::request::input::generate_passive_suggestions as request_gps;

    match trigger {
        request_gps::Trigger::ShellCommandCompleted(completed) => {
            message_gps::Trigger::ShellCommandCompleted(message_gps::ShellCommandCompleted {
                executed_shell_command: completed.executed_shell_command.clone(),
                relevant_files: completed.relevant_files.clone(),
            })
        }
        request_gps::Trigger::AgentResponseCompleted(_) => {
            message_gps::Trigger::AgentResponseCompleted(message_gps::AgentResponseCompleted {})
        }
        request_gps::Trigger::FilesChanged(()) => message_gps::Trigger::FilesChanged(()),
        request_gps::Trigger::CommandRun(()) => message_gps::Trigger::CommandRun(()),
    }
}

fn new_task_message(task_id: &str, content: api::message::Message) -> api::Message {
    api::Message {
        id: Uuid::new_v4().to_string(),
        task_id: task_id.to_string(),
        request_id: String::new(),
        timestamp: None,
        server_message_data: String::new(),
        citations: vec![],
        fetched_memories: vec![],
        message: Some(content),
    }
}

/// Appends one task-tree message to the transcript. Messages the local loop
/// does not understand (reasoning, todos, server events, ...) are skipped;
/// tool calls of kinds the registry cannot express are skipped together with
/// their results so the transcript never references an undefined call id.
fn append_history_message(
    transcript: &mut Vec<local_agent::ChatMessage>,
    known_call_ids: &mut Vec<String>,
    message: &api::Message,
    tools: &ToolRegistry,
) {
    use api::message::Message;
    match message.message.as_ref() {
        Some(Message::UserQuery(query)) => {
            transcript.push(local_agent::ChatMessage::User(query.query.clone()));
        }
        Some(Message::AgentOutput(output)) => {
            if !output.text.is_empty() {
                transcript.push(local_agent::ChatMessage::Assistant {
                    text: output.text.clone(),
                    tool_calls: vec![],
                });
            }
        }
        Some(Message::ToolCall(tool_call)) => {
            let Some(request) = tools.to_request_tool_call(tool_call) else {
                log::warn!(
                    "local agent loop: dropping unsupported historic tool call {}",
                    tool_call.tool_call_id
                );
                return;
            };
            known_call_ids.push(request.id.clone());
            // Parallel calls and text from the same LLM turn must live in
            // one assistant message, or strict OpenAI-compatible servers
            // reject the tool results that follow.
            if let Some(local_agent::ChatMessage::Assistant { tool_calls, .. }) =
                transcript.last_mut()
            {
                tool_calls.push(request);
            } else {
                transcript.push(local_agent::ChatMessage::Assistant {
                    text: String::new(),
                    tool_calls: vec![request],
                });
            }
        }
        Some(Message::ToolCallResult(result)) => {
            if !known_call_ids.contains(&result.tool_call_id) {
                return;
            }
            transcript.push(local_agent::ChatMessage::Tool {
                tool_call_id: result.tool_call_id.clone(),
                content: render_tool_result_text(result),
            });
        }
        // Reasoning, model markers, todos, server events, etc. carry no
        // information the next LLM call needs.
        _ => {}
    }
}

fn environment_from_context(context: Option<&api::InputContext>) -> local_agent::EnvironmentInfo {
    fn non_empty(value: String) -> Option<String> {
        (!value.is_empty()).then_some(value)
    }

    let Some(context) = context else {
        return local_agent::EnvironmentInfo::default();
    };
    local_agent::EnvironmentInfo {
        pwd: context
            .directory
            .as_ref()
            .and_then(|directory| non_empty(directory.pwd.clone())),
        home: context
            .directory
            .as_ref()
            .and_then(|directory| non_empty(directory.home.clone())),
        shell: context
            .shell
            .as_ref()
            .and_then(|shell| non_empty(shell.name.clone())),
        operating_system: context.operating_system.as_ref().and_then(|os| {
            non_empty(
                format!("{} {}", os.platform, os.distribution)
                    .trim()
                    .to_string(),
            )
        }),
        git_branch: context
            .git
            .as_ref()
            .and_then(|git| non_empty(git.branch.clone())),
        git_repository: context.git.as_ref().and_then(|git| {
            git.repository.as_ref().map(|repository| {
                if repository.owner.is_empty() {
                    repository.name.clone()
                } else {
                    format!("{}/{}", repository.owner, repository.name)
                }
            })
        }),
        current_time: context.current_time.as_ref().map(|time| time.to_string()),
        project_rules: context
            .project_rules
            .iter()
            .flat_map(|rules| rules.active_rule_files.iter())
            .filter(|file| !file.content.is_empty())
            .map(|file| file.content.clone())
            .collect(),
    }
}

/// Converts a fresh tool-call-result input into its message-log counterpart
/// (the oneof tags differ between the two protos, so this is an explicit
/// match). Only results of tools the local loop can issue are mapped.
fn convert_input_tool_result(
    result: Option<&api::request::input::tool_call_result::Result>,
) -> Option<api::message::tool_call_result::Result> {
    use api::message::tool_call_result::Result as MessageResult;
    use api::request::input::tool_call_result::Result as InputResult;
    match result? {
        InputResult::RunShellCommand(result) => {
            Some(MessageResult::RunShellCommand(result.clone()))
        }
        InputResult::ReadFiles(result) => Some(MessageResult::ReadFiles(result.clone())),
        InputResult::ApplyFileDiffs(result) => Some(MessageResult::ApplyFileDiffs(result.clone())),
        InputResult::Grep(result) => Some(MessageResult::Grep(result.clone())),
        #[allow(deprecated)]
        InputResult::FileGlob(result) => Some(MessageResult::FileGlob(result.clone())),
        InputResult::FileGlobV2(result) => Some(MessageResult::FileGlobV2(result.clone())),
        InputResult::ReadMcpResource(result) => {
            Some(MessageResult::ReadMcpResource(result.clone()))
        }
        InputResult::CallMcpTool(result) => Some(MessageResult::CallMcpTool(result.clone())),
        other => {
            log::warn!("local agent loop: dropping unsupported tool call result {other:?}");
            None
        }
    }
}

const RUN_SHELL_COMMAND: &str = "run_shell_command";
const READ_FILES: &str = "read_files";
const EDIT_FILES: &str = "edit_files";
const GREP: &str = "grep";
const FILE_GLOB: &str = "file_glob";
const READ_MCP_RESOURCE: &str = "read_mcp_resource";
const SUGGEST_PROMPT: &str = "suggest_prompt";

/// Builtin tool definitions plus the per-request MCP projection (mangled
/// OpenAI tool names mapped back to `(server_id, tool_name)`).
#[derive(Clone)]
pub(crate) struct ToolRegistry {
    /// Keyed by the mangled OpenAI-facing tool name.
    mcp_tools: HashMap<String, McpToolBinding>,
    /// Passive suggestion turns are one-shot and read-only: the model gets
    /// only the `suggest_prompt` tool.
    suggestion_only: bool,
}

#[derive(Clone)]
struct McpToolBinding {
    server_id: String,
    tool_name: String,
    description: String,
    parameters: serde_json::Value,
}

impl ToolRegistry {
    fn new(mcp_context: Option<&api::request::McpContext>) -> Self {
        let mut mcp_tools = HashMap::new();
        for server in mcp_context
            .map(|context| context.servers.as_slice())
            .unwrap_or_default()
        {
            for tool in &server.tools {
                let name = mangle_mcp_tool_name(&server.name, &tool.name);
                let parameters = tool
                    .input_schema
                    .as_ref()
                    .map(prost_struct_to_json)
                    .unwrap_or_else(|| json!({ "type": "object" }));
                if mcp_tools
                    .insert(
                        name.clone(),
                        McpToolBinding {
                            server_id: server.id.clone(),
                            tool_name: tool.name.clone(),
                            description: tool.description.clone(),
                            parameters,
                        },
                    )
                    .is_some()
                {
                    log::warn!("local agent loop: duplicate MCP tool name {name}");
                }
            }
        }
        Self {
            mcp_tools,
            suggestion_only: false,
        }
    }

    fn for_passive_suggestions() -> Self {
        Self {
            mcp_tools: HashMap::new(),
            suggestion_only: true,
        }
    }

    pub fn definitions(&self) -> Vec<local_agent::ToolDefinition> {
        if self.suggestion_only {
            return vec![suggest_prompt_definition()];
        }
        let mut definitions = builtin_definitions();
        let mut mcp_names: Vec<&String> = self.mcp_tools.keys().collect();
        mcp_names.sort();
        for name in mcp_names {
            let binding = &self.mcp_tools[name];
            definitions.push(local_agent::ToolDefinition {
                name: name.clone(),
                description: binding.description.clone(),
                parameters: binding.parameters.clone(),
            });
        }
        definitions
    }

    /// Converts a completed model tool call into the proto `ToolCall` message
    /// the existing action executors consume. The OpenAI call id becomes
    /// `tool_call_id` and round-trips back as `ToolCallResult.tool_call_id`.
    pub fn to_proto_tool_call(
        &self,
        call: &local_agent::ToolCallRequest,
    ) -> Result<api::message::ToolCall, AdapterError> {
        use api::message::tool_call::Tool;

        fn parse_args<T: serde::de::DeserializeOwned>(
            call: &local_agent::ToolCallRequest,
        ) -> Result<T, AdapterError> {
            serde_json::from_value(call.arguments.clone()).map_err(|err| {
                AdapterError::BadToolArguments {
                    tool: call.name.clone(),
                    message: err.to_string(),
                }
            })
        }

        let tool = match call.name.as_str() {
            RUN_SHELL_COMMAND => {
                let args: RunShellCommandArgs = parse_args(call)?;
                Tool::RunShellCommand(api::message::tool_call::RunShellCommand {
                    command: args.command,
                    is_read_only: args.is_read_only,
                    is_risky: args.is_risky,
                    // v1 has no long-running command support: every command
                    // runs to completion before its result comes back.
                    wait_until_complete_value: Some(
                        api::message::tool_call::run_shell_command::WaitUntilCompleteValue::WaitUntilComplete(true),
                    ),
                    ..Default::default()
                })
            }
            READ_FILES => {
                let args: ReadFilesArgs = parse_args(call)?;
                Tool::ReadFiles(api::message::tool_call::ReadFiles {
                    files: args
                        .files
                        .into_iter()
                        .map(|file| api::message::tool_call::read_files::File {
                            name: file.path,
                            line_ranges: file
                                .line_ranges
                                .into_iter()
                                .map(|range| api::FileContentLineRange {
                                    start: range.start,
                                    end: range.end,
                                })
                                .collect(),
                        })
                        .collect(),
                })
            }
            EDIT_FILES => {
                let args: EditFilesArgs = parse_args(call)?;
                Tool::ApplyFileDiffs(api::message::tool_call::ApplyFileDiffs {
                    summary: args.summary,
                    diffs: args
                        .diffs
                        .into_iter()
                        .map(|diff| api::message::tool_call::apply_file_diffs::FileDiff {
                            file_path: diff.file_path,
                            search: diff.search,
                            replace: diff.replace,
                        })
                        .collect(),
                    new_files: args
                        .new_files
                        .into_iter()
                        .map(|file| api::message::tool_call::apply_file_diffs::NewFile {
                            file_path: file.file_path,
                            content: file.content,
                            allow_overwrite: file.allow_overwrite,
                        })
                        .collect(),
                    deleted_files: args
                        .deleted_files
                        .into_iter()
                        .map(
                            |file_path| api::message::tool_call::apply_file_diffs::DeleteFile {
                                file_path,
                            },
                        )
                        .collect(),
                    ..Default::default()
                })
            }
            GREP => {
                let args: GrepArgs = parse_args(call)?;
                Tool::Grep(api::message::tool_call::Grep {
                    queries: args.queries,
                    path: args.path,
                })
            }
            FILE_GLOB => {
                let args: FileGlobArgs = parse_args(call)?;
                Tool::FileGlobV2(api::message::tool_call::FileGlobV2 {
                    patterns: args.patterns,
                    search_dir: args.search_dir,
                    ..Default::default()
                })
            }
            READ_MCP_RESOURCE => {
                let args: ReadMcpResourceArgs = parse_args(call)?;
                Tool::ReadMcpResource(api::message::tool_call::ReadMcpResource {
                    uri: args.uri,
                    server_id: args.server_id,
                })
            }
            SUGGEST_PROMPT => {
                let args: SuggestPromptArgs = parse_args(call)?;
                Tool::SuggestPrompt(api::message::tool_call::SuggestPrompt {
                    is_trigger_irrelevant: args.is_trigger_irrelevant,
                    display_mode: Some(
                        api::message::tool_call::suggest_prompt::DisplayMode::PromptChip(
                            api::message::tool_call::suggest_prompt::PromptChip {
                                prompt: args.prompt,
                                label: args.label,
                            },
                        ),
                    ),
                })
            }
            name => {
                let Some(binding) = self.mcp_tools.get(name) else {
                    return Err(AdapterError::UnknownTool(name.to_string()));
                };
                let serde_json::Value::Object(arguments) = &call.arguments else {
                    return Err(AdapterError::BadToolArguments {
                        tool: name.to_string(),
                        message: "MCP tool arguments must be a JSON object".to_string(),
                    });
                };
                Tool::CallMcpTool(api::message::tool_call::CallMcpTool {
                    name: binding.tool_name.clone(),
                    args: Some(json_map_to_prost_struct(arguments)),
                    server_id: binding.server_id.clone(),
                })
            }
        };
        Ok(api::message::ToolCall {
            tool_call_id: call.id.clone(),
            tool: Some(tool),
        })
    }

    /// The reverse of [`Self::to_proto_tool_call`], used to rebuild the
    /// transcript from historic `ToolCall` messages. Returns `None` for
    /// tools the local loop cannot express (server-only tools from earlier
    /// server-routed turns of the same conversation).
    fn to_request_tool_call(
        &self,
        tool_call: &api::message::ToolCall,
    ) -> Option<local_agent::ToolCallRequest> {
        use api::message::tool_call::Tool;
        let (name, arguments) = match tool_call.tool.as_ref()? {
            Tool::RunShellCommand(shell) => (
                RUN_SHELL_COMMAND.to_string(),
                json!({
                    "command": shell.command,
                    "is_read_only": shell.is_read_only,
                    "is_risky": shell.is_risky,
                }),
            ),
            Tool::ReadFiles(read) => (
                READ_FILES.to_string(),
                json!({
                    "files": read.files.iter().map(|file| {
                        json!({
                            "path": file.name,
                            "line_ranges": file.line_ranges.iter().map(|range| {
                                json!({ "start": range.start, "end": range.end })
                            }).collect::<Vec<_>>(),
                        })
                    }).collect::<Vec<_>>(),
                }),
            ),
            Tool::ApplyFileDiffs(apply) => (
                EDIT_FILES.to_string(),
                json!({
                    "summary": apply.summary,
                    "diffs": apply.diffs.iter().map(|diff| {
                        json!({
                            "file_path": diff.file_path,
                            "search": diff.search,
                            "replace": diff.replace,
                        })
                    }).collect::<Vec<_>>(),
                    "new_files": apply.new_files.iter().map(|file| {
                        json!({ "file_path": file.file_path, "content": file.content })
                    }).collect::<Vec<_>>(),
                    "deleted_files": apply.deleted_files.iter()
                        .map(|file| file.file_path.clone())
                        .collect::<Vec<_>>(),
                }),
            ),
            Tool::Grep(grep) => (
                GREP.to_string(),
                json!({ "queries": grep.queries, "path": grep.path }),
            ),
            #[allow(deprecated)]
            Tool::FileGlob(glob) => (
                FILE_GLOB.to_string(),
                json!({ "patterns": glob.patterns, "search_dir": glob.path }),
            ),
            Tool::FileGlobV2(glob) => (
                FILE_GLOB.to_string(),
                json!({ "patterns": glob.patterns, "search_dir": glob.search_dir }),
            ),
            Tool::ReadMcpResource(read) => (
                READ_MCP_RESOURCE.to_string(),
                json!({ "uri": read.uri, "server_id": read.server_id }),
            ),
            Tool::SuggestPrompt(suggest) => {
                let (prompt, label) = match suggest.display_mode.as_ref() {
                    Some(api::message::tool_call::suggest_prompt::DisplayMode::PromptChip(
                        chip,
                    )) => (chip.prompt.clone(), chip.label.clone()),
                    _ => return None,
                };
                (
                    SUGGEST_PROMPT.to_string(),
                    json!({
                        "prompt": prompt,
                        "label": label,
                        "is_trigger_irrelevant": suggest.is_trigger_irrelevant,
                    }),
                )
            }
            Tool::CallMcpTool(call) => {
                let name = self
                    .mcp_tools
                    .iter()
                    .find(|(_, binding)| {
                        binding.server_id == call.server_id && binding.tool_name == call.name
                    })
                    .map(|(name, _)| name.clone())
                    // The server may no longer be configured; reconstruct a
                    // plausible name so the transcript stays coherent.
                    .unwrap_or_else(|| mangle_mcp_tool_name(&call.server_id, &call.name));
                let arguments = call
                    .args
                    .as_ref()
                    .map(prost_struct_to_json)
                    .unwrap_or_else(|| json!({}));
                (name, arguments)
            }
            _ => return None,
        };
        Some(local_agent::ToolCallRequest {
            id: tool_call.tool_call_id.clone(),
            name,
            arguments,
        })
    }
}

#[derive(Deserialize)]
struct RunShellCommandArgs {
    command: String,
    #[serde(default)]
    is_read_only: bool,
    #[serde(default)]
    is_risky: bool,
}

#[derive(Deserialize)]
struct ReadFilesArgs {
    files: Vec<ReadFileArg>,
}

#[derive(Deserialize)]
struct ReadFileArg {
    path: String,
    #[serde(default)]
    line_ranges: Vec<LineRangeArg>,
}

#[derive(Deserialize)]
struct LineRangeArg {
    start: u32,
    end: u32,
}

#[derive(Deserialize)]
struct EditFilesArgs {
    #[serde(default)]
    summary: String,
    #[serde(default)]
    diffs: Vec<FileDiffArg>,
    #[serde(default)]
    new_files: Vec<NewFileArg>,
    #[serde(default)]
    deleted_files: Vec<String>,
}

#[derive(Deserialize)]
struct FileDiffArg {
    file_path: String,
    search: String,
    replace: String,
}

#[derive(Deserialize)]
struct NewFileArg {
    file_path: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    allow_overwrite: bool,
}

#[derive(Deserialize)]
struct GrepArgs {
    queries: Vec<String>,
    #[serde(default)]
    path: String,
}

#[derive(Deserialize)]
struct FileGlobArgs {
    patterns: Vec<String>,
    #[serde(default)]
    search_dir: String,
}

#[derive(Deserialize)]
struct SuggestPromptArgs {
    prompt: String,
    #[serde(default)]
    label: String,
    #[serde(default)]
    is_trigger_irrelevant: bool,
}

#[derive(Deserialize)]
struct ReadMcpResourceArgs {
    uri: String,
    #[serde(default)]
    server_id: String,
}

fn suggest_prompt_definition() -> local_agent::ToolDefinition {
    local_agent::ToolDefinition {
        name: SUGGEST_PROMPT.to_string(),
        description: "Proposes one follow-up prompt the user could send to the AI agent. \
                      Call at most once, and only when a genuinely useful, concrete \
                      follow-up exists for the provided context."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The full prompt to send to the agent if the user accepts the suggestion.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short display label; omit to display the prompt itself.",
                },
                "is_trigger_irrelevant": {
                    "type": "boolean",
                    "description": "True if the suggestion is useful but unrelated to the event that triggered it.",
                },
            },
            "required": ["prompt"],
        }),
    }
}

fn builtin_definitions() -> Vec<local_agent::ToolDefinition> {
    vec![
        local_agent::ToolDefinition {
            name: RUN_SHELL_COMMAND.to_string(),
            description: "Executes a shell command in the user's terminal and returns its \
                          output and exit code. The command runs to completion before the \
                          result is returned; there is no interactive input."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command to execute.",
                    },
                    "is_read_only": {
                        "type": "boolean",
                        "description": "True if the command only reads state and has no side effects.",
                    },
                    "is_risky": {
                        "type": "boolean",
                        "description": "True if the command could have destructive or hard-to-reverse effects.",
                    },
                },
                "required": ["command"],
            }),
        },
        local_agent::ToolDefinition {
            name: READ_FILES.to_string(),
            description: "Reads the contents of one or more files, optionally restricted to \
                          specific line ranges."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": { "type": "string", "description": "Path to the file." },
                                "line_ranges": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "start": { "type": "integer", "description": "First line, 1-based, inclusive." },
                                            "end": { "type": "integer", "description": "Last line, 1-based, inclusive." },
                                        },
                                        "required": ["start", "end"],
                                    },
                                },
                            },
                            "required": ["path"],
                        },
                    },
                },
                "required": ["files"],
            }),
        },
        local_agent::ToolDefinition {
            name: EDIT_FILES.to_string(),
            description: "Edits files via exact search/replace blocks, creates new files, and \
                          deletes files. Each diff's `search` must match the file contents \
                          exactly (including whitespace) and is replaced by `replace`."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A short human-readable summary of the change.",
                    },
                    "diffs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": { "type": "string" },
                                "search": { "type": "string", "description": "Exact text to find in the file." },
                                "replace": { "type": "string", "description": "Text to replace it with." },
                            },
                            "required": ["file_path", "search", "replace"],
                        },
                    },
                    "new_files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": { "type": "string" },
                                "content": { "type": "string" },
                                "allow_overwrite": { "type": "boolean" },
                            },
                            "required": ["file_path", "content"],
                        },
                    },
                    "deleted_files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Paths of files to delete.",
                    },
                },
                "required": ["summary"],
            }),
        },
        local_agent::ToolDefinition {
            name: GREP.to_string(),
            description: "Searches file contents for the given regular expressions and returns \
                          matching files and line numbers."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Regular expressions to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in. Defaults to the working directory.",
                    },
                },
                "required": ["queries"],
            }),
        },
        local_agent::ToolDefinition {
            name: FILE_GLOB.to_string(),
            description: "Finds files whose names match the given glob patterns (supports ?, * \
                          and [])."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "patterns": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Glob patterns to match file names against.",
                    },
                    "search_dir": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to the working directory.",
                    },
                },
                "required": ["patterns"],
            }),
        },
        local_agent::ToolDefinition {
            name: READ_MCP_RESOURCE.to_string(),
            description: "Reads an MCP resource identified by its URI.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "uri": { "type": "string", "description": "URI of the resource to read." },
                    "server_id": {
                        "type": "string",
                        "description": "Identifier of the MCP server providing the resource.",
                    },
                },
                "required": ["uri"],
            }),
        },
    ]
}

/// Mangles an MCP server/tool pair into a name OpenAI-compatible servers
/// accept (`[a-zA-Z0-9_-]`, at most 64 characters).
fn mangle_mcp_tool_name(server: &str, tool: &str) -> String {
    let sanitize = |value: &str| {
        value
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>()
    };
    let mut name = format!("mcp__{}__{}", sanitize(server), sanitize(tool));
    name.truncate(64);
    name
}

/// Renders a proto tool call result into the plain-text content of a
/// `role:"tool"` transcript message.
pub(crate) fn render_tool_result_text(result: &api::message::ToolCallResult) -> String {
    use api::message::tool_call_result::Result;
    let Some(result) = result.result.as_ref() else {
        return "(no result)".to_string();
    };
    match result {
        Result::RunShellCommand(shell) => render_shell_result(shell),
        Result::ReadFiles(read) => match read.result.as_ref() {
            Some(api::read_files_result::Result::TextFilesSuccess(success)) => success
                .files
                .iter()
                .map(render_file_content)
                .collect::<Vec<_>>()
                .join("\n\n"),
            Some(api::read_files_result::Result::AnyFilesSuccess(success)) => success
                .files
                .iter()
                .map(|file| match file.content.as_ref() {
                    Some(api::any_file_content::Content::TextContent(content)) => {
                        render_file_content(content)
                    }
                    Some(api::any_file_content::Content::BinaryContent(content)) => {
                        format!(
                            "=== {} ===\n(binary file, {} bytes)",
                            content.file_path,
                            content.data.len()
                        )
                    }
                    None => "(empty file entry)".to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
            Some(api::read_files_result::Result::Error(error)) => {
                format!("Error: {}", error.message)
            }
            None => "(no result)".to_string(),
        },
        Result::ApplyFileDiffs(apply) => match apply.result.as_ref() {
            Some(api::apply_file_diffs_result::Result::Success(success)) => {
                let mut lines = vec!["Edits applied successfully.".to_string()];
                for updated in &success.updated_files_v2 {
                    if let Some(file) = &updated.file {
                        lines.push(format!("Updated: {}", file.file_path));
                    }
                }
                for deleted in &success.deleted_files {
                    lines.push(format!("Deleted: {}", deleted.file_path));
                }
                lines.join("\n")
            }
            Some(api::apply_file_diffs_result::Result::Error(error)) => {
                format!("Error: {}", error.message)
            }
            None => "(no result)".to_string(),
        },
        Result::Grep(grep) => match grep.result.as_ref() {
            Some(api::grep_result::Result::Success(success)) => {
                if success.matched_files.is_empty() {
                    "No matches found.".to_string()
                } else {
                    success
                        .matched_files
                        .iter()
                        .map(|file| {
                            let lines = file
                                .matched_lines
                                .iter()
                                .map(|line| line.line_number.to_string())
                                .collect::<Vec<_>>()
                                .join(", ");
                            format!("{}: lines {}", file.file_path, lines)
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            }
            Some(api::grep_result::Result::Error(error)) => format!("Error: {}", error.message),
            None => "(no result)".to_string(),
        },
        #[allow(deprecated)]
        Result::FileGlob(glob) => match glob.result.as_ref() {
            Some(api::file_glob_result::Result::Success(success)) => {
                if success.matched_files.is_empty() {
                    "No matches found.".to_string()
                } else {
                    success.matched_files.clone()
                }
            }
            Some(api::file_glob_result::Result::Error(error)) => {
                format!("Error: {}", error.message)
            }
            None => "(no result)".to_string(),
        },
        Result::FileGlobV2(glob) => match glob.result.as_ref() {
            Some(api::file_glob_v2_result::Result::Success(success)) => {
                let mut text = success
                    .matched_files
                    .iter()
                    .map(|file| file.file_path.clone())
                    .collect::<Vec<_>>()
                    .join("\n");
                if text.is_empty() {
                    text = "No matches found.".to_string();
                }
                if !success.warnings.is_empty() {
                    text.push_str(&format!("\n\nWarnings:\n{}", success.warnings));
                }
                text
            }
            Some(api::file_glob_v2_result::Result::Error(error)) => {
                format!("Error: {}", error.message)
            }
            None => "(no result)".to_string(),
        },
        Result::ReadMcpResource(read) => match read.result.as_ref() {
            Some(api::read_mcp_resource_result::Result::Success(success)) => success
                .contents
                .iter()
                .map(|content| match content.content_type.as_ref() {
                    Some(api::mcp_resource_content::ContentType::Text(text)) => {
                        format!("=== {} ===\n{}", content.uri, text.content)
                    }
                    Some(api::mcp_resource_content::ContentType::Binary(binary)) => {
                        format!(
                            "=== {} ===\n(binary content, {} bytes, {})",
                            content.uri,
                            binary.data.len(),
                            binary.mime_type
                        )
                    }
                    None => format!("=== {} ===\n(empty)", content.uri),
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
            Some(api::read_mcp_resource_result::Result::Error(error)) => {
                format!("Error: {}", error.message)
            }
            None => "(no result)".to_string(),
        },
        Result::CallMcpTool(call) => match call.result.as_ref() {
            Some(api::call_mcp_tool_result::Result::Success(success)) => success
                .results
                .iter()
                .map(|entry| match entry.result.as_ref() {
                    Some(api::call_mcp_tool_result::success::result::Result::Text(text)) => {
                        text.text.clone()
                    }
                    Some(other) => format!("(non-text MCP content: {other:?})"),
                    None => "(empty MCP content)".to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n"),
            Some(api::call_mcp_tool_result::Result::Error(error)) => {
                format!("Error: {}", error.message)
            }
            None => "(no result)".to_string(),
        },
        Result::Cancel(()) => "The user canceled this tool call.".to_string(),
        other => {
            // The local loop never issues the remaining tools; render a debug
            // dump so mixed (server + local) conversations stay usable.
            format!("{other:?}")
        }
    }
}

fn render_shell_result(shell: &api::RunShellCommandResult) -> String {
    match shell.result.as_ref() {
        Some(api::run_shell_command_result::Result::CommandFinished(finished)) => {
            format!(
                "Exit code: {}\n\nOutput:\n{}",
                finished.exit_code, finished.output
            )
        }
        Some(api::run_shell_command_result::Result::PermissionDenied(_)) => {
            "The user denied permission to run this command.".to_string()
        }
        Some(api::run_shell_command_result::Result::LongRunningCommandSnapshot(snapshot)) => {
            format!("The command is still running. Latest output snapshot:\n{snapshot:?}")
        }
        #[allow(deprecated)]
        None => format!(
            "Exit code: {}\n\nOutput:\n{}",
            shell.exit_code, shell.output
        ),
    }
}

fn render_file_content(content: &api::FileContent) -> String {
    match &content.line_range {
        Some(range) => format!(
            "=== {} (lines {}-{}) ===\n{}",
            content.file_path, range.start, range.end, content.content
        ),
        None => format!("=== {} ===\n{}", content.file_path, content.content),
    }
}

fn json_map_to_prost_struct(
    map: &serde_json::Map<String, serde_json::Value>,
) -> prost_types::Struct {
    prost_types::Struct {
        fields: map
            .iter()
            .map(|(key, value)| (key.clone(), json_value_to_prost(value)))
            .collect(),
    }
}

fn json_value_to_prost(value: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(value) => Kind::BoolValue(*value),
        serde_json::Value::Number(number) => Kind::NumberValue(number.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(value) => Kind::StringValue(value.clone()),
        serde_json::Value::Array(values) => Kind::ListValue(prost_types::ListValue {
            values: values.iter().map(json_value_to_prost).collect(),
        }),
        serde_json::Value::Object(map) => Kind::StructValue(json_map_to_prost_struct(map)),
    };
    prost_types::Value { kind: Some(kind) }
}

fn prost_struct_to_json(value: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), prost_value_to_json(value)))
            .collect(),
    )
}

fn prost_value_to_json(value: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match value.kind.as_ref() {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(value)) => serde_json::Value::Bool(*value),
        Some(Kind::NumberValue(number)) => serde_json::Number::from_f64(*number)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Some(Kind::StringValue(value)) => serde_json::Value::String(value.clone()),
        Some(Kind::ListValue(values)) => {
            serde_json::Value::Array(values.values.iter().map(prost_value_to_json).collect())
        }
        Some(Kind::StructValue(map)) => prost_struct_to_json(map),
    }
}

/// Field-mask path for appending streamed text to an `AgentOutput` message.
/// Paths are resolved against `api::MESSAGE_DESCRIPTOR`, where oneof members
/// are direct fields of `Message` (the oneof name itself is not addressable).
const AGENT_OUTPUT_TEXT_MASK: &str = "agent_output.text";

/// Stateful mapper from engine events to the proto `ResponseEvent` sequence.
/// Pure (no I/O), so golden tests can drive it directly.
pub(crate) struct EventMapper {
    conversation_id: String,
    request_id: String,
    root_task_id: String,
    is_first_turn: bool,
    config_key: String,
    echo_messages: Vec<api::Message>,
    registry: ToolRegistry,
    /// Id of the `AgentOutput` message currently receiving text appends.
    streaming_message_id: Option<String>,
    input_tokens: u64,
    output_tokens: u64,
}

impl EventMapper {
    pub fn new(plan: &TurnPlan) -> Self {
        EventMapper {
            conversation_id: plan.conversation_id.clone(),
            request_id: plan.request_id.clone(),
            root_task_id: plan.root_task_id.clone(),
            is_first_turn: plan.is_first_turn,
            config_key: plan.config_key.clone(),
            echo_messages: plan.input_echo_messages.clone(),
            registry: plan.tools.clone(),
            streaming_message_id: None,
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    /// Events emitted before the first engine event: `Init`,
    /// `BeginTransaction`, `CreateTask` (first turn only), input echoes and
    /// `ModelUsed`.
    pub fn initial_events(&mut self) -> Vec<Event> {
        use api::client_action::Action;
        let mut events = vec![
            Ok(api::ResponseEvent {
                r#type: Some(api::response_event::Type::Init(
                    api::response_event::StreamInit {
                        conversation_id: self.conversation_id.clone(),
                        request_id: self.request_id.clone(),
                        run_id: String::new(),
                    },
                )),
            }),
            client_actions(vec![Action::BeginTransaction(
                api::client_action::BeginTransaction {},
            )]),
        ];
        if self.is_first_turn {
            // The client upgrades its optimistically-created task to this
            // one; follow-up turns must not re-create it.
            events.push(client_actions(vec![Action::CreateTask(
                api::client_action::CreateTask {
                    task: Some(api::Task {
                        id: self.root_task_id.clone(),
                        ..Default::default()
                    }),
                },
            )]));
        }
        let mut messages = std::mem::take(&mut self.echo_messages);
        messages.push(new_task_message(
            &self.root_task_id,
            api::message::Message::ModelUsed(api::message::ModelUsed {
                model_id: self.config_key.clone(),
                ..Default::default()
            }),
        ));
        events.push(client_actions(vec![Action::AddMessagesToTask(
            api::client_action::AddMessagesToTask {
                task_id: self.root_task_id.clone(),
                messages,
            },
        )]));
        events
    }

    /// Maps one engine event; text deltas become `AddMessagesToTask` then
    /// `AppendToMessageContent`, completed tool calls become
    /// `AddMessagesToTask`, `Done` becomes `CommitTransaction` + `Finished`,
    /// errors become `RollbackTransaction` + terminal events.
    pub fn on_event(
        &mut self,
        event: Result<local_agent::AgentEvent, local_agent::LocalAgentError>,
    ) -> Vec<Event> {
        use api::client_action::Action;
        use local_agent::AgentEvent;

        match event {
            Ok(AgentEvent::TextDelta(text)) => match self.streaming_message_id.clone() {
                Some(message_id) => vec![client_actions(vec![Action::AppendToMessageContent(
                    api::client_action::AppendToMessageContent {
                        task_id: self.root_task_id.clone(),
                        message: Some(api::Message {
                            id: message_id,
                            ..new_task_message(
                                &self.root_task_id,
                                api::message::Message::AgentOutput(api::message::AgentOutput {
                                    text,
                                }),
                            )
                        }),
                        mask: Some(prost_types::FieldMask {
                            paths: vec![AGENT_OUTPUT_TEXT_MASK.to_string()],
                        }),
                    },
                )])],
                None => {
                    let message = new_task_message(
                        &self.root_task_id,
                        api::message::Message::AgentOutput(api::message::AgentOutput { text }),
                    );
                    self.streaming_message_id = Some(message.id.clone());
                    vec![self.add_messages(vec![message])]
                }
            },
            // Reasoning is not surfaced in v1; the engine already keeps it
            // out of the final text.
            Ok(AgentEvent::ReasoningDelta(_)) => vec![],
            Ok(AgentEvent::ToolCall(call)) => {
                // Any text after this belongs to a new output message.
                self.streaming_message_id = None;
                match self.registry.to_proto_tool_call(&call) {
                    Ok(tool_call) => vec![self.add_messages(vec![new_task_message(
                        &self.root_task_id,
                        api::message::Message::ToolCall(tool_call),
                    )])],
                    Err(err) => self.finish_with_rollback(
                        api::response_event::stream_finished::Reason::InternalError(
                            api::response_event::stream_finished::InternalError {
                                message: err.to_string(),
                            },
                        ),
                    ),
                }
            }
            Ok(AgentEvent::Usage {
                input_tokens,
                output_tokens,
            }) => {
                self.input_tokens += input_tokens;
                self.output_tokens += output_tokens;
                vec![]
            }
            Ok(AgentEvent::Done) => {
                vec![
                    client_actions(vec![Action::CommitTransaction(
                        api::client_action::CommitTransaction {},
                    )]),
                    self.finished(api::response_event::stream_finished::Reason::Done(
                        api::response_event::stream_finished::Done {},
                    )),
                ]
            }
            Err(local_agent::LocalAgentError::InvalidApiKey { model_slug }) => self
                .finish_with_rollback(api::response_event::stream_finished::Reason::InvalidApiKey(
                    api::response_event::stream_finished::InvalidApiKey {
                        provider: api::LlmProvider::Unknown as i32,
                        model_name: model_slug,
                    },
                )),
            Err(local_agent::LocalAgentError::ContextWindowExceeded { .. }) => self
                .finish_with_rollback(
                    api::response_event::stream_finished::Reason::ContextWindowExceeded(
                        api::response_event::stream_finished::ContextWindowExceeded {},
                    ),
                ),
            Err(err) => {
                // Transport-level failures surface as stream errors so the
                // existing retry machinery in `ResponseStream` applies.
                vec![
                    client_actions(vec![Action::RollbackTransaction(
                        api::client_action::RollbackTransaction {},
                    )]),
                    Err(Arc::new(map_engine_error(err))),
                ]
            }
        }
    }

    fn add_messages(&self, messages: Vec<api::Message>) -> Event {
        client_actions(vec![api::client_action::Action::AddMessagesToTask(
            api::client_action::AddMessagesToTask {
                task_id: self.root_task_id.clone(),
                messages,
            },
        )])
    }

    fn finish_with_rollback(
        &mut self,
        reason: api::response_event::stream_finished::Reason,
    ) -> Vec<Event> {
        vec![
            client_actions(vec![api::client_action::Action::RollbackTransaction(
                api::client_action::RollbackTransaction {},
            )]),
            self.finished(reason),
        ]
    }

    fn finished(&mut self, reason: api::response_event::stream_finished::Reason) -> Event {
        let token_usage = if self.input_tokens > 0 || self.output_tokens > 0 {
            vec![api::response_event::stream_finished::TokenUsage {
                model_id: self.config_key.clone(),
                total_input: self.input_tokens as u32,
                output: self.output_tokens as u32,
                ..Default::default()
            }]
        } else {
            vec![]
        };
        Ok(api::ResponseEvent {
            r#type: Some(api::response_event::Type::Finished(
                api::response_event::StreamFinished {
                    token_usage,
                    reason: Some(reason),
                    ..Default::default()
                },
            )),
        })
    }
}

fn client_actions(actions: Vec<api::client_action::Action>) -> Event {
    Ok(api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: actions
                    .into_iter()
                    .map(|action| api::ClientAction {
                        action: Some(action),
                    })
                    .collect(),
            },
        )),
    })
}

fn map_engine_error(err: local_agent::LocalAgentError) -> AIApiError {
    match err {
        local_agent::LocalAgentError::ErrorStatus { status, body } => AIApiError::ErrorStatus(
            http::StatusCode::from_u16(status).unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR),
            body,
        ),
        other => AIApiError::Stream {
            stream_type: "local agent loop",
            source: anyhow!(other),
        },
    }
}

/// Serves one agent request via the local loop, producing a stream that is a
/// drop-in replacement for `ServerApi::generate_multi_agent_output`'s.
pub(crate) fn generate_local_agent_output(
    request: api::Request,
    route: LocalRoute,
) -> ResponseStream {
    let plan = match plan_turn(&request, &route) {
        Ok(plan) => plan,
        Err(err) => {
            return Box::pin(futures::stream::once(async move {
                Err(Arc::new(AIApiError::Other(anyhow!(err))))
            }));
        }
    };
    let mut mapper = EventMapper::new(&plan);
    let initial_events = mapper.initial_events();
    let engine_events =
        local_agent::run_turn(plan.transcript, plan.tools.definitions(), route.endpoint);
    let mapped_events =
        engine_events.flat_map(move |event| futures::stream::iter(mapper.on_event(event)));
    Box::pin(futures::stream::iter(initial_events).chain(mapped_events))
}

#[cfg(test)]
#[path = "local_adapter_tests.rs"]
mod tests;
