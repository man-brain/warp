//! Tests for the proto ↔ neutral-model adapter: transcript rebuilding from
//! the task tree, tool schema/argument mapping, tool result rendering, and
//! golden `ResponseEvent` sequences from [`EventMapper`].

use local_agent::{AgentEvent, ChatMessage, LocalAgentError, LocalEndpointConfig, ToolCallRequest};
use serde_json::json;
use warp_multi_agent_api as api;

use super::*;

const ENDPOINT_URL: &str = "https://llm.example.com/v1";
const CONFIG_KEY: &str = "cfg-123";
const MODEL_SLUG: &str = "llama-3.3-70b";

fn route() -> LocalRoute {
    LocalRoute {
        endpoint: LocalEndpointConfig {
            base_url: ENDPOINT_URL.to_string(),
            api_key: "sk-test".to_string(),
            model_slug: MODEL_SLUG.to_string(),
        },
        config_key: CONFIG_KEY.to_string(),
    }
}

fn message(id: &str, content: api::message::Message) -> api::Message {
    api::Message {
        id: id.to_string(),
        task_id: "task-1".to_string(),
        request_id: String::new(),
        timestamp: None,
        server_message_data: String::new(),
        citations: vec![],
        fetched_memories: vec![],
        message: Some(content),
    }
}

fn user_query_message(id: &str, query: &str) -> api::Message {
    message(
        id,
        api::message::Message::UserQuery(api::message::UserQuery {
            query: query.to_string(),
            ..Default::default()
        }),
    )
}

fn agent_output_message(id: &str, text: &str) -> api::Message {
    message(
        id,
        api::message::Message::AgentOutput(api::message::AgentOutput {
            text: text.to_string(),
        }),
    )
}

fn shell_tool_call_message(id: &str, call_id: &str, command: &str) -> api::Message {
    message(
        id,
        api::message::Message::ToolCall(api::message::ToolCall {
            tool_call_id: call_id.to_string(),
            tool: Some(api::message::tool_call::Tool::RunShellCommand(
                api::message::tool_call::RunShellCommand {
                    command: command.to_string(),
                    ..Default::default()
                },
            )),
        }),
    )
}

fn shell_finished_result(output: &str, exit_code: i32) -> api::RunShellCommandResult {
    api::RunShellCommandResult {
        result: Some(api::run_shell_command_result::Result::CommandFinished(
            api::ShellCommandFinished {
                output: output.to_string(),
                exit_code,
                ..Default::default()
            },
        )),
        ..Default::default()
    }
}

fn shell_result_message(id: &str, call_id: &str, output: &str, exit_code: i32) -> api::Message {
    message(
        id,
        api::message::Message::ToolCallResult(api::message::ToolCallResult {
            tool_call_id: call_id.to_string(),
            context: None,
            result: Some(api::message::tool_call_result::Result::RunShellCommand(
                shell_finished_result(output, exit_code),
            )),
        }),
    )
}

fn root_task(messages: Vec<api::Message>) -> api::Task {
    api::Task {
        id: "task-1".to_string(),
        description: String::new(),
        dependencies: None,
        messages,
        summary: String::new(),
        server_data: String::new(),
    }
}

fn user_query_input(query: &str) -> api::request::input::user_inputs::user_input::Input {
    api::request::input::user_inputs::user_input::Input::UserQuery(api::request::input::UserQuery {
        query: query.to_string(),
        ..Default::default()
    })
}

fn shell_result_input(
    call_id: &str,
    output: &str,
    exit_code: i32,
) -> api::request::input::user_inputs::user_input::Input {
    api::request::input::user_inputs::user_input::Input::ToolCallResult(
        api::request::input::ToolCallResult {
            tool_call_id: call_id.to_string(),
            result: Some(
                api::request::input::tool_call_result::Result::RunShellCommand(
                    shell_finished_result(output, exit_code),
                ),
            ),
        },
    )
}

fn user_inputs(
    inputs: Vec<api::request::input::user_inputs::user_input::Input>,
) -> api::request::input::Type {
    api::request::input::Type::UserInputs(api::request::input::UserInputs {
        inputs: inputs
            .into_iter()
            .map(|input| api::request::input::user_inputs::UserInput { input: Some(input) })
            .collect(),
    })
}

fn request(
    tasks: Vec<api::Task>,
    input_type: api::request::input::Type,
    conversation_id: &str,
) -> api::Request {
    api::Request {
        task_context: Some(api::request::TaskContext { tasks }),
        input: Some(api::request::Input {
            context: None,
            r#type: Some(input_type),
        }),
        settings: Some(api::request::Settings {
            model_config: Some(api::request::settings::ModelConfig {
                base: CONFIG_KEY.to_string(),
                ..Default::default()
            }),
            ..Default::default()
        }),
        metadata: Some(api::request::Metadata {
            conversation_id: conversation_id.to_string(),
            ..Default::default()
        }),
        existing_suggestions: None,
        mcp_context: None,
    }
}

fn json_schema_struct() -> prost_types::Struct {
    prost_types::Struct {
        fields: [(
            "type".to_string(),
            prost_types::Value {
                kind: Some(prost_types::value::Kind::StringValue("object".to_string())),
            },
        )]
        .into_iter()
        .collect(),
    }
}

fn mcp_context() -> api::request::McpContext {
    #[allow(deprecated)]
    api::request::McpContext {
        resources: vec![],
        tools: vec![],
        servers: vec![api::request::mcp_context::McpServer {
            name: "Linear".to_string(),
            description: "Issue tracker".to_string(),
            id: "srv-1".to_string(),
            resources: vec![],
            tools: vec![api::request::mcp_context::McpTool {
                name: "create_issue".to_string(),
                description: "Creates an issue".to_string(),
                input_schema: Some(json_schema_struct()),
            }],
        }],
    }
}

/// Flattens a `ResponseEvent` sequence into human-readable kind labels so
/// golden tests can assert ordering without exhaustively matching payloads.
fn event_kinds(events: &[Event]) -> Vec<String> {
    let mut kinds = Vec::new();
    for event in events {
        match event {
            Err(_) => kinds.push("error".to_string()),
            Ok(response_event) => match response_event.r#type.as_ref().expect("event type") {
                api::response_event::Type::Init(_) => kinds.push("init".to_string()),
                api::response_event::Type::Finished(_) => kinds.push("finished".to_string()),
                api::response_event::Type::ClientActions(actions) => {
                    for action in &actions.actions {
                        let kind = match action.action.as_ref().expect("action type") {
                            api::client_action::Action::BeginTransaction(_) => "begin_transaction",
                            api::client_action::Action::CommitTransaction(_) => {
                                "commit_transaction"
                            }
                            api::client_action::Action::RollbackTransaction(_) => {
                                "rollback_transaction"
                            }
                            api::client_action::Action::CreateTask(_) => "create_task",
                            api::client_action::Action::AddMessagesToTask(_) => "add_messages",
                            api::client_action::Action::AppendToMessageContent(_) => "append",
                            _ => "other_action",
                        };
                        kinds.push(kind.to_string());
                    }
                }
            },
        }
    }
    kinds
}

fn added_messages(events: &[Event]) -> Vec<api::Message> {
    events
        .iter()
        .filter_map(|event| event.as_ref().ok())
        .filter_map(|event| match event.r#type.as_ref() {
            Some(api::response_event::Type::ClientActions(actions)) => Some(&actions.actions),
            _ => None,
        })
        .flatten()
        .filter_map(|action| match action.action.as_ref() {
            Some(api::client_action::Action::AddMessagesToTask(add)) => Some(add.messages.clone()),
            _ => None,
        })
        .flatten()
        .collect()
}

mod transcript {
    use super::*;

    #[test]
    fn rebuilds_history_from_task_tree_and_appends_new_query() {
        let req = request(
            vec![root_task(vec![
                user_query_message("m1", "first question"),
                agent_output_message("m2", "first answer"),
            ])],
            user_inputs(vec![user_query_input("second question")]),
            "conv-1",
        );

        let plan = plan_turn(&req, &route()).expect("plan");

        assert!(
            matches!(&plan.transcript[0], ChatMessage::System(prompt) if !prompt.is_empty()),
            "transcript must start with a non-empty system prompt"
        );
        assert_eq!(
            &plan.transcript[1..],
            &[
                ChatMessage::User("first question".to_string()),
                ChatMessage::Assistant {
                    text: "first answer".to_string(),
                    tool_calls: vec![],
                },
                ChatMessage::User("second question".to_string()),
            ]
        );
        assert!(!plan.is_first_turn);
        assert_eq!(plan.conversation_id, "conv-1");
        assert_eq!(plan.root_task_id, "task-1");
        assert_eq!(plan.config_key, CONFIG_KEY);
    }

    #[test]
    fn maps_historic_tool_call_and_result_to_assistant_and_tool_messages() {
        let req = request(
            vec![root_task(vec![
                user_query_message("m1", "list files"),
                shell_tool_call_message("m2", "call_1", "ls"),
                shell_result_message("m3", "call_1", "file.txt\n", 0),
            ])],
            user_inputs(vec![user_query_input("now what?")]),
            "conv-1",
        );

        let plan = plan_turn(&req, &route()).expect("plan");

        let assistant = plan
            .transcript
            .iter()
            .find_map(|m| match m {
                ChatMessage::Assistant { tool_calls, .. } if !tool_calls.is_empty() => {
                    Some(tool_calls[0].clone())
                }
                _ => None,
            })
            .expect("assistant message with the historic tool call");
        assert_eq!(assistant.id, "call_1");
        assert_eq!(assistant.name, "run_shell_command");
        assert_eq!(assistant.arguments["command"], json!("ls"));

        let tool_message = plan
            .transcript
            .iter()
            .find_map(|m| match m {
                ChatMessage::Tool {
                    tool_call_id,
                    content,
                } if tool_call_id == "call_1" => Some(content.clone()),
                _ => None,
            })
            .expect("tool result message for call_1");
        assert!(
            tool_message.contains("file.txt"),
            "missing output: {tool_message}"
        );
    }

    #[test]
    fn fresh_tool_call_results_become_tool_messages_and_are_echoed() {
        let req = request(
            vec![root_task(vec![
                user_query_message("m1", "list files"),
                shell_tool_call_message("m2", "call_1", "ls"),
            ])],
            user_inputs(vec![shell_result_input("call_1", "file.txt\n", 0)]),
            "conv-1",
        );

        let plan = plan_turn(&req, &route()).expect("plan");

        assert!(
            matches!(
                plan.transcript.last(),
                Some(ChatMessage::Tool { tool_call_id, .. }) if tool_call_id == "call_1"
            ),
            "fresh tool result must be the last transcript message, got {:?}",
            plan.transcript.last()
        );
        assert!(
            plan.input_echo_messages.iter().any(|m| matches!(
                &m.message,
                Some(api::message::Message::ToolCallResult(r)) if r.tool_call_id == "call_1"
            )),
            "tool result must be echoed into the task tree for the next turn"
        );
    }

    #[test]
    fn first_turn_mints_conversation_and_task_ids() {
        let req = request(vec![], user_inputs(vec![user_query_input("hi")]), "");

        let plan = plan_turn(&req, &route()).expect("plan");

        assert!(plan.is_first_turn);
        assert!(!plan.root_task_id.is_empty());
        // The client parses conversation tokens as UUIDs in places, so the
        // minted id must be one.
        assert!(
            uuid::Uuid::parse_str(&plan.conversation_id).is_ok(),
            "conversation id must be a UUID, got {:?}",
            plan.conversation_id
        );
    }

    #[test]
    fn rejects_unsupported_system_inputs() {
        let req = request(
            vec![],
            api::request::input::Type::CodeReview(api::request::input::CodeReview::default()),
            "",
        );

        assert!(matches!(
            plan_turn(&req, &route()),
            Err(AdapterError::UnsupportedInput(_))
        ));
    }
}

mod tools {
    use super::*;

    fn plan_for(req: &api::Request) -> TurnPlan {
        plan_turn(req, &route()).expect("plan")
    }

    fn simple_request() -> api::Request {
        request(vec![], user_inputs(vec![user_query_input("hi")]), "")
    }

    #[test]
    fn builtin_definitions_cover_core_tools() {
        let plan = plan_for(&simple_request());
        let names: Vec<String> = plan
            .tools
            .definitions()
            .into_iter()
            .map(|d| d.name)
            .collect();

        for expected in [
            "run_shell_command",
            "read_files",
            "edit_files",
            "grep",
            "file_glob",
        ] {
            assert!(
                names.contains(&expected.to_string()),
                "missing tool {expected}: {names:?}"
            );
        }
        for definition in plan.tools.definitions() {
            assert_eq!(
                definition.parameters["type"],
                json!("object"),
                "{} must have an object schema",
                definition.name
            );
            assert!(!definition.description.is_empty());
        }
    }

    #[test]
    fn maps_run_shell_command_arguments_to_proto() {
        let plan = plan_for(&simple_request());

        let tool_call = plan
            .tools
            .to_proto_tool_call(&ToolCallRequest {
                id: "call_9".to_string(),
                name: "run_shell_command".to_string(),
                arguments: json!({ "command": "echo hi" }),
            })
            .expect("mapping");

        assert_eq!(tool_call.tool_call_id, "call_9");
        let Some(api::message::tool_call::Tool::RunShellCommand(shell)) = tool_call.tool else {
            panic!("expected RunShellCommand, got {:?}", tool_call.tool);
        };
        assert_eq!(shell.command, "echo hi");
        // v1 always waits for completion: no long-running command support.
        assert_eq!(
            shell.wait_until_complete_value,
            Some(api::message::tool_call::run_shell_command::WaitUntilCompleteValue::WaitUntilComplete(true))
        );
    }

    #[test]
    fn maps_edit_files_arguments_to_apply_file_diffs() {
        let plan = plan_for(&simple_request());

        let tool_call = plan
            .tools
            .to_proto_tool_call(&ToolCallRequest {
                id: "call_2".to_string(),
                name: "edit_files".to_string(),
                arguments: json!({
                    "summary": "rename variable",
                    "diffs": [{
                        "file_path": "src/main.rs",
                        "search": "let a = 1;",
                        "replace": "let b = 1;",
                    }],
                }),
            })
            .expect("mapping");

        let Some(api::message::tool_call::Tool::ApplyFileDiffs(apply)) = tool_call.tool else {
            panic!("expected ApplyFileDiffs, got {:?}", tool_call.tool);
        };
        assert_eq!(apply.summary, "rename variable");
        assert_eq!(apply.diffs.len(), 1);
        assert_eq!(apply.diffs[0].file_path, "src/main.rs");
        assert_eq!(apply.diffs[0].search, "let a = 1;");
        assert_eq!(apply.diffs[0].replace, "let b = 1;");
    }

    #[test]
    fn projects_mcp_tools_and_routes_calls_back_to_the_server() {
        let mut req = simple_request();
        req.mcp_context = Some(mcp_context());
        let plan = plan_for(&req);

        let mcp_definition = plan
            .tools
            .definitions()
            .into_iter()
            .find(|d| d.name.starts_with("mcp__") && d.name.ends_with("__create_issue"))
            .expect("projected MCP tool definition");
        assert_eq!(mcp_definition.parameters["type"], json!("object"));

        let tool_call = plan
            .tools
            .to_proto_tool_call(&ToolCallRequest {
                id: "call_3".to_string(),
                name: mcp_definition.name.clone(),
                arguments: json!({ "title": "bug" }),
            })
            .expect("mapping");

        let Some(api::message::tool_call::Tool::CallMcpTool(call)) = tool_call.tool else {
            panic!("expected CallMcpTool, got {:?}", tool_call.tool);
        };
        assert_eq!(call.server_id, "srv-1");
        assert_eq!(call.name, "create_issue");
        let args = call.args.expect("args struct");
        assert_eq!(
            args.fields["title"].kind,
            Some(prost_types::value::Kind::StringValue("bug".to_string()))
        );
    }

    #[test]
    fn unknown_tool_names_are_an_error() {
        let plan = plan_for(&simple_request());

        assert!(matches!(
            plan.tools.to_proto_tool_call(&ToolCallRequest {
                id: "call_4".to_string(),
                name: "made_up_tool".to_string(),
                arguments: json!({}),
            }),
            Err(AdapterError::UnknownTool(_))
        ));
    }
}

mod tool_result_rendering {
    use super::*;

    #[test]
    fn renders_finished_shell_command_with_output_and_exit_code() {
        let result = api::message::ToolCallResult {
            tool_call_id: "call_1".to_string(),
            context: None,
            result: Some(api::message::tool_call_result::Result::RunShellCommand(
                shell_finished_result("file.txt\n", 2),
            )),
        };

        let text = render_tool_result_text(&result);

        assert!(text.contains("file.txt"), "missing output: {text}");
        assert!(text.contains('2'), "missing exit code: {text}");
    }

    #[test]
    fn renders_read_files_error() {
        let result = api::message::ToolCallResult {
            tool_call_id: "call_1".to_string(),
            context: None,
            result: Some(api::message::tool_call_result::Result::ReadFiles(
                api::ReadFilesResult {
                    result: Some(api::read_files_result::Result::Error(
                        api::read_files_result::Error {
                            message: "no such file: missing.rs".to_string(),
                        },
                    )),
                },
            )),
        };

        let text = render_tool_result_text(&result);

        assert!(
            text.contains("no such file: missing.rs"),
            "missing error: {text}"
        );
    }
}

mod event_mapping {
    use super::*;

    fn first_turn_plan() -> TurnPlan {
        let req = request(vec![], user_inputs(vec![user_query_input("hi")]), "");
        plan_turn(&req, &route()).expect("plan")
    }

    fn follow_up_plan() -> TurnPlan {
        let req = request(
            vec![root_task(vec![
                user_query_message("m1", "list files"),
                shell_tool_call_message("m2", "call_1", "ls"),
            ])],
            user_inputs(vec![shell_result_input("call_1", "file.txt\n", 0)]),
            "conv-1",
        );
        plan_turn(&req, &route()).expect("plan")
    }

    #[test]
    fn first_turn_starts_with_init_transaction_create_task_and_echoes() {
        let plan = first_turn_plan();
        let mut mapper = EventMapper::new(&plan);

        let events = mapper.initial_events();

        assert_eq!(
            event_kinds(&events),
            vec!["init", "begin_transaction", "create_task", "add_messages"],
        );
        let Some(Ok(api::ResponseEvent {
            r#type: Some(api::response_event::Type::Init(init)),
        })) = events.first()
        else {
            panic!("first event must be Init");
        };
        assert_eq!(init.conversation_id, plan.conversation_id);
        assert!(!init.request_id.is_empty());

        let echoes = added_messages(&events);
        assert!(
            echoes.iter().any(|m| matches!(
                &m.message,
                Some(api::message::Message::UserQuery(q)) if q.query == "hi"
            )),
            "user query must be echoed into the task tree"
        );
        assert!(
            echoes.iter().any(|m| matches!(
                &m.message,
                Some(api::message::Message::ModelUsed(used)) if used.model_id == CONFIG_KEY
            )),
            "ModelUsed must report the custom model's config key"
        );
    }

    #[test]
    fn follow_up_turn_does_not_recreate_the_root_task() {
        let plan = follow_up_plan();
        let mut mapper = EventMapper::new(&plan);

        let events = mapper.initial_events();

        assert!(
            !event_kinds(&events).contains(&"create_task".to_string()),
            "follow-up turns must not re-emit CreateTask"
        );
        let Some(Ok(api::ResponseEvent {
            r#type: Some(api::response_event::Type::Init(init)),
        })) = events.first()
        else {
            panic!("first event must be Init");
        };
        assert_eq!(init.conversation_id, "conv-1");
    }

    #[test]
    fn text_deltas_stream_as_one_add_then_appends() {
        let plan = first_turn_plan();
        let mut mapper = EventMapper::new(&plan);
        mapper.initial_events();

        let first = mapper.on_event(Ok(AgentEvent::TextDelta("Hel".to_string())));
        let second = mapper.on_event(Ok(AgentEvent::TextDelta("lo".to_string())));

        let first_added = added_messages(&first);
        assert_eq!(first_added.len(), 1, "first delta creates the message");
        let Some(api::message::Message::AgentOutput(output)) = &first_added[0].message else {
            panic!("expected AgentOutput, got {:?}", first_added[0].message);
        };
        assert_eq!(output.text, "Hel");

        assert_eq!(event_kinds(&second), vec!["append"]);
        let appended = second
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find_map(|e| match e.r#type.as_ref() {
                Some(api::response_event::Type::ClientActions(actions)) => actions
                    .actions
                    .iter()
                    .find_map(|a| match a.action.as_ref() {
                        Some(api::client_action::Action::AppendToMessageContent(append)) => {
                            Some(append.clone())
                        }
                        _ => None,
                    }),
                _ => None,
            })
            .expect("append action");
        assert_eq!(appended.task_id, plan.root_task_id);
        let appended_message = appended.message.expect("appended message");
        assert_eq!(
            appended_message.id, first_added[0].id,
            "appends must target the message created by the first delta"
        );
        assert!(
            matches!(
                &appended_message.message,
                Some(api::message::Message::AgentOutput(o)) if o.text == "lo"
            ),
            "append must carry only the delta"
        );
        assert!(appended.mask.is_some_and(|mask| !mask.paths.is_empty()));
    }

    /// Applies the mapper's append events through the same field-mask
    /// machinery `task.rs` uses, proving the emitted mask path actually
    /// appends text on the real client (risky assumption #2 of the plan).
    #[test]
    fn emitted_append_mask_path_works_with_the_client_field_mask_machinery() {
        let plan = first_turn_plan();
        let mut mapper = EventMapper::new(&plan);
        mapper.initial_events();

        let first = mapper.on_event(Ok(AgentEvent::TextDelta("Hel".to_string())));
        let second = mapper.on_event(Ok(AgentEvent::TextDelta("lo".to_string())));

        let existing = added_messages(&first).remove(0);
        let append = second
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find_map(|e| match e.r#type.as_ref() {
                Some(api::response_event::Type::ClientActions(actions)) => actions
                    .actions
                    .iter()
                    .find_map(|a| match a.action.as_ref() {
                        Some(api::client_action::Action::AppendToMessageContent(append)) => {
                            Some(append.clone())
                        }
                        _ => None,
                    }),
                _ => None,
            })
            .expect("append action");

        let merged = field_mask::FieldMaskOperation::append(
            &api::MESSAGE_DESCRIPTOR,
            &existing,
            &append.message.expect("appended message"),
            append.mask.expect("mask"),
        )
        .apply()
        .expect("field mask append must succeed");

        assert!(
            matches!(
                &merged.message,
                Some(api::message::Message::AgentOutput(o)) if o.text == "Hello"
            ),
            "append must concatenate the delta onto the existing text, got {:?}",
            merged.message
        );
    }

    #[test]
    fn tool_calls_are_added_to_the_task_with_their_openai_call_id() {
        let plan = first_turn_plan();
        let mut mapper = EventMapper::new(&plan);
        mapper.initial_events();

        let events = mapper.on_event(Ok(AgentEvent::ToolCall(ToolCallRequest {
            id: "call_9".to_string(),
            name: "run_shell_command".to_string(),
            arguments: json!({ "command": "echo hi" }),
        })));

        let added = added_messages(&events);
        let tool_call = added
            .iter()
            .find_map(|m| match &m.message {
                Some(api::message::Message::ToolCall(call)) => Some(call.clone()),
                _ => None,
            })
            .expect("tool call message");
        assert_eq!(tool_call.tool_call_id, "call_9");
        assert!(matches!(
            tool_call.tool,
            Some(api::message::tool_call::Tool::RunShellCommand(shell)) if shell.command == "echo hi"
        ));
    }

    #[test]
    fn done_commits_the_transaction_and_finishes_with_token_usage() {
        let plan = first_turn_plan();
        let mut mapper = EventMapper::new(&plan);
        mapper.initial_events();
        mapper.on_event(Ok(AgentEvent::TextDelta("hi".to_string())));
        mapper.on_event(Ok(AgentEvent::Usage {
            input_tokens: 42,
            output_tokens: 7,
        }));

        let events = mapper.on_event(Ok(AgentEvent::Done));

        assert_eq!(event_kinds(&events), vec!["commit_transaction", "finished"]);
        let finished = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find_map(|e| match e.r#type.as_ref() {
                Some(api::response_event::Type::Finished(finished)) => Some(finished.clone()),
                _ => None,
            })
            .expect("finished event");
        assert!(matches!(
            finished.reason,
            Some(api::response_event::stream_finished::Reason::Done(_))
        ));
        assert_eq!(finished.token_usage.len(), 1);
        assert_eq!(finished.token_usage[0].model_id, CONFIG_KEY);
        assert_eq!(finished.token_usage[0].total_input, 42);
        assert_eq!(finished.token_usage[0].output, 7);
    }

    #[test]
    fn invalid_api_key_rolls_back_and_finishes_with_invalid_api_key() {
        let plan = first_turn_plan();
        let mut mapper = EventMapper::new(&plan);
        mapper.initial_events();
        mapper.on_event(Ok(AgentEvent::TextDelta("partial".to_string())));

        let events = mapper.on_event(Err(LocalAgentError::InvalidApiKey {
            model_slug: MODEL_SLUG.to_string(),
        }));

        let kinds = event_kinds(&events);
        assert!(
            kinds.contains(&"rollback_transaction".to_string()),
            "got {kinds:?}"
        );
        let finished = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find_map(|e| match e.r#type.as_ref() {
                Some(api::response_event::Type::Finished(finished)) => Some(finished.clone()),
                _ => None,
            })
            .expect("finished event");
        assert!(matches!(
            finished.reason,
            Some(api::response_event::stream_finished::Reason::InvalidApiKey(
                _
            ))
        ));
    }

    #[test]
    fn context_window_overflow_finishes_with_context_window_exceeded() {
        let plan = first_turn_plan();
        let mut mapper = EventMapper::new(&plan);
        mapper.initial_events();

        let events = mapper.on_event(Err(LocalAgentError::ContextWindowExceeded {
            message: "maximum context length exceeded".to_string(),
        }));

        let finished = events
            .iter()
            .filter_map(|e| e.as_ref().ok())
            .find_map(|e| match e.r#type.as_ref() {
                Some(api::response_event::Type::Finished(finished)) => Some(finished.clone()),
                _ => None,
            })
            .expect("finished event");
        assert!(matches!(
            finished.reason,
            Some(api::response_event::stream_finished::Reason::ContextWindowExceeded(_))
        ));
    }

    #[test]
    fn transport_errors_surface_as_stream_errors_for_retry() {
        let plan = first_turn_plan();
        let mut mapper = EventMapper::new(&plan);
        mapper.initial_events();

        let events = mapper.on_event(Err(LocalAgentError::Transport(
            "connection reset".to_string(),
        )));

        assert!(
            events.iter().any(|e| e.is_err()),
            "transport failures must surface as stream errors so the existing retry machinery applies"
        );
    }
}

mod accepted_passive_suggestion {
    use super::*;

    fn passive_result_input(
        command: &str,
        output: &str,
        exit_code: i32,
        prompt: &str,
    ) -> api::request::input::user_inputs::user_input::Input {
        api::request::input::user_inputs::user_input::Input::PassiveSuggestionResult(
            api::request::input::user_inputs::PassiveSuggestionResultInput {
                result: Some(api::PassiveSuggestionResultType {
                    trigger: Some(
                        api::passive_suggestion_result_type::Trigger::ExecutedShellCommand(
                            api::ExecutedShellCommand {
                                command: command.to_string(),
                                output: output.to_string(),
                                exit_code,
                                ..Default::default()
                            },
                        ),
                    ),
                    suggestion: Some(api::passive_suggestion_result_type::Suggestion::Prompt(
                        api::passive_suggestion_result_type::Prompt {
                            prompt: prompt.to_string(),
                        },
                    )),
                }),
            },
        )
    }

    #[test]
    fn accepted_suggestion_brings_trigger_into_context_alongside_the_query() {
        // The client sends an accepted suggestion only as a PassiveSuggestionResult
        // (no separate UserQuery), so the adapter derives the durable query from it.
        let req = request(
            vec![],
            user_inputs(vec![passive_result_input(
                "fd . .go",
                "[fd error]: Search path '.go' is not a directory.",
                1,
                "Find go files",
            )]),
            "",
        );

        let plan = plan_turn(&req, &route())
            .expect("accepted passive suggestion must plan as an agent turn");

        assert!(
            plan.transcript.iter().any(|m| matches!(
                m,
                ChatMessage::User(text) if text.contains("fd . .go") && text.contains("[fd error]")
            )),
            "the triggering command and its output must be brought into context"
        );
        assert!(
            plan.transcript
                .iter()
                .any(|m| matches!(m, ChatMessage::User(text) if text == "Find go files")),
            "the accepted prompt must be present as the user query"
        );
        assert_eq!(plan.config_key, CONFIG_KEY);
    }

    #[test]
    fn accepted_suggestion_echoes_the_query_into_the_task_tree() {
        let req = request(
            vec![],
            user_inputs(vec![passive_result_input(
                "fd . .go",
                "err",
                1,
                "Find go files",
            )]),
            "",
        );

        let plan = plan_turn(&req, &route()).expect("plan");

        // The trigger is transient per-turn context; the accepted prompt is the
        // durable record, echoed into the task tree as a user query so follow-up
        // turns see it.
        assert!(
            plan.input_echo_messages.iter().any(|m| matches!(
                &m.message,
                Some(api::message::Message::UserQuery(q)) if q.query == "Find go files"
            )),
            "the accepted prompt must be echoed as the durable user query"
        );
        assert!(
            !plan.input_echo_messages.iter().any(|m| matches!(
                &m.message,
                Some(api::message::Message::PassiveSuggestionResult(_))
            )),
            "the passive suggestion result is not echoed as a durable message"
        );
    }

    #[test]
    fn follow_up_turn_after_accepted_suggestion_keeps_the_new_query() {
        // Regression: after accepting a passive suggestion that ran a tool, the
        // follow-up (tool-result) turn used to rebuild a transcript whose last
        // user message was the *previous* conversation turn, so the model answered
        // the old question. The accepted prompt — echoed as a durable UserQuery on
        // the accept turn — must be the last user message instead.
        let tasks = vec![root_task(vec![
            // A previous, unrelated exchange in the same conversation.
            user_query_message("m1", "Are you working?"),
            agent_output_message("m2", "Yes, I'm working. How can I help?"),
            // The accepted suggestion (echoed as the durable query on the accept
            // turn) plus the tool call the model made for it.
            user_query_message("m3", "Find go files"),
            shell_tool_call_message("m4", "call-1", "fd . -e go"),
        ])];

        let req = request(
            tasks,
            user_inputs(vec![shell_result_input("call-1", "main.go\nutil.go", 0)]),
            "conv-1",
        );

        let plan = plan_turn(&req, &route()).expect("plan");

        let last_user = plan
            .transcript
            .iter()
            .rev()
            .find_map(|m| match m {
                ChatMessage::User(text) => Some(text.as_str()),
                _ => None,
            })
            .expect("transcript must contain a user message");
        assert_eq!(
            last_user, "Find go files",
            "the follow-up turn must answer the accepted prompt, not the previous turn"
        );
    }
}

mod passive_suggestions {
    use super::*;

    fn shell_completed_input(
        command: &str,
        output: &str,
        exit_code: i32,
    ) -> api::request::input::Type {
        api::request::input::Type::GeneratePassiveSuggestions(
            api::request::input::GeneratePassiveSuggestions {
                attachments: vec![],
                trigger: Some(
                    api::request::input::generate_passive_suggestions::Trigger::ShellCommandCompleted(
                        api::request::input::generate_passive_suggestions::ShellCommandCompleted {
                            executed_shell_command: Some(api::ExecutedShellCommand {
                                command: command.to_string(),
                                output: output.to_string(),
                                exit_code,
                                ..Default::default()
                            }),
                            relevant_files: vec![],
                        },
                    ),
                ),
            },
        )
    }

    fn agent_response_completed_input() -> api::request::input::Type {
        api::request::input::Type::GeneratePassiveSuggestions(
            api::request::input::GeneratePassiveSuggestions {
                attachments: vec![],
                trigger: Some(
                    api::request::input::generate_passive_suggestions::Trigger::AgentResponseCompleted(
                        api::request::input::generate_passive_suggestions::AgentResponseCompleted {},
                    ),
                ),
            },
        )
    }

    fn shell_trigger_plan() -> TurnPlan {
        let req = request(
            vec![],
            shell_completed_input("cargo test", "error[E0308]: mismatched types\n", 101),
            "",
        );
        plan_turn(&req, &route()).expect("passive suggestion turns must plan")
    }

    #[test]
    fn shell_trigger_plans_a_suggestion_turn_with_only_the_suggest_prompt_tool() {
        let plan = shell_trigger_plan();

        let tool_names: Vec<_> = plan
            .tools
            .definitions()
            .into_iter()
            .map(|tool| tool.name)
            .collect();
        assert_eq!(
            tool_names,
            vec!["suggest_prompt".to_string()],
            "suggestion turns are one-shot and read-only: no shell/file/MCP tools"
        );
    }

    #[test]
    fn shell_trigger_renders_command_and_output_into_the_transcript() {
        let plan = shell_trigger_plan();

        let ChatMessage::System(system) = &plan.transcript[0] else {
            panic!("transcript must start with a system prompt");
        };
        assert!(
            system.contains("suggest_prompt"),
            "system prompt must instruct the model to use the suggest_prompt tool, got: {system}"
        );

        let ChatMessage::User(instruction) = plan
            .transcript
            .last()
            .expect("transcript must not be empty")
        else {
            panic!("transcript must end with the trigger rendered as a user message");
        };
        assert!(
            instruction.contains("cargo test"),
            "missing command: {instruction}"
        );
        assert!(
            instruction.contains("error[E0308]"),
            "missing command output: {instruction}"
        );
        assert!(
            instruction.contains("101"),
            "missing exit code: {instruction}"
        );
    }

    #[test]
    fn shell_trigger_echoes_the_trigger_message_into_the_task_tree() {
        let plan = shell_trigger_plan();

        assert!(
            plan.input_echo_messages.iter().any(|m| matches!(
                &m.message,
                Some(api::message::Message::SystemQuery(
                    api::message::SystemQuery {
                        r#type: Some(
                            api::message::system_query::Type::GeneratePassiveSuggestions(_)
                        ),
                        ..
                    }
                ))
            )),
            "the trigger must be echoed so follow-up turns can replay it from the task tree"
        );
    }

    #[test]
    fn agent_response_trigger_replays_the_conversation_history() {
        let req = request(
            vec![root_task(vec![
                user_query_message("m1", "how do I deploy this"),
                agent_output_message("m2", "Use kubectl apply with the manifest."),
            ])],
            agent_response_completed_input(),
            "conv-1",
        );

        let plan = plan_turn(&req, &route()).expect("plan");

        assert!(
            plan.transcript
                .iter()
                .any(|m| matches!(m, ChatMessage::User(text) if text == "how do I deploy this")),
            "conversation history must be replayed for follow-up suggestions"
        );
        assert!(
            plan.transcript.iter().any(|m| matches!(
                m,
                ChatMessage::Assistant { text, .. } if text.contains("kubectl apply")
            )),
            "agent responses must be part of the replayed history"
        );
        let ChatMessage::User(instruction) = plan
            .transcript
            .last()
            .expect("transcript must not be empty")
        else {
            panic!("transcript must end with the suggestion instruction");
        };
        assert!(
            instruction.contains("suggest_prompt"),
            "the final instruction must point at the suggest_prompt tool: {instruction}"
        );
    }

    #[test]
    fn suggest_prompt_tool_call_maps_to_a_prompt_chip_message() {
        let plan = shell_trigger_plan();
        let mut mapper = EventMapper::new(&plan);
        mapper.initial_events();

        let call_events = mapper.on_event(Ok(AgentEvent::ToolCall(ToolCallRequest {
            id: "call-1".to_string(),
            name: "suggest_prompt".to_string(),
            arguments: json!({
                "prompt": "Fix the type mismatch in cargo test",
                "label": "Fix failing test",
            }),
        })));
        let done_events = mapper.on_event(Ok(AgentEvent::Done));

        let added = added_messages(&call_events);
        let suggestion = added
            .iter()
            .find_map(|m| match &m.message {
                Some(api::message::Message::ToolCall(api::message::ToolCall {
                    tool: Some(api::message::tool_call::Tool::SuggestPrompt(suggest)),
                    ..
                })) => Some(suggest.clone()),
                _ => None,
            })
            .expect("a suggest_prompt call must map to a ToolCall::SuggestPrompt message");
        let Some(api::message::tool_call::suggest_prompt::DisplayMode::PromptChip(chip)) =
            suggestion.display_mode
        else {
            panic!("local suggestions must use the PromptChip display mode");
        };
        assert_eq!(chip.prompt, "Fix the type mismatch in cargo test");
        assert_eq!(chip.label, "Fix failing test");

        assert_eq!(
            event_kinds(&done_events),
            vec!["commit_transaction", "finished"],
        );
    }

    #[test]
    fn suggest_prompt_label_is_optional() {
        let plan = shell_trigger_plan();
        let mut mapper = EventMapper::new(&plan);
        mapper.initial_events();

        let events = mapper.on_event(Ok(AgentEvent::ToolCall(ToolCallRequest {
            id: "call-1".to_string(),
            name: "suggest_prompt".to_string(),
            arguments: json!({ "prompt": "Re-run with --verbose" }),
        })));

        let added = added_messages(&events);
        let chip = added
            .iter()
            .find_map(|m| match &m.message {
                Some(api::message::Message::ToolCall(api::message::ToolCall {
                    tool: Some(api::message::tool_call::Tool::SuggestPrompt(
                        api::message::tool_call::SuggestPrompt {
                            display_mode:
                                Some(api::message::tool_call::suggest_prompt::DisplayMode::PromptChip(
                                    chip,
                                )),
                            ..
                        },
                    )),
                    ..
                })) => Some(chip.clone()),
                _ => None,
            })
            .expect("suggestion message");
        assert_eq!(chip.prompt, "Re-run with --verbose");
        assert_eq!(chip.label, "");
    }

    #[test]
    fn suggestion_turn_with_no_tool_call_finishes_cleanly() {
        let plan = shell_trigger_plan();
        let mut mapper = EventMapper::new(&plan);
        mapper.initial_events();

        let text_events = mapper.on_event(Ok(AgentEvent::TextDelta(
            "Nothing worth suggesting.".to_string(),
        )));
        let done_events = mapper.on_event(Ok(AgentEvent::Done));

        assert!(
            !added_messages(&text_events)
                .iter()
                .any(|m| matches!(&m.message, Some(api::message::Message::ToolCall(_)))),
            "declining must not fabricate a suggestion"
        );
        assert_eq!(
            event_kinds(&done_events),
            vec!["commit_transaction", "finished"],
        );
    }

    #[test]
    fn suggestion_turns_run_against_the_routed_endpoint_model() {
        let plan = shell_trigger_plan();
        assert_eq!(plan.config_key, CONFIG_KEY);
        assert!(plan.is_first_turn);
    }
}
