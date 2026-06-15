//! Decides whether an agent request should be served by the local agent loop
//! (`crates/local_agent`) instead of the Warp API server.
//!
//! This build always runs the agent loop locally. A request routes locally
//! when the selected base model belongs to one of the user's custom endpoints
//! (its `config_key` appears in `settings.custom_model_providers`) and the
//! input is of a kind the local engine supports (user queries, tool call
//! results, and passive prompt suggestions). Anything else is blocked rather
//! than forwarded — the client never talks to the Warp API server.

use warp_multi_agent_api as api;

/// A resolved local route: the endpoint to call plus the `config_key` the
/// adapter reports back in `TokenUsage`/`ModelUsed` events.
#[derive(Debug, Clone)]
pub(crate) struct LocalRoute {
    pub endpoint: local_agent::LocalEndpointConfig,
    pub config_key: String,
}

/// What `generate_multi_agent_output` should do with a request.
#[derive(Debug)]
pub(crate) enum RequestDisposition {
    /// Serve the request locally via the resolved route.
    Local(LocalRoute),
    /// Forward the request to the Warp API server (the legacy path). This build
    /// never produces it on the native path — the local loop is always on — but
    /// the variant is retained so the request type stays total.
    #[allow(dead_code)]
    Server,
    /// The request cannot be served locally. It must NOT reach the server;
    /// surface the carried reason as an error instead. This is a hard guard:
    /// the client never talks to the Warp API server, so a request that doesn't
    /// resolve locally fails loudly rather than silently leaking prompts (and
    /// referencing tasks the server never created — see the "non-existent
    /// task" failure mode).
    BlockedFromServer(&'static str),
}

/// Decides whether a request is served locally or blocked. Nothing reaches the
/// server: the only two outcomes are [`RequestDisposition::Local`] and
/// [`RequestDisposition::BlockedFromServer`].
pub(crate) fn route_request(request: &api::Request) -> RequestDisposition {
    if let Some(route) = resolve_local_route(request) {
        return RequestDisposition::Local(route);
    }
    RequestDisposition::BlockedFromServer(block_reason(request))
}

/// Explains why a request did not resolve to a local route. Reached only after
/// [`resolve_local_route`] returned `None`, so exactly one of the preconditions
/// failed.
fn block_reason(request: &api::Request) -> &'static str {
    let input_supported = request
        .input
        .as_ref()
        .and_then(|input| input.r#type.as_ref())
        .is_some_and(is_supported_input);
    if !input_supported {
        "the local agent loop does not support this kind of request"
    } else {
        "the selected model is not a custom endpoint model"
    }
}

pub(crate) fn resolve_local_route(request: &api::Request) -> Option<LocalRoute> {
    if !is_supported_input(request.input.as_ref()?.r#type.as_ref()?) {
        return None;
    }

    let settings = request.settings.as_ref()?;
    let model = &settings.model_config.as_ref()?.base;
    settings
        .custom_model_providers
        .as_ref()?
        .providers
        .iter()
        .find_map(|provider| {
            provider
                .models
                .iter()
                .find(|custom_model| custom_model.config_key == *model)
                .map(|custom_model| LocalRoute {
                    endpoint: local_agent::LocalEndpointConfig {
                        base_url: provider.base_url.clone(),
                        api_key: provider.api_key.clone(),
                        model_slug: custom_model.slug.clone(),
                    },
                    config_key: custom_model.config_key.clone(),
                })
        })
}

/// The local engine understands the plain agent-loop inputs and passive
/// prompt suggestions; everything else system-driven (code review,
/// environment creation, skills, orchestration, ...) keeps going to the
/// server.
fn is_supported_input(input_type: &api::request::input::Type) -> bool {
    use api::request::input::user_inputs::user_input::Input;
    match input_type {
        api::request::input::Type::UserInputs(user_inputs) => {
            user_inputs.inputs.iter().all(|user_input| {
                matches!(
                    user_input.input,
                    Some(
                        Input::UserQuery(_)
                            | Input::ToolCallResult(_)
                            // Sent alongside a query when the user accepts a
                            // passive suggestion, to bring it into context.
                            | Input::PassiveSuggestionResult(_)
                    )
                )
            })
        }
        api::request::input::Type::GeneratePassiveSuggestions(_) => true,
        _ => false,
    }
}

#[cfg(test)]
#[path = "local_route_tests.rs"]
mod tests;
