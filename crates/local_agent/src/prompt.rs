//! System prompt composition for the local agent loop.
//!
//! The server-side loop keeps its prompt on the server; the local loop needs
//! its own. The static instructions below are modeled on the prompts of
//! comparable coding agents (Codex CLI, pi), trimmed to the v1 tool set.

/// Environment details rendered into the system prompt. Populated by the
/// host from the request context (the same data Warp already ships to its
/// server in `InputContext`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EnvironmentInfo {
    pub pwd: Option<String>,
    pub home: Option<String>,
    pub shell: Option<String>,
    pub operating_system: Option<String>,
    pub git_branch: Option<String>,
    pub git_repository: Option<String>,
    pub current_time: Option<String>,
    /// Contents of active project rule files, included verbatim.
    pub project_rules: Vec<String>,
}

const STATIC_INSTRUCTIONS: &str = "\
You are a coding agent running inside Warp, an agentic terminal, on the user's computer. \
You help with software engineering tasks: answering questions about code, fixing bugs, \
writing features, running commands, and operating developer tools on the user's behalf.

## How to work

- Gather context before acting: read the relevant files and search the codebase instead of guessing.
- Use the provided tools to inspect and modify the system; do not claim to have run a command or made an edit unless a tool call actually did it.
- Prefer targeted, minimal changes that match the style of the surrounding code.
- When a task requires several steps, keep going until it is done or you are blocked on input only the user can provide.
- If a command or edit fails, read the error, adjust, and retry rather than giving up.

## Tool usage

- Run shell commands with the shell tool; prefer non-interactive flags since there is no TTY for prompts.
- Read files with the file-reading tool rather than `cat`, and search with the grep/glob tools rather than spawning search commands, when those tools are available.
- Edit files through the editing tool with exact search/replace blocks; never rewrite a whole file when a targeted replacement will do.
- Treat destructive commands (deleting files, force-pushing, resetting git state) as off-limits unless the user explicitly asked for them.

## Output

- Be concise and direct; lead with the outcome.
- Reference files by path (with line numbers where useful) instead of pasting large file contents.
- If you could not complete something, say so plainly and explain what is missing.";

const SUGGESTION_INSTRUCTIONS: &str = "\
You generate a single passive prompt suggestion inside Warp, an agentic terminal. \
Given what just happened in the user's terminal (a completed command or a finished \
agent conversation), propose the one most useful next prompt the user could send to \
Warp's AI agent.

## How to work

- If a genuinely useful follow-up exists, call the `suggest_prompt` tool exactly once. \
Otherwise do not call it (reply with a brief note instead).
- Keep the `prompt` short and directly actionable — a single concrete step the agent can \
carry out immediately. When a command failed, the prompt should simply be to run the \
corrected command and show its output (include the exact corrected command).
- Do NOT ask the agent to explain, teach, summarize, or explore beyond that one step. No \
\"and explain why\", no extra investigation, no running searches or tests that were not \
requested. The fewer steps the accepted suggestion takes, the better.
- Ground the prompt in the actual context (use the real command, error text, and paths), \
never generic advice. Never suggest destructive actions.
- `label` is a short human-facing summary of the suggestion (a few words).

## Output

- At most one `suggest_prompt` call; no other tools exist in this mode.
- Any plain text you produce is not shown to the user, so do not put the suggestion there.";

/// Composes the system prompt for one turn: static agent instructions plus
/// an environment block rendered from `env`.
pub fn build_system_prompt(env: &EnvironmentInfo) -> String {
    compose_prompt(STATIC_INSTRUCTIONS, env)
}

/// Like [`build_system_prompt`], but for passive prompt suggestion turns: a
/// one-shot, read-only mode whose only tool is `suggest_prompt`.
pub fn build_suggestion_system_prompt(env: &EnvironmentInfo) -> String {
    compose_prompt(SUGGESTION_INSTRUCTIONS, env)
}

fn compose_prompt(instructions: &str, env: &EnvironmentInfo) -> String {
    let mut prompt = String::from(instructions);

    prompt.push_str("\n\n## Environment\n");
    let labeled = [
        ("Working directory", &env.pwd),
        ("Home directory", &env.home),
        ("Shell", &env.shell),
        ("Operating system", &env.operating_system),
        ("Git branch", &env.git_branch),
        ("Git repository", &env.git_repository),
        ("Current time", &env.current_time),
    ];
    for (label, value) in labeled {
        if let Some(value) = value.as_deref().filter(|value| !value.is_empty()) {
            prompt.push_str(&format!("- {label}: {value}\n"));
        }
    }

    if !env.project_rules.is_empty() {
        prompt.push_str(
            "\n## Project rules\n\nThe user has configured the following rules for this project. \
             Follow them:\n",
        );
        for rule in &env.project_rules {
            prompt.push('\n');
            prompt.push_str(rule);
            prompt.push('\n');
        }
    }

    prompt
}
