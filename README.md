<a href="https://www.warp.dev">
    <img width="1024" alt="Warp Agentic Development Environment product preview" src="https://github.com/user-attachments/assets/9976b2da-2edd-4604-a36c-8fd53719c6d4" />
</a>

<h1></h1>

## About this fork

This is a **fully-local fork of [Warp](https://www.warp.dev)**. It keeps Warp's
terminal and agentic development environment, but removes the dependency on
Warp's backend:

- **No Warp API.** The client never talks to Warp's servers for AI.
- **No authentication.** There is no login/account step — the app starts
  straight into a usable session.
- **The agent loop runs locally.** Instead of delegating the multi-agent loop to
  Warp's backend, this fork drives the loop on your machine and talks directly
  to an OpenAI-compatible API.

### AI features require a custom endpoint

Because nothing is served by Warp, **AI features only work once you configure a
custom OpenAI-compatible endpoint.** Point it at any compatible server — local
(Ollama, LM Studio, vLLM, llama.cpp's server, …) or remote (OpenAI, OpenRouter,
or any gateway that speaks the OpenAI API).

Configure it in **Settings → AI → Custom inference / model providers**: set the
base URL (e.g. `http://localhost:11434/v1`), an API key if your endpoint
requires one, and the model slug to use. Loopback and private-network hosts are
allowed (over plain HTTP for local hosts, HTTPS for public ones). Until an
endpoint is configured, the terminal works normally but agent/AI actions have
nowhere to go.

## Building and running this fork

The build is driven by the scripts in [`script/`](script). This fork builds as
the **OSS channel**, which needs no Warp-internal access.

### 1. One-time setup

Install the toolchain and build dependencies (Rust, `protoc`, GUI/runtime libs,
and the bundler):

```bash
./script/bootstrap
```

> The `cargo` toolchain installs under `~/.cargo`; make sure `~/.cargo/bin` is on
> your `PATH` (`export PATH="$HOME/.cargo/bin:$PATH"`).

See [WARP.md](WARP.md) for the full engineering guide (coding style, testing,
and platform notes).

### 2. Debug build — for development & debugging

Build and run straight from source with debug assertions. This is the fastest
edit/run loop and what you want while hacking on the fork:

```bash
./script/run
```

### 3. Release build — a real, installed app for daily use

For everyday use you want an optimized, packaged application you launch from your
desktop — not a binary you start from another terminal. Build a release bundle
of the OSS channel:

```bash
./script/bundle --channel oss
```

**Linux.** To get a proper desktop entry (launcher icon, system integration) on
Debian/Ubuntu, build a `.deb` instead and install it system-wide:

```bash
./script/bundle --channel oss --packages deb --release-tag v0.1.0
sudo dpkg -i target/release-lto/bundle/linux/warp-terminal-oss_0.1.0_amd64.deb
```

After that, **WarpOss** appears in your application launcher like any other
installed app. (`--packages` also accepts `rpm` and `arch`.)

**macOS.** `./script/bundle --channel oss` produces a `.app` bundle (and a
matching `.dmg`) at:

```
target/release-lto/bundle/osx/WarpOss.app
target/release-lto/bundle/osx/WarpOss.dmg
```

Copy the app into `/Applications` and launch it normally:

```bash
cp -R target/release-lto/bundle/osx/WarpOss.app /Applications/
```

(Or open `WarpOss.dmg` and drag **WarpOss** into Applications.)

---

The rest of this README is inherited from upstream Warp.

## About

[Warp](https://www.warp.dev) is an agentic development environment, born out of the terminal. Use Warp's built-in coding agent, or bring your own CLI agent (Claude Code, Codex, Gemini CLI, and others).

## Installation

You can [download Warp](https://www.warp.dev/download) and [read our docs](https://docs.warp.dev/) for platform-specific instructions.

## Warp Contributions Overview Dashboard

Explore [build.warp.dev](https://build.warp.dev) to:
- Watch thousands of Oz agents triage issues, write specs, implement changes, and review PRs
- View top contributors and in-flight features
- Track your own issues with GitHub sign-in
- Click into active agent sessions in a web-compiled Warp terminal

## Oz for OSS

Maintaining a popular open-source project? [Apply for Oz credits](https://tally.so/r/LZWxqG) to explore [Oz for OSS](https://github.com/warpdotdev/oz-for-oss).

Oz for OSS is our partner program for bringing the same agentic open-source management workflows used in this repository to select partner repositories. We work directly with maintainers to implement workflows for issue triage, PR review, community management, and contributor coordination in a way that fits each project.

## Licensing

Warp's UI framework (the `warpui_core` and `warpui` crates) are licensed under the [MIT license](LICENSE-MIT).

The rest of the code in this repository is licensed under the [AGPL v3](LICENSE-AGPL).

## Open Source & Contributing

Warp's client codebase is open source and lives in this repository. We welcome community contributions and have designed a lightweight workflow to help new contributors get started. For the full contribution flow, read our [CONTRIBUTING.md](CONTRIBUTING.md) guide.

> [!TIP]
> **Chat with contributors and the Warp team** in the [`#oss-contributors`](https://warpcommunity.slack.com/archives/C0B0LM8N4DB) Slack channel — a good place for ad-hoc questions, design discussion, and pairing with maintainers. New here? [Join the Warp Slack community](https://go.warp.dev/join-preview) first, then jump into `#oss-contributors`.

### Issue to PR

Before filing, [search existing issues](https://github.com/warpdotdev/warp/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc) for your bug or feature request. If nothing exists, [file an issue](https://github.com/warpdotdev/warp/issues/new/choose) using our templates. Security vulnerabilities should be reported privately as described in [CONTRIBUTING.md](CONTRIBUTING.md#reporting-security-issues).

Once filed, a Warp maintainer reviews the issue and may apply a readiness label: [`ready-to-spec`](https://github.com/warpdotdev/warp/issues?q=is%3Aissue+is%3Aopen+label%3Aready-to-spec) signals the design is open for contributors to spec out, and [`ready-to-implement`](https://github.com/warpdotdev/warp/issues?q=is%3Aissue+is%3Aopen+label%3Aready-to-implement) signals the design is settled and code PRs are welcome. Anyone can pick up a labeled issue — mention **@oss-maintainers** on an issue if you'd like it considered for a readiness label.

### Building the Repo Locally

To build and run Warp from source:

```bash
./script/bootstrap   # platform-specific setup
./script/run         # build and run Warp
./script/presubmit   # fmt, clippy, and tests
```

See [AGENTS.md](AGENTS.md) for the full engineering guide, including coding style, testing, and platform-specific notes.

## Joining the Team

Interested in joining the team? See our [open roles](https://www.warp.dev/careers).

## Support and Questions

1. See our [docs](https://docs.warp.dev/) for a comprehensive guide to Warp's features.
2. Join our [Slack Community](https://go.warp.dev/join-preview) to connect with other users and get help from the Warp team — contributors hang out in [`#oss-contributors`](https://warpcommunity.slack.com/archives/C0B0LM8N4DB).
3. Try our [Preview build](https://www.warp.dev/download-preview) to test the latest experimental features.
4. Mention **@oss-maintainers** on any issue to escalate to the team — for example, if you encounter problems with the automated agents.

## Code of Conduct

We ask everyone to be respectful and empathetic. Warp follows the [Code of Conduct](CODE_OF_CONDUCT.md). To report violations, email warp-coc at warp.dev.

## Open Source Dependencies

We'd like to call out a few of the [open source dependencies](https://docs.warp.dev/help/licenses) that have helped Warp to get off the ground:

- [Tokio](https://github.com/tokio-rs/tokio)
- [NuShell](https://github.com/nushell/nushell)
- [Fig Completion Specs](https://github.com/withfig/autocomplete)
- [Warp Server Framework](https://github.com/seanmonstar/warp)
- [Alacritty](https://github.com/alacritty/alacritty)
- [Hyper HTTP library](https://github.com/hyperium/hyper)
- [FontKit](https://github.com/servo/font-kit)
- [Core-foundation](https://github.com/servo/core-foundation-rs)
- [Smol](https://github.com/smol-rs/smol)
