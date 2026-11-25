# Security

VisualAgents is a local developer tool. This page says what it protects, and what it deliberately leaves open, so
you can decide where it is safe to run.

## Reporting a problem

If you find a security problem, please report it privately through GitHub's "Report a vulnerability" button on the
Security tab, instead of opening a public issue.

## The dashboard

- It listens on `127.0.0.1` only, and answers only requests addressed to `127.0.0.1`, `localhost` or `[::1]`, so a
  web page that re-points its own domain at your computer (DNS rebinding) cannot read it or press anything.
- **Reading it needs no password.** The page and its read-only API (`/api/state`, `/api/blueprint`,
  `/api/projects`, `/api/stream`) show your agents' live prompts, output and map to any program, or any other user
  account, on the same computer that connects to that port. Don't run it on a shared machine while your agents
  handle data other users of that machine must not see.
- Writing to it is protected. Events from your agent need a token made for that session, and **Start** and
  **Reset** need a second token that is written only into pages served to this computer.
- **`--host`** (anything other than `127.0.0.1`) binds the dashboard to the network: anyone who can reach that
  address can read the live prompts, output and map, with no password, and VisualAgents prints a warning when it
  starts that way. Start and Reset still work only from this computer. Use `--host` only on a network you trust.

## Your code and your keys

- Nothing leaves your computer unless you choose an LLM provider for the map. Then the scan sends part of the
  project's source to that provider. Before sending, it skips `.env` files, key and credential files and files git
  ignores, and replaces strings that look like API keys or private keys with `<redacted>`.
- Scanning someone else's repository cannot send your saved keys elsewhere: a project's own `.env` may not choose
  the provider or the server requests go to, and links that point out of the project are never read.
- Keys saved by the guided start live in `~/.visualagents/keys.json`, readable only by you (file mode 600).
- There is no telemetry.

## The hook

`visualagents hook <folder>` writes one file, `visualagents_hook.pth`, into the `site-packages` folder of each
virtualenv in that folder (or of an interpreter you name with `--python`). Python runs it every time it starts in
that environment; it stays silent unless the process calls an LLM or an agent framework, and then sends its events
to the background dashboard on this computer. `visualagents unhook <folder>` removes the file again.
