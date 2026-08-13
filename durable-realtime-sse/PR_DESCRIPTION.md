# Durable realtime chat across refresh, reconnect, and tabs

This is intentionally a larger-than-usual change, but it solves one problem end to end: Evonic's live chat state currently has no durable source of truth. A refresh during a long turn can make active Thinking or tool progress disappear from the UI, and a message sent in one tab may not appear in another until the page is refreshed.

The diff crosses runtime, storage, SSE, and the browser because those layers currently own different parts of the same turn. Fixing only one layer would leave the race in another.

## Summary

This change gives browser-facing realtime state a durable, ordered source of truth. Chat history still stores the conversation itself, while queued/running turns and mid-turn telemetry live in the dedicated `shared/db/realtime.db` journal and reach every open tab through one SSE gateway.

The goal is not simply to add SSE. The current flow already has SSE and a unified endpoint. The problem is that history, live telemetry, busy state, reconnect recovery, and cross-tab rendering still depend on different sources with different lifetimes.

With this change, a page can load stable history, continue from the exact event cursor captured with that history, rebuild an active turn, and then stay on the same ordered stream.

## How the current flow works

![Upstream realtime flow](https://raw.githubusercontent.com/binsarjr/evonic/durable-realtime-sse-assets/durable-realtime-sse/upstream-realtime-flow.png)

The current design splits chat state across three paths:

- chat history persists user and assistant messages;
- `EventStream` keeps at most 500 telemetry entries per session in process memory and schedules that buffer for deletion 30 seconds after a turn;
- a separate in-memory tracker answers whether an agent is busy.

The browser combines those paths using history requests, content polling, `/chat/events`, `/busy`, and SSE. The unified SSE route reduces the number of browser connections, but its chat replay still comes from the process-local buffer. It unifies transport, not the underlying source of truth.

### What this flow does well

- Live events travel directly from the runtime to connected listeners with very little overhead.
- Transient telemetry does not create a database write for every thinking or tool event.
- The bounded ring buffer keeps memory use predictable.
- A reconnect within the short post-turn buffer window can usually recover if it reaches the same process.
- Chat history stays small because it only contains conversation records.

For one tab that remains connected through a normal, short turn, this is a reasonable and fast design.

## Why the failures are intermittent

![Upstream failure timelines](https://raw.githubusercontent.com/binsarjr/evonic/durable-realtime-sse-assets/durable-realtime-sse/upstream-failure-timelines.png)

### Refresh during a long turn

Mid-turn Thinking and tool activity are not part of stable chat history. After a refresh, the page has to reconstruct the UI by independently reading history, the process-local event buffer, and the busy endpoint.

Those reads do not share one cursor or snapshot, so the turn can change between them. There are also two time-based assumptions: the frontend can auto-finalize a turn after five minutes without an event, while the in-memory busy entry expires after ten minutes. A long silent model call or tool can therefore still be running even though a refreshed page no longer has enough authoritative state to display it.

A process restart is a harder boundary: the session buffer, its sequence counter, and the busy tracker disappear together. Chat history survives, but the live representation does not.

### A message sent in another tab

Tab A renders its own message optimistically, then the runtime saves it and emits `message_received`. The normal agent-detail handler only reloads history for escalated messages. Its idle poll also advances over user entries but filters them from rendering because it assumes they are local echoes.

That assumption is correct for Tab A and wrong for Tab B. Tab B can receive the surrounding realtime activity while still not rendering the new user message; a full history refresh finally makes it visible.

## Why this is one cohesive change

The symptoms appear in the browser, but the missing state begins earlier in the lifecycle. Queue acceptance, busy state, runtime telemetry, saved messages, reconnect cursors, and rendering all use different ownership and timing rules. A frontend-only patch cannot replay state the server no longer has; persisting events without changing the history handoff still leaves a race; replacing the busy tracker without updating queue transitions can report idle while accepted work is waiting.

This change therefore follows one turn through the complete path:

- message acceptance creates a durable queued turn;
- runtime transitions it to running and emits ordered telemetry;
- history and SSE hand off through one cursor;
- every tab consumes the same event sequence and deduplicates overlapping sources;
- completion, cancellation, failure, or restart produces a terminal state;
- terminal telemetry remains replayable for a bounded period and is then cleaned up.

The breadth is a consequence of closing that lifecycle, not a collection of unrelated features.

## The durable flow

![Durable realtime architecture](https://raw.githubusercontent.com/binsarjr/evonic/durable-realtime-sse-assets/durable-realtime-sse/architecture-overview.png)

Browser-facing events are now normalized and appended to `shared/db/realtime.db` before asynchronous plugin listeners run. Each row receives one global event ID and timestamp. Keeping this journal global, rather than creating one database per agent, gives the gateway one ordering domain for cross-agent status, approvals, agent-to-agent activity, and chat events while isolating the write load from the main application database.

The responsibilities are explicit:

- chat history stores stable user and assistant messages;
- `realtime_events` stores ordered browser-facing telemetry;
- `active_turns` projects queued and running work;
- per-session replay floors record the newest cursor that cleanup has made unavailable;
- `/api/realtime/stream` performs initial replay, reconnect replay, snapshots, and live delivery.

Payloads are sanitized before they enter the journal and capped at 256 KiB. Oversized fields keep a bounded preview and explicit truncation metadata instead of allowing one tool result to grow the database without limit.

This keeps history and telemetry separate without making live state disposable.

## Before and after

| Area | Current flow | Durable flow |
| --- | --- | --- |
| Live delivery | Direct in-process publish/subscribe; minimal write overhead | Append once to SQLite, then deliver through SSE |
| Source of truth | History, event buffer, and busy tracker have separate lifetimes | History, event journal, and active-turn projection have explicit roles |
| Refresh | Rebuild from separate history, `/chat/events`, and `/busy` reads | History returns a cursor; SSE replays exactly after that cursor |
| Long silent turn | Five-minute UI timeout and ten-minute busy TTL can hide ongoing work | A turn remains active until it reaches a real terminal state |
| Cross-tab messages | Sender renders optimistically; another tab can filter the user message as an echo | Every tab renders the durable event; the sender reconciles its optimistic bubble |
| Reconnect | Per-session sequence and replay buffer exist only in the current process | `Last-Event-ID` resumes from the durable global sequence |
| Post-turn replay | Session buffer is scheduled for deletion after 30 seconds | Completed telemetry remains replayable for one hour |
| Cursor older than retained telemetry | No durable way to distinguish an empty replay from missing data | Gateway requests a history resync, then the browser reconnects from a fresh cursor |
| Server restart | Live buffers vanish without a reliable terminal event | Old active turns are closed as interrupted after startup |
| Ordering | History, chat SSE, status SSE, and polling use different cursors | Browser-visible events share one global ID and timestamp |
| Operational cost | Fewer database writes, but recovery logic is spread across endpoints and frontend paths | More SQLite writes and lifecycle logic, but one deterministic browser contract |
| Scaling characteristic | No journal write contention, but no cross-process replay | SQLite WAL is sufficient for the current deployment model; high-volume multi-worker use must monitor contention |

## Live message and cross-tab flow

![Cross-tab message flow](https://raw.githubusercontent.com/binsarjr/evonic/durable-realtime-sse-assets/durable-realtime-sse/cross-tab-message-flow.png)

`client_message_id` correlates the sender's optimistic bubble with the durable `message_received` event. The sender keeps one bubble, while every other tab renders the same message immediately. Stable `message_id` values prevent duplicates when history, the POST response, and SSE overlap.

The same stream carries `turn_queued`, `turn_begin`, Thinking updates, tool progress, response chunks, and `done`, so all tabs follow the same turn rather than independently guessing its state.

## Agent-to-agent delivery

Agent-to-agent calls use the same queued/running/terminal lifecycle. The sender now registers its completion waiter before dispatching the target agent, preventing a fast target from completing before the sender is ready to observe the result. A target can wake the waiting agent even when no human-facing channel route exists, while any browser viewing the session receives the same durable activity through the gateway.

## Refresh and reconnect

![Refresh and reconnect flow](https://raw.githubusercontent.com/binsarjr/evonic/durable-realtime-sse-assets/durable-realtime-sse/refresh-reconnect-flow.png)

A fresh page renders history and captures `X-Evonic-Realtime-Cursor` from that response. It opens SSE from the cursor, replays any active-turn telemetry represented by the journal, closes the history-to-stream gap, receives a current-state snapshot, and continues with live events.

A transient disconnect within the retained window uses native `EventSource` reconnection and `Last-Event-ID`. The gateway returns only the missing journal rows and does not need a second content-polling recovery path.

If a tab reconnects with a cursor older than the available journal, or with an invalid or future cursor, the gateway emits `history_resync_required` and closes that stream. The agent-detail and sessions views reload stable chat history, take its new cursor, and reconnect automatically. An expired cursor therefore becomes an explicit recovery path instead of looking like an empty replay.

## Turn lifecycle and retention

![Durable turn lifecycle](https://raw.githubusercontent.com/binsarjr/evonic/durable-realtime-sse-assets/durable-realtime-sse/turn-lifecycle.png)

Events belonging to an active turn do not expire. Once a turn reaches a terminal state, its journal remains replayable for one hour and is then eligible for cleanup. Events that do not belong to a turn use the same one-hour window from their occurrence. Session clear removes completed realtime history without deleting an active turn that is still needed to finish safely.

On startup, abandoned active-turn records are emitted as interrupted and closed. If a terminal `done` event was already journaled before the process died, recovery closes the stale projection without writing a duplicate terminal event. This restores an honest UI state; it does not claim to resume an LLM call or tool process that died with the server.

## Trade-offs and limits

The durable design intentionally accepts several costs:

- thinking, tool, status, and message events create additional SQLite writes;
- replay, snapshot ordering, cursor validation, retention, and deduplication add backend logic;
- retained telemetry has a data-lifecycle responsibility that an in-memory buffer did not have;
- one global realtime database creates a shared write path, although it also provides the ordering the gateway needs;
- SQLite WAL is not a distributed event broker and may become a bottleneck under substantially higher write concurrency;
- clients still need idempotent rendering because history, an HTTP response, and replay can legitimately overlap.

The implementation keeps that cost bounded: it uses a dedicated SQLite WAL database instead of adding a broker, caps each stored payload at 256 KiB, retains completed telemetry for one hour, preserves active events until termination, tracks replay floors per session, validates cursors, and deduplicates with event and message IDs.

## Compatibility and interface changes

- `GET /api/realtime/stream` multiplexes `chat`, `status`, `approvals`, `workplace`, and `update` events.
- `cursor_version=2` identifies durable global cursors; `snapshot=1` requests initial state.
- Native reconnects use `Last-Event-ID` and suppress duplicate initial snapshots.
- Chat streams emit `history_resync_required` when a versioned cursor is expired, invalid, or ahead of the journal; both chat views recover automatically.
- Chat history exposes `X-Evonic-Realtime-Cursor` for the history-to-stream handoff.
- Message sends accept `client_message_id`; durable events expose stable `message_id` values.
- Busy state comes from durable queued/running turns rather than a separate expiring agent tracker.
- Legacy stream endpoints redirect to the unified gateway, legacy update event names remain available, and the legacy gap reader returns a safe reset contract when its old cursor cannot be mapped.
- Internal `EventStream` plugin listeners continue to receive runtime events.

## Result

- Mid-turn Thinking state survives page refresh and transient reconnects.
- New user messages appear in every tab viewing the same session.
- Long silent work is not hidden by an arbitrary browser timeout.
- Browser events remain globally ordered across history handoff and reconnect.
- Expired cursors reload stable history instead of silently skipping unavailable telemetry.
- Restarted servers report abandoned turns as interrupted instead of leaving a false busy state.
- Chat delivery no longer depends on content polling or process-local replay buffers.
- Agent-to-agent completion cannot outrun waiter registration or depend on a human delivery route.

## Testing

```text
pytest -q \
  unit_tests/test_agent_messaging.py \
  unit_tests/test_chat_sessions.py \
  unit_tests/test_chat_buffer_replay.py \
  unit_tests/test_frontend_sse_lifecycle.py \
  unit_tests/test_sse_connection_limit.py \
  unit_tests/test_state_changed_sse.py \
  unit_tests/test_test_isolation.py

146 passed

pytest -q plugins/kanban/tests

105 passed

cd evonet && go test ./...

ok github.com/evonic/evonet/internal/executor
ok github.com/evonic/evonet/internal/ws
```

The focused suite covers durable ordering and scoping, the dedicated database path, one-hour terminal retention, active-turn replay, monotonic cursors after cleanup, history resync, payload bounds, history-cursor handoff, invalid and future cursors, `Last-Event-ID`, cross-tab delivery, optimistic deduplication, long-turn UI behavior, atomic queued-turn cancellation, agent-to-agent completion races, restart recovery without duplicate terminal events, session cleanup, approval snapshots, cache busting, test isolation, and legacy route compatibility.

The full local unit suite completed with `2158 passed`, `84 skipped`, and five failures. The same five failures reproduce on the clean `dev` base: two artifact-path expectations tied to another checkout path, one environment-dependent local classifier expectation, one Python 3.13 SFTP mock incompatibility, and one pre-existing token-budget assertion. No additional failure appears on this branch.

Manual browser validation covered two tabs on the same session, optimistic message deduplication, queued and split turns, refresh while a turn is active, reconnect replay, agent switching, stop/cancel behavior, and agent-to-agent delivery without a human-facing route.
