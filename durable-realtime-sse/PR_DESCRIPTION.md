# Durable realtime chat over one SSE gateway

## Summary

This change makes the realtime event journal the durable source of truth for live chat state. Chat history remains normal persisted history, while queued/running turns and mid-turn telemetry are stored separately and delivered through one ordered SSE gateway.

As a result, refreshing during a long turn no longer loses the visible Thinking state, reconnects can replay the exact gap, and every tab viewing the same session receives new messages without waiting for a content poll.

## Why

The previous flow mixed short-lived in-memory buffers, polling, and several SSE paths. It worked while one page stayed connected, but recovery became unreliable when a request was still running during a refresh or network interruption. A second tab also had no authoritative live message feed, so it could remain stale until a refresh.

The new split is simpler:

- chat history stores durable user and assistant messages;
- `realtime_events` stores ordered browser-facing telemetry;
- `active_turns` tracks queued and running work;
- `/api/realtime/stream` is the single replay and live-delivery gateway.

## Architecture

![Durable realtime architecture](https://raw.githubusercontent.com/binsarjr/evonic/durable-realtime-sse-assets/durable-realtime-sse/architecture-overview.png)

The runtime still emits one internal event stream for plugins. Before listeners run asynchronously, browser-facing events are normalized and appended to SQLite with a global sequence and timestamp. The browser never receives backend paths or browser-local attachment URLs.

Chat history and live telemetry have separate jobs. History provides stable messages and a cursor; the realtime journal provides the active turn, ordered progress, and replay after a connection gap.

## Live message and cross-tab flow

![Cross-tab message flow](https://raw.githubusercontent.com/binsarjr/evonic/durable-realtime-sse-assets/durable-realtime-sse/cross-tab-message-flow.png)

`client_message_id` reconciles the sender's optimistic bubble with the durable `message_received` event. The sender keeps its existing bubble while every other tab renders the same message immediately. `message_id` prevents the final response from being rendered twice when history, the POST response, and SSE overlap.

The same stream carries `turn_queued`, `turn_begin`, Thinking updates, tool progress, response chunks, and `done`, so every tab follows the same turn in real time.

## Refresh and reconnect

![Refresh and reconnect flow](https://raw.githubusercontent.com/binsarjr/evonic/durable-realtime-sse-assets/durable-realtime-sse/refresh-reconnect-flow.png)

A fresh page first renders stable history, then starts SSE from the cursor returned with that history. If a turn is still active, the stream rebuilds its current Thinking state before continuing with live events.

A transient disconnect is different: native `EventSource` reconnects with `Last-Event-ID`, and the gateway sends only the missing journal rows. Neither path needs content polling or a second gap-fill endpoint.

## Turn lifecycle and retention

![Durable turn lifecycle](https://raw.githubusercontent.com/binsarjr/evonic/durable-realtime-sse-assets/durable-realtime-sse/turn-lifecycle.png)

Events belonging to an active turn do not expire. Once the turn finishes, its journal remains replayable for 24 hours. On startup, turns owned by an old process are closed as interrupted instead of remaining permanently busy. Clearing a session removes both its active-turn projection and its realtime journal rows.

## Interface changes

- `GET /api/realtime/stream` multiplexes `chat`, `status`, `approvals`, `workplace`, and `update` events.
- Chat streams are scoped by `agent_id` and `session_id`, with replay starting from `after` or `Last-Event-ID`.
- Chat history responses expose `X-Evonic-Realtime-Cursor` so rendering history and opening SSE have a deterministic handoff.
- Message sends accept `client_message_id`; durable events include `message_id` for cross-path deduplication.
- Busy state is derived from durable queued/running turns instead of a separate in-memory agent tracker.

## Result

- Mid-turn Thinking state survives refresh and reconnect.
- New user messages appear immediately in every tab viewing the same session.
- Runtime events stay ordered by one global sequence and timestamp.
- Browser delivery no longer depends on content polling or process-local replay buffers.
- Existing plugin listeners continue to receive the internal EventStream.

## Testing

```text
./venv/bin/pytest -q \
  unit_tests/test_chat_buffer_replay.py \
  unit_tests/test_frontend_sse_lifecycle.py \
  unit_tests/test_state_changed_sse.py

17 passed
```

This covers durable ordering and scoping, active-turn recovery, history-cursor handoff, `Last-Event-ID` reconnect behavior, cross-tab delivery, optimistic deduplication, busy-state transitions, and session cleanup.
