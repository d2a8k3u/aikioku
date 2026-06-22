// Minimal reconnecting WebSocket client for backend EventBus events (/ws/events).
//
// The socket is unauthenticated (Starlette HTTP auth middleware does not see the
// WebSocket scope), so no bearer token is attached. Consumers filter by event
// `type`. Used by the reembed banner to react to background reembed progress.

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8869';

export type WsEvent = { id?: string; type: string; data: unknown; created?: string };

export function connectEvents(onEvent: (e: WsEvent) => void): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  let keepalive: ReturnType<typeof setInterval> | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let backoff = 1000;

  const url = API_BASE.replace(/^http/, 'ws') + '/ws/events';

  const clearKeepalive = () => {
    if (keepalive) {
      clearInterval(keepalive);
      keepalive = null;
    }
  };

  const scheduleReconnect = () => {
    if (closed) return;
    reconnectTimer = setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 30000);
  };

  function connect() {
    if (closed) return;
    try {
      ws = new WebSocket(url);
    } catch {
      scheduleReconnect();
      return;
    }
    ws.onopen = () => {
      backoff = 1000;
      keepalive = setInterval(() => {
        try {
          ws?.send('ping');
        } catch {
          /* ignore */
        }
      }, 25000);
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as WsEvent;
        if (msg && typeof msg.type === 'string') onEvent(msg);
      } catch {
        /* ignore malformed frame */
      }
    };
    ws.onclose = () => {
      clearKeepalive();
      scheduleReconnect();
    };
    ws.onerror = () => {
      try {
        ws?.close();
      } catch {
        /* ignore */
      }
    };
  }

  connect();

  return () => {
    closed = true;
    clearKeepalive();
    if (reconnectTimer) clearTimeout(reconnectTimer);
    try {
      ws?.close();
    } catch {
      /* ignore */
    }
  };
}
