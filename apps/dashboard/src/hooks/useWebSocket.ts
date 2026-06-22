'use client';

import { useEffect, useRef } from 'react';
import { useConnectionStore } from '../stores/connectionStore';
import { useNeuralStore } from '../stores/neuralStore';

function getWsUrl(): string {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8869';
  return apiBase.replace(/^http/, 'ws') + '/ws/events';
}

type WsMessage = {
  type: string;
  [key: string]: unknown;
};

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelayRef = useRef(1000);
  const intentionalCloseRef = useRef(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    intentionalCloseRef.current = false;

    function scheduleReconnect() {
      if (!mountedRef.current || intentionalCloseRef.current) return;
      if (reconnectTimerRef.current) return;

      reconnectTimerRef.current = setTimeout(() => {
        reconnectTimerRef.current = null;
        reconnectDelayRef.current = Math.min(reconnectDelayRef.current * 2, 30000);
        connect();
      }, reconnectDelayRef.current);
    }

    function connect() {
      if (!mountedRef.current || intentionalCloseRef.current) return;

      useConnectionStore.getState().setStatus('connecting');

      let ws: WebSocket;
      try {
        ws = new WebSocket(getWsUrl());
      } catch {
        useConnectionStore.getState().setStatus('disconnected');
        scheduleReconnect();
        return;
      }

      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        reconnectDelayRef.current = 1000;
        useConnectionStore.getState().setStatus('connected');
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const msg: WsMessage = JSON.parse(String(event.data));
          handleMessage(msg);
        } catch {
          // ignore malformed messages
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        wsRef.current = null;
        if (!intentionalCloseRef.current) {
          useConnectionStore.getState().setStatus('reconnecting');
          scheduleReconnect();
        } else {
          useConnectionStore.getState().setStatus('disconnected');
        }
      };

      ws.onerror = () => {
        try {
          ws.close();
        } catch {
          // ignore
        }
      };
    }

    function handleMessage(msg: WsMessage) {
      switch (msg.type) {
        case 'buddy.state': {
          break;
        }
        case 'chat.streaming': {
          // Chat streaming active/inactive
          break;
        }
        case 'chat.message_updated': {
          // A placeholder was promoted to final content — dispatch a DOM event
          // so the chat page can update the message in-place without polling.
          if (typeof window !== 'undefined') {
            window.dispatchEvent(
              new CustomEvent('aikioku:message_updated', { detail: msg }),
            );
          }
          break;
        }
        case 'note.updated': {
          // Note created/updated — invalidate notes cache
          break;
        }
        case 'note.deleted': {
          // Note deleted — invalidate notes cache
          break;
        }
        case 'review.due_changed': {
          // Review due count changed
          break;
        }
        case 'memory.extracted': {
          // New memories extracted
          break;
        }
        case 'stats.updated': {
          // System stats changed
          break;
        }
        case 'goal.progress': {
          const level = 0.3 + Math.random() * 0.5; // 0.3–0.8
          useNeuralStore.getState().setActivityLevel(level);
          useNeuralStore.getState().fireEvent('processing');
          break;
        }
        case 'goal.completed': {
          useNeuralStore.getState().fireEvent('success');
          useNeuralStore.getState().setActivityLevel(0);
          setTimeout(() => {
            if (mountedRef.current) {
              useNeuralStore.getState().fireEvent('idle');
            }
          }, 2000);
          break;
        }
        case 'goal.error': {
          useNeuralStore.getState().fireEvent('error');
          useNeuralStore.getState().setActivityLevel(0);
          setTimeout(() => {
            if (mountedRef.current) {
              useNeuralStore.getState().fireEvent('idle');
            }
          }, 2000);
          break;
        }
        default:
          break;
      }
    }

    connect();

    return () => {
      mountedRef.current = false;
      intentionalCloseRef.current = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      try {
        wsRef.current?.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
    };
  }, []);

  return {
    send: (msg: Record<string, unknown>) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(msg));
      }
    },
    disconnect: () => {
      intentionalCloseRef.current = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      try {
        wsRef.current?.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
      useConnectionStore.getState().setStatus('disconnected');
    },
  };
}
