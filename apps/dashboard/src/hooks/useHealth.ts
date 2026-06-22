'use client';

import { useEffect, useRef } from 'react';
import { getToken } from '../lib/api';
import { useSystemStore, type SystemHealth, type SystemStatus } from '../stores/systemStore';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8869';
const POLL_INTERVAL = 30_000;

function authHeaders(): Record<string, string> {
  const token = getToken();
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}

async function fetchHealth(): Promise<SystemHealth | null> {
  try {
    const res = await fetch(`${API_BASE}/health`, { headers: authHeaders() });
    if (!res.ok) return null;
    const data = (await res.json()) as Record<string, unknown>;

    if (data && typeof data === 'object') {
      const memory = (data as { memory?: { heapUsed?: number; heapTotal?: number } }).memory;
      if (memory && typeof memory.heapUsed === 'number' && typeof memory.heapTotal === 'number') {
        return { memory: { heapUsed: memory.heapUsed, heapTotal: memory.heapTotal } };
      }
    }
    return null;
  } catch {
    return null;
  }
}

async function fetchStats(): Promise<SystemStatus | null> {
  try {
    const res = await fetch(`${API_BASE}/api/stats/`, { headers: authHeaders() });
    if (!res.ok) return null;
    const data = (await res.json()) as Record<string, unknown>;

    if (data && typeof data === 'object') {
      return {
        uptime: 0,
        version:
          typeof (data as { version?: string }).version === 'string'
            ? (data as { version: string }).version
            : '—',
        goalsActiveCount:
          typeof (data as { cards?: number }).cards === 'number'
            ? (data as { cards: number }).cards
            : 0,
      };
    }
    return null;
  } catch {
    return null;
  }
}

export function useHealth(enabled = true) {
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    if (!enabled) return;

    const poll = async () => {
      if (!mountedRef.current) return;
      const [health, status] = await Promise.all([fetchHealth(), fetchStats()]);
      if (!mountedRef.current) return;
      if (health) useSystemStore.getState().setHealth(health);
      if (status) useSystemStore.getState().setStatus(status);
    };

    poll();
    const id = setInterval(poll, POLL_INTERVAL);

    const handleVisibility = () => {
      if (document.visibilityState === 'visible') {
        poll();
      }
    };
    document.addEventListener('visibilitychange', handleVisibility);

    return () => {
      mountedRef.current = false;
      clearInterval(id);
      document.removeEventListener('visibilitychange', handleVisibility);
    };
  }, [enabled]);

  return {
    refetch: async () => {
      const [health, status] = await Promise.all([fetchHealth(), fetchStats()]);
      if (health) useSystemStore.getState().setHealth(health);
      if (status) useSystemStore.getState().setStatus(status);
    },
  };
}
