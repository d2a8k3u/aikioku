'use client';

import { createContext, useContext, useEffect, useState } from 'react';
import { reembedApi, getToken } from '@/lib/api';
import { connectEvents, type WsEvent } from '@/lib/reembed-ws';
import type { ReembedStatus } from '@/types';

const DEFAULT: ReembedStatus = {
  state: 'idle',
  target_fp: null,
  processed_notes: 0,
  total_notes: 0,
  processed_convs: 0,
  total_convs: 0,
  error: null,
};

const ReembedContext = createContext<ReembedStatus>(DEFAULT);

export function useReembedStatus(): ReembedStatus {
  return useContext(ReembedContext);
}

function reduce(prev: ReembedStatus, e: WsEvent): ReembedStatus {
  const d = (e.data ?? {}) as Partial<ReembedStatus> & { target_fp?: string };
  switch (e.type) {
    case 'reembed.started':
      return {
        ...prev,
        state: 'running',
        target_fp: d.target_fp ?? prev.target_fp,
        processed_notes: 0,
        processed_convs: 0,
        total_notes: d.total_notes ?? 0,
        total_convs: d.total_convs ?? 0,
        error: null,
      };
    case 'reembed.progress':
      return {
        ...prev,
        state: 'running',
        target_fp: d.target_fp ?? prev.target_fp,
        processed_notes: d.processed_notes ?? prev.processed_notes,
        total_notes: d.total_notes ?? prev.total_notes,
        processed_convs: d.processed_convs ?? prev.processed_convs,
        total_convs: d.total_convs ?? prev.total_convs,
      };
    case 'reembed.complete':
      return { ...prev, state: 'idle', error: null };
    case 'reembed.failed':
      return { ...prev, state: 'failed', error: d.error ?? 'reembed failed' };
    default:
      return prev;
  }
}

export default function ReembedProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<ReembedStatus>(DEFAULT);

  useEffect(() => {
    if (typeof window === 'undefined' || !getToken()) return;
    let cancelled = false;
    reembedApi
      .status()
      .then((s) => {
        if (!cancelled) setStatus(s);
      })
      .catch(() => {
        /* status endpoint unreachable — banner just stays idle */
      });
    const disconnect = connectEvents((e) => {
      if (!e.type.startsWith('reembed.')) return;
      setStatus((prev) => reduce(prev, e));
    });
    return () => {
      cancelled = true;
      disconnect();
    };
  }, []);

  return <ReembedContext.Provider value={status}>{children}</ReembedContext.Provider>;
}
