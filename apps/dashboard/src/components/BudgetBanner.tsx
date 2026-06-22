'use client';

import { useEffect, useRef } from 'react';
import { useBudgetStatus } from './BudgetProvider';
import { useToast } from './hud/HudToast';

function usd(n: number): string {
  return `$${(Number.isFinite(n) ? n : 0).toFixed(2)}`;
}

function resetLabel(iso: string): string {
  if (!iso) return '00:00 UTC';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '00:00 UTC';
  return d.toLocaleString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    timeZoneName: 'short',
  });
}

export default function BudgetBanner() {
  const status = useBudgetStatus();
  const { addToast } = useToast();
  const prevState = useRef(status.state);

  // One toast on the active/warning → paused transition.
  useEffect(() => {
    if (prevState.current !== 'paused' && status.state === 'paused') {
      addToast('Daily LLM budget reached — processing paused', 'warning');
    }
    prevState.current = status.state;
  }, [status.state, addToast]);

  if (status.state === 'paused') {
    const queued = status.pending_count;
    return (
      <div className="flex items-center gap-2 border-b border-aiki-danger/30 bg-aiki-danger/10 px-4 py-2 text-sm text-aiki-danger">
        <svg
          aria-hidden="true"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="flex-shrink-0"
        >
          <rect x="6" y="4" width="4" height="16" />
          <rect x="14" y="4" width="4" height="16" />
        </svg>
        <span className="flex-1">
          Daily LLM budget reached ({usd(status.today_cost)} / {usd(status.daily_budget)}) —
          processing paused.
          {queued > 0 && ` ${queued} item${queued === 1 ? '' : 's'} queued.`} Resumes at{' '}
          {resetLabel(status.reset_at)}. New notes &amp; memories are still saved.
        </span>
      </div>
    );
  }

  if (status.state === 'warning') {
    const pct = Math.round(status.fraction * 100);
    return (
      <div className="flex items-center gap-2 border-b border-aiki-accent/30 bg-aiki-accent/10 px-4 py-2 text-sm text-aiki-accent">
        <svg
          aria-hidden="true"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="flex-shrink-0"
        >
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
          <line x1="12" y1="9" x2="12" y2="13" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
        <span>
          LLM budget {pct}% used ({usd(status.today_cost)} / {usd(status.daily_budget)} today) —
          processing pauses at the limit.
        </span>
      </div>
    );
  }

  return null;
}
