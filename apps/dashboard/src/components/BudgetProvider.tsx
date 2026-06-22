'use client';

import { createContext, useContext, useEffect, useState } from 'react';
import { budgetApi } from '@/lib/api';
import { connectEvents } from '@/lib/reembed-ws';
import type { BudgetStatus } from '@/types';

const DEFAULT: BudgetStatus = {
  state: 'active',
  daily_budget: 0,
  today_cost: 0,
  remaining: 0,
  fraction: 0,
  pending_count: 0,
  warning_fraction: 0.9,
  reset_at: '',
};

const BudgetContext = createContext<BudgetStatus>(DEFAULT);

export function useBudgetStatus(): BudgetStatus {
  return useContext(BudgetContext);
}

export default function BudgetProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<BudgetStatus>(DEFAULT);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    let cancelled = false;
    budgetApi
      .status()
      .then((s) => {
        if (!cancelled) setStatus(s);
      })
      .catch(() => {
        /* status endpoint unreachable — banner just stays idle */
      });
    // The backend emits a single `budget.status` event carrying the full
    // snapshot, so a transition just replaces the state.
    const disconnect = connectEvents((e) => {
      if (e.type !== 'budget.status') return;
      setStatus((prev) => ({ ...prev, ...(e.data as Partial<BudgetStatus>) }));
    });
    return () => {
      cancelled = true;
      disconnect();
    };
  }, []);

  return <BudgetContext.Provider value={status}>{children}</BudgetContext.Provider>;
}
