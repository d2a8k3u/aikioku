'use client';

import { useCallback, useEffect, useRef } from 'react';
import { authApi, setupApi } from '../lib/api';
import { AUTO_LOCK_TIMEOUT } from '../lib/constants';
import { useAuthStore } from '../stores/authStore';

export function useAuth() {
  const {
    unlocked,
    hasPassword,
    hydrated,
    lock,
    unlock,
    setHasPassword,
    setHydrated,
    touchActivity,
  } = useAuthStore();

  // Hydrate: check server for auth status + restore session
  useEffect(() => {
    if (hydrated) return;
    let cancelled = false;

    (async () => {
      try {
        const { auth_required } = await setupApi.status();
        if (cancelled) return;
        setHasPassword(auth_required);

        // Restore session if sessionStorage says we were unlocked
        if (
          auth_required &&
          typeof window !== 'undefined' &&
          sessionStorage.getItem('aikioku_dashboard_session') === 'true'
        ) {
          unlock();
        }
      } catch {
        // Server unreachable — show login anyway, will fail on submit
        if (!cancelled) setHasPassword(false);
      }
      if (!cancelled) setHydrated();
    })();

    return () => {
      cancelled = true;
    };
  }, [hydrated, setHasPassword, setHydrated, unlock]);

  const setupPassword = useCallback(
    async (username: string, password: string) => {
      await authApi.register(username, password);
      setHasPassword(true);
      unlock();
    },
    [setHasPassword, unlock],
  );

  const login = useCallback(
    async (username: string, password: string): Promise<boolean> => {
      try {
        await authApi.login(username, password);
        unlock();
        return true;
      } catch {
        return false;
      }
    },
    [unlock],
  );

  // Auto-lock after inactivity
  useEffect(() => {
    if (!unlocked) return;
    const check = setInterval(() => {
      if (Date.now() - useAuthStore.getState().lastActivity > AUTO_LOCK_TIMEOUT) {
        lock();
      }
    }, 60_000);
    return () => clearInterval(check);
  }, [unlocked, lock]);

  // Track activity (mousemove throttled to once per 30s)
  const lastTouchRef = useRef(0);
  useEffect(() => {
    if (!unlocked) return;
    const throttledHandler = () => {
      const now = Date.now();
      if (now - lastTouchRef.current < 30_000) return;
      lastTouchRef.current = now;
      touchActivity();
    };
    const keyHandler = () => touchActivity();
    window.addEventListener('mousemove', throttledHandler, { passive: true });
    window.addEventListener('keydown', keyHandler, { passive: true });
    return () => {
      window.removeEventListener('mousemove', throttledHandler);
      window.removeEventListener('keydown', keyHandler);
    };
  }, [unlocked, touchActivity]);

  return { unlocked, hasPassword, hydrated, setupPassword, login, lock };
}
