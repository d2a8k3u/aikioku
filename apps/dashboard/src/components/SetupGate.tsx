'use client';

import { useEffect, useState } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { setupApi, getToken } from '@/lib/api';
import { HudSpinner } from '@/components/hud';

type Status = { configured: boolean; auth_required: boolean };

function BrainIcon() {
  return (
    <svg
      aria-hidden="true"
      width="32"
      height="32"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-aiki-accent"
    >
      <path d="M12 4.5a2.5 2.5 0 0 0-4.96-.46 2.5 2.5 0 0 0-1.98 3.46 2.5 2.5 0 0 0-1.32 4.24 3 3 0 0 0 .34 5.58 2.5 2.5 0 0 0 2.96 3.08 2.5 2.5 0 0 0 4.91.05L12 20V4.5Z" />
      <path d="M16 8V5c0-1.1.9-2 2-2" />
      <path d="M12 4h4a2 2 0 0 1 2 2v2M12 12h4a2 2 0 0 1 2 2v2M12 20h4a2 2 0 0 0 2-2v-2" />
    </svg>
  );
}

function Splash({ message }: { message?: string }) {
  return (
    <div className="flex h-screen flex-col items-center justify-center gap-4 bg-aiki-bg text-aiki-text-secondary">
      <BrainIcon />
      <HudSpinner size={24} />
      {message && <p className="text-sm text-aiki-text-tertiary">{message}</p>}
    </div>
  );
}

/**
 * First-run + auth gate. Checks /api/setup/status on mount and redirects:
 *  - not configured        → /setup (everywhere except /setup)
 *  - configured + on /setup → /
 *  - configured + auth wall + no token → /login (except /login, /setup)
 * Renders a splash while deciding so protected content never flashes.
 */
export default function SetupGate({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setupApi
      .status()
      .then((s) => {
        if (!cancelled) setStatus(s);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!status) return;
    const hasToken = !!getToken();
    if (!status.configured) {
      if (pathname !== '/setup') router.replace('/setup');
      return;
    }
    if (pathname === '/setup') {
      router.replace('/');
      return;
    }
    if (status.auth_required && !hasToken && pathname !== '/login') {
      router.replace('/login');
    }
  }, [status, pathname, router]);

  if (error) {
    return <Splash message={`Cannot reach the backend: ${error}`} />;
  }
  if (!status) {
    return <Splash />;
  }

  // Hold the splash during a pending redirect so guarded UI never flashes.
  const hasToken = !!getToken();
  const redirecting =
    (!status.configured && pathname !== '/setup') ||
    (status.configured && pathname === '/setup') ||
    (status.configured && status.auth_required && !hasToken && pathname !== '/login');
  if (redirecting) {
    return <Splash />;
  }

  return <>{children}</>;
}
