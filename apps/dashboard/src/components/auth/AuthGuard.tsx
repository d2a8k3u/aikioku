'use client';

import type { ReactNode } from 'react';
import { useAuth } from '../../hooks/useAuth';
import { HudSpinner } from '../hud/HudSpinner';
import { LockScreen } from './LockScreen';

export function AuthGuard({ children }: { readonly children: ReactNode }) {
  const { unlocked, hydrated, hasPassword } = useAuth();

  if (!hydrated) {
    return (
      <div className="flex h-screen items-center justify-center bg-aiki-bg">
        <HudSpinner size={32} />
      </div>
    );
  }

  // No auth wall configured → nothing to lock against.
  if (hasPassword === false) return <>{children}</>;
  if (!unlocked) return <LockScreen />;
  return <>{children}</>;
}
