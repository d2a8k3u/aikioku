'use client';

import { cn } from '@/lib/cn';
import { useAuth } from '@/hooks/useAuth';

function LockIcon() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      width="19"
      height="19"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
    >
      <rect x="5" y="11" width="14" height="9" rx="2" />
      <path d="M8 11V8a4 4 0 0 1 8 0v3" />
    </svg>
  );
}

type LockButtonProps = {
  readonly variant?: 'rail' | 'expanded';
  readonly onAction?: () => void;
};

export function LockButton({ variant = 'rail', onAction }: LockButtonProps) {
  const { lock } = useAuth();
  const expanded = variant === 'expanded';

  const handleClick = () => {
    onAction?.();
    lock();
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      aria-label="Lock"
      className={cn(
        'group relative flex w-full items-center rounded-xl py-2.5 text-aiki-text-tertiary transition-colors hover:bg-white/[0.04]',
        expanded
          ? 'justify-start gap-3 px-3'
          : 'mx-2 justify-center px-0 lg:mx-0 lg:justify-start lg:gap-3 lg:px-3',
      )}
    >
      <LockIcon />
      <span
        className={cn(
          'font-mono font-medium uppercase tracking-[0.12em]',
          expanded ? 'text-[11px]' : 'sr-only lg:not-sr-only lg:text-[11px]',
        )}
      >
        Lock
      </span>
      {!expanded && (
        <span
          aria-hidden="true"
          className="pointer-events-none absolute left-full top-1/2 z-20 ml-2 -translate-y-1/2 whitespace-nowrap rounded-md border border-aiki-border-subtle bg-aiki-elevated px-2 py-1 font-mono text-[10px] uppercase tracking-[0.12em] text-aiki-text-secondary opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 lg:hidden"
        >
          Lock
        </span>
      )}
    </button>
  );
}
