'use client';

import Link from 'next/link';

import { cn } from '@/lib/cn';
import { useConnectionStore } from '@/stores/connectionStore';

type LogoProps = {
  readonly wordmark?: 'always' | 'responsive';
  readonly wordmarkId?: string;
  readonly onNavigate?: () => void;
};

export function Logo({ wordmark = 'responsive', wordmarkId, onNavigate }: LogoProps) {
  return (
    <Link
      href="/"
      onClick={onNavigate}
      aria-label="KIO home"
      className={cn(
        'flex items-center gap-2.5 no-underline',
        wordmark === 'responsive' && 'justify-center lg:justify-start',
      )}
    >
      <span className="relative flex h-[30px] w-[30px] items-center justify-center">
        <span
          aria-hidden="true"
          className="absolute h-[30px] w-[30px] rounded-full"
          style={{
            background:
              'radial-gradient(circle at 38% 32%, rgba(184,115,51,0.7), rgba(52,214,196,0.25) 70%, transparent)',
            filter: 'blur(1px)',
            animation: 'auraPulse 4s ease-in-out infinite',
          }}
        />
        <span
          aria-hidden="true"
          className="h-[13px] w-[13px] rounded-full"
          style={{
            background: 'radial-gradient(circle at 36% 32%, #ffe9cf, #B87333 55%, #2aa8b8)',
            boxShadow: '0 0 10px rgba(184,115,51,0.7)',
          }}
        />
      </span>
      <span
        id={wordmarkId}
        className={cn(
          'font-mono text-sm font-bold tracking-[0.22em] text-aiki-text-secondary',
          wordmark === 'responsive' && 'hidden lg:inline',
        )}
      >
        KIO
      </span>
    </Link>
  );
}

type StatusDotProps = {
  readonly label?: 'always' | 'responsive';
};

export function StatusDot({ label = 'always' }: StatusDotProps) {
  const status = useConnectionStore((s) => s.status);
  const online = status === 'connected';

  return (
    <div role="status" aria-live="polite" className="flex items-center gap-1.5">
      <span
        aria-hidden="true"
        className={cn('h-1.5 w-1.5 rounded-full', online ? 'bg-aiki-success' : 'bg-aiki-danger')}
        style={online ? { animation: 'pulseDot 2.6s ease-in-out infinite' } : undefined}
      />
      <span
        className={cn(
          'font-mono text-[10px] tracking-[0.16em]',
          label === 'responsive' && 'sr-only lg:not-sr-only',
          online ? 'text-aiki-success' : 'text-aiki-danger',
        )}
      >
        {online ? 'ONLINE' : 'OFFLINE'}
      </span>
    </div>
  );
}
