'use client';

import { cn } from '@/lib/cn';

type HudProgressBarProps = {
  readonly value: number;
  readonly max?: number;
  readonly className?: string;
};

export function HudProgressBar({ value, max = 100, className = '' }: HudProgressBarProps) {
  const percentage = max > 0 ? Math.min((value / max) * 100, 100) : 0;

  return (
    <div
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={max}
      className={cn('h-1.5 w-full bg-white/[0.06] overflow-hidden rounded-full', className)}
    >
      <div
        className="h-full rounded-full transition-all duration-500 ease-out"
        style={{
          width: `${percentage}%`,
          background: 'var(--color-aiki-accent)',
        }}
      />
    </div>
  );
}
