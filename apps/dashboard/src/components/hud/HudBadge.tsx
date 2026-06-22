'use client';

import { cn } from '@/lib/cn';

type HudBadgeProps = {
  readonly status: 'active' | 'queued' | 'completed' | 'failed' | 'paused' | 'pending';
  readonly label?: string;
  readonly pulse?: boolean;
};

const statusColors: Record<HudBadgeProps['status'], string> = {
  active: 'bg-aiki-accent/20 border-aiki-accent/30 text-aiki-accent',
  queued: 'bg-white/[0.04] border-white/[0.08] text-aiki-text-tertiary',
  completed: 'bg-aiki-success/10 border-aiki-success/30 text-aiki-success',
  failed: 'bg-aiki-danger/10 border-aiki-danger/30 text-aiki-danger',
  paused: 'bg-aiki-warning/10 border-aiki-warning/30 text-aiki-warning',
  pending: 'bg-white/[0.04] border-white/[0.08] text-aiki-text-tertiary',
};

export function HudBadge({ status, label, pulse = false }: HudBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-2 py-0.5',
        'text-[9px] font-sans font-medium',
        'border rounded-md',
        statusColors[status],
        pulse && 'animate-pulse-subtle',
      )}
    >
      <span
        className={cn(
          'w-1 h-1 rounded-full bg-current',
          status === 'active' && 'animate-pulse-subtle',
        )}
      />
      {label ?? status}
    </span>
  );
}
