'use client';

import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';

type HudPanelProps = {
  readonly title?: string;
  readonly titleId?: string;
  readonly glow?: boolean;
  readonly className?: string;
  readonly children: ReactNode;
};

export function HudPanel({
  title,
  titleId,
  glow = false,
  className = '',
  children,
}: HudPanelProps) {
  return (
    <div
      className={cn(
        'relative overflow-hidden',
        'backdrop-blur-sm rounded-xl',
        'border border-white/[0.06] bg-aiki-panel',
        glow &&
          'border-aiki-accent-border shadow-[0_0_30px_rgba(184,115,51,0.12),inset_0_1px_0_rgba(255,255,255,0.05)]',
        className,
      )}
    >
      {/* Inner glow overlay */}
      <div className="pointer-events-none absolute inset-0 rounded-xl bg-gradient-to-b from-white/[0.03] to-transparent" />

      {title && (
        <div className="flex items-center gap-2 px-3 py-1.5 border-b border-white/[0.06] bg-white/[0.02]">
          <h3 id={titleId} className="text-sm font-sans font-semibold text-aiki-text">
            {title}
          </h3>
        </div>
      )}

      <div className="relative flex-1 min-h-0 flex flex-col">{children}</div>
    </div>
  );
}
