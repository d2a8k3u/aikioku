'use client';

import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';

export type SegmentedOption<T extends string> = {
  readonly value: T;
  readonly label: string;
  readonly icon?: ReactNode;
};

type HudSegmentedProps<T extends string> = {
  readonly options: ReadonlyArray<SegmentedOption<T>>;
  readonly value: T;
  readonly onChange: (value: T) => void;
  readonly ariaLabel: string;
  readonly className?: string;
  // Applied to each option's text label — e.g. "hidden @lg:inline" to collapse to icon-only.
  readonly labelClassName?: string;
};

export function HudSegmented<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  className,
  labelClassName,
}: HudSegmentedProps<T>) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className={cn(
        'inline-flex rounded-lg border border-aiki-border bg-white/[0.02] p-0.5',
        className,
      )}
    >
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            aria-pressed={active}
            aria-label={opt.label}
            onClick={() => onChange(opt.value)}
            className={cn(
              'inline-flex items-center gap-1.5 rounded-md px-3 py-1',
              'font-mono text-[11px] font-medium tracking-wide',
              'transition-all duration-200',
              active
                ? 'bg-aiki-accent-bg text-aiki-accent'
                : 'text-aiki-text-tertiary hover:text-aiki-text-secondary',
            )}
          >
            {opt.icon}
            <span className={labelClassName}>{opt.label}</span>
          </button>
        );
      })}
    </div>
  );
}
