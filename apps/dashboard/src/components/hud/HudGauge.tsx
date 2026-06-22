'use client';

import { useId } from 'react';

type HudGaugeProps = {
  readonly value: number;
  readonly max?: number;
  readonly label: string;
  readonly unit?: string;
  readonly size?: number;
  readonly thresholds?: { warn: number; danger: number };
};

export function HudGauge({
  value: rawValue,
  max = 100,
  label,
  unit = '%',
  size = 80,
  thresholds,
}: HudGaugeProps) {
  const value = Number.isFinite(rawValue) ? rawValue : 0;
  const radius = (size - 12) / 2;
  const circumference = 2 * Math.PI * radius;
  const safeMax = max || 1;
  const percentage = Math.min(value / safeMax, 1);
  const offset = circumference * (1 - percentage);
  const uid = useId();
  const gradientId = `gauge-grad-${uid}`;

  const getColor = (): string => {
    if (!thresholds) return 'var(--color-aiki-accent)';
    if (value >= thresholds.danger) return 'var(--color-aiki-danger)';
    if (value >= thresholds.warn) return 'var(--color-aiki-warning)';
    return 'var(--color-aiki-success)';
  };

  const color = getColor();

  return (
    <div className="flex flex-col items-center gap-1">
      <meter value={value} min={0} max={max} aria-label={label} className="relative sr-only" />
      <div className="relative" style={{ width: size, height: size }}>
        <svg aria-hidden="true" width={size} height={size} className="absolute inset-0 -rotate-90">
          <defs>
            <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={color} />
              <stop offset="100%" stopColor={color} stopOpacity="0.6" />
            </linearGradient>
          </defs>
          {/* Background ring */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255, 255, 255, 0.06)"
            strokeWidth={4}
          />
          {/* Value ring */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={`url(#${gradientId})`}
            strokeWidth={4}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-sans text-sm font-semibold text-aiki-text">
            {value % 1 === 0 ? value : value.toFixed(1)}
          </span>
          {unit && <span className="text-[9px] text-aiki-text-tertiary">{unit}</span>}
        </div>
      </div>
      <span className="text-[9px] font-sans font-medium text-aiki-text-tertiary">{label}</span>
    </div>
  );
}
