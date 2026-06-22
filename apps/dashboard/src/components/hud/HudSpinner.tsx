'use client';

type HudSpinnerProps = {
  readonly size?: number;
};

export function HudSpinner({ size = 20 }: HudSpinnerProps) {
  return (
    <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" className="animate-spin">
      <circle
        cx="12"
        cy="12"
        r="10"
        fill="none"
        stroke="rgba(255, 255, 255, 0.06)"
        strokeWidth="2"
      />
      <path
        d="M12 2a10 10 0 0 1 10 10"
        fill="none"
        stroke="var(--color-aiki-accent)"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}
