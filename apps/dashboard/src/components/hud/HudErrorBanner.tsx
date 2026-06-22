'use client';

import { HudButton } from './HudButton';

type HudErrorBannerProps = {
  readonly message: string;
  readonly onRetry?: () => void;
};

export function HudErrorBanner({ message, onRetry }: HudErrorBannerProps) {
  return (
    <div
      role="alert"
      className="flex items-center justify-between gap-2 px-3 py-1.5 bg-aiki-danger/10 border-b border-aiki-danger/20 text-[10px] text-aiki-danger"
    >
      <span className="truncate">{message}</span>
      {onRetry && (
        <HudButton size="sm" variant="ghost" onClick={onRetry}>
          Retry
        </HudButton>
      )}
    </div>
  );
}
