'use client';

import { forwardRef, type InputHTMLAttributes } from 'react';

import { cn } from '@/lib/cn';

type HudInputProps = InputHTMLAttributes<HTMLInputElement>;

export const HudInput = forwardRef<HTMLInputElement, HudInputProps>(
  ({ className = '', ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={cn(
          'w-full bg-white/[0.03] backdrop-blur-sm border border-aiki-border rounded-lg px-3 py-2',
          'text-sm font-mono text-aiki-text-secondary',
          'placeholder:text-aiki-text-tertiary/40',
          'focus:border-aiki-accent/50 focus:shadow-[0_0_16px_rgba(184,115,51,0.1)] focus:bg-white/[0.05]',
          'transition-all duration-300 outline-none',
          className,
        )}
        {...props}
      />
    );
  },
);

HudInput.displayName = 'HudInput';
