'use client';

import type { ButtonHTMLAttributes, ReactNode } from 'react';

import { cn } from '@/lib/cn';

type HudButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  readonly variant?: 'default' | 'danger' | 'ghost';
  readonly size?: 'sm' | 'md';
  readonly children: ReactNode;
};

const variants = {
  default:
    'border-aiki-accent-border bg-aiki-accent-bg text-aiki-accent hover:bg-aiki-accent/20 hover:border-aiki-accent/40',
  danger:
    'border-aiki-danger/30 text-aiki-danger bg-white/[0.03] hover:bg-aiki-danger/15 hover:border-aiki-danger/50',
  ghost:
    'border-transparent text-aiki-text-tertiary hover:text-aiki-text-secondary hover:bg-white/[0.04]',
};

const sizes = {
  sm: 'px-2 py-0.5 text-[10px]',
  md: 'px-3 py-1 text-xs',
};

export function HudButton({
  variant = 'default',
  size = 'md',
  className = '',
  children,
  ...props
}: HudButtonProps) {
  return (
    <button
      type="button"
      className={cn(
        'inline-flex items-center justify-center gap-1.5',
        'font-sans font-medium',
        'border rounded-lg',
        'transition-all duration-300',
        'disabled:opacity-30 disabled:pointer-events-none',
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}
