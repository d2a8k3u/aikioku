'use client';

import { type ReactNode, useCallback, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';

import { cn } from '@/lib/cn';

type HudDrawerProps = {
  readonly open: boolean;
  readonly onClose: () => void;
  readonly side?: 'left' | 'right';
  readonly labelledById?: string;
  readonly ariaLabel?: string;
  readonly className?: string;
  readonly children: ReactNode;
};

const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

export function HudDrawer({
  open,
  onClose,
  side = 'left',
  labelledById,
  ariaLabel,
  className,
  children,
}: HudDrawerProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  const trapFocus = useCallback((e: KeyboardEvent) => {
    if (e.key !== 'Tab' || !dialogRef.current) return;

    const focusable = dialogRef.current.querySelectorAll<HTMLElement>(FOCUSABLE);
    if (focusable.length === 0) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }, []);

  useEffect(() => {
    if (!open) return;

    previousFocusRef.current = document.activeElement as HTMLElement;

    // Inert the app shell (the drawer is portaled to body as a sibling, so it stays interactive).
    const shell = document.getElementById('app-shell');
    shell?.setAttribute('inert', '');

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    const raf = requestAnimationFrame(() => {
      dialogRef.current?.querySelector<HTMLElement>(FOCUSABLE)?.focus();
    });

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCloseRef.current();
      trapFocus(e);
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      cancelAnimationFrame(raf);
      shell?.removeAttribute('inert');
      document.body.style.overflow = previousOverflow;
      previousFocusRef.current?.focus();
    };
  }, [open, trapFocus]);

  if (!open || typeof document === 'undefined') return null;

  return createPortal(
    <div className="fixed inset-0 z-[60]">
      <button
        type="button"
        tabIndex={-1}
        aria-label="Close navigation menu"
        className="absolute inset-0 cursor-default border-none bg-aiki-overlay animate-fade-in"
        onClick={onClose}
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledById}
        aria-label={ariaLabel}
        className={cn(
          'absolute inset-y-0 flex w-72 flex-col',
          side === 'left' ? 'left-0 animate-drawer-in-left' : 'right-0',
          className,
        )}
      >
        {children}
      </div>
    </div>,
    document.body,
  );
}
