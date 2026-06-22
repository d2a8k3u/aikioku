'use client';

import { type ReactNode, useCallback, useEffect, useId, useRef } from 'react';
import { createPortal } from 'react-dom';

import { HudPanel } from './HudPanel';

type HudModalProps = {
  readonly title: string;
  readonly open: boolean;
  readonly onClose: () => void;
  readonly children: ReactNode;
};

export function HudModal({ title, open, onClose, children }: HudModalProps) {
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  const trapFocus = useCallback((e: KeyboardEvent) => {
    if (e.key !== 'Tab' || !dialogRef.current) return;

    const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
    );
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

    // Mark background content as inert for accessibility
    const root = document.getElementById('app-shell');
    if (root) root.setAttribute('inert', '');

    const raf = requestAnimationFrame(() => {
      const firstFocusable = dialogRef.current?.querySelector<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      firstFocusable?.focus();
    });

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCloseRef.current();
      trapFocus(e);
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      cancelAnimationFrame(raf);
      if (root) root.removeAttribute('inert');
      previousFocusRef.current?.focus();
    };
  }, [open, trapFocus]);

  if (!open) return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <button
        type="button"
        tabIndex={-1}
        aria-label="Close modal"
        className="absolute inset-0 bg-aiki-overlay border-none cursor-default"
        onClick={onClose}
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="relative w-full max-w-lg max-h-[80vh] m-4 animate-scale-in"
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="absolute top-1 right-2 z-10 text-aiki-text-tertiary/60 hover:text-aiki-accent transition-colors text-xs font-sans rounded-md hover:bg-white/[0.05] px-2 py-1"
        >
          ESC
        </button>
        <HudPanel title={title} titleId={titleId} className="flex flex-col max-h-[80vh]">
          <div className="overflow-y-auto p-5 text-sm font-mono text-aiki-text-secondary">
            {children}
          </div>
        </HudPanel>
      </div>
    </div>,
    document.body,
  );
}
