'use client';

import { forwardRef, type TextareaHTMLAttributes, useCallback, useRef } from 'react';

import { cn } from '@/lib/cn';

type HudTextareaProps = TextareaHTMLAttributes<HTMLTextAreaElement> & {
  maxRows?: number;
  minRows?: number;
};

export const HudTextarea = forwardRef<HTMLTextAreaElement, HudTextareaProps>(
  ({ className = '', maxRows = 10, minRows = 1, ...props }, extRef) => {
    const intRef = useRef<HTMLTextAreaElement | null>(null);

    const textareaRef = (el: HTMLTextAreaElement | null) => {
      intRef.current = el;
      if (extRef) {
        if (typeof extRef === 'function') extRef(el);
        else (extRef as unknown as { current: HTMLTextAreaElement | null }).current = el;
      }
    };

    const handleResize = useCallback(() => {
      const textarea = intRef.current;
      if (!textarea) return;
      textarea.style.height = 'auto';
      const lineHeight = 24; // px
      const maxHeight = maxRows * lineHeight;
      textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
    }, [maxRows]);

    return (
      <textarea
        ref={textareaRef}
        rows={minRows}
        className={cn(
          'w-full bg-white/[0.03] backdrop-blur-sm border border-aiki-border rounded-lg px-3 py-2',
          'resize-none overflow-y-auto',
          'text-sm font-mono text-aiki-text-secondary',
          'placeholder:text-aiki-text-tertiary/40',
          'focus:border-aiki-accent/50 focus:shadow-[0_0_16px_rgba(184,115,51,0.1)] focus:bg-white/[0.05]',
          'transition-all duration-300 outline-none',
          'whitespace-pre-wrap',
          className,
        )}
        style={{
          minHeight: `${minRows * 24}px`,
          maxHeight: `${maxRows * 24}px`,
        }}
        onInput={handleResize}
        {...props}
      />
    );
  },
);

HudTextarea.displayName = 'HudTextarea';
