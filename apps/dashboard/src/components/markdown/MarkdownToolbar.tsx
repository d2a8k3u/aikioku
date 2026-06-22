'use client';

import { type KeyboardEvent, type ReactNode, useRef, useState } from 'react';

import { cn } from '@/lib/cn';
import type { MdAction } from './markdown-actions';

const svg = (children: ReactNode) => (
  <svg
    aria-hidden="true"
    focusable="false"
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    {children}
  </svg>
);

const glyph = (text: string) => (
  <span aria-hidden="true" className="text-[12px] font-semibold leading-none">
    {text}
  </span>
);

type ToolbarItem = { readonly action: MdAction; readonly label: string; readonly icon: ReactNode };

const ITEMS: ReadonlyArray<ToolbarItem> = [
  { action: 'h1', label: 'Heading 1', icon: glyph('H1') },
  { action: 'h2', label: 'Heading 2', icon: glyph('H2') },
  {
    action: 'bold',
    label: 'Bold',
    icon: (
      <span aria-hidden="true" className="text-[13px] font-bold leading-none">
        B
      </span>
    ),
  },
  {
    action: 'italic',
    label: 'Italic',
    icon: (
      <span aria-hidden="true" className="font-serif text-[13px] italic leading-none">
        I
      </span>
    ),
  },
  {
    action: 'bulletList',
    label: 'Bulleted list',
    icon: svg(
      <>
        <line x1="9" y1="6" x2="20" y2="6" />
        <line x1="9" y1="12" x2="20" y2="12" />
        <line x1="9" y1="18" x2="20" y2="18" />
        <circle cx="4.5" cy="6" r="1.2" fill="currentColor" stroke="none" />
        <circle cx="4.5" cy="12" r="1.2" fill="currentColor" stroke="none" />
        <circle cx="4.5" cy="18" r="1.2" fill="currentColor" stroke="none" />
      </>,
    ),
  },
  {
    action: 'orderedList',
    label: 'Numbered list',
    icon: svg(
      <>
        <line x1="10" y1="6" x2="20" y2="6" />
        <line x1="10" y1="12" x2="20" y2="12" />
        <line x1="10" y1="18" x2="20" y2="18" />
        <text x="2" y="8.5" fontSize="7" fill="currentColor" stroke="none">
          1
        </text>
        <text x="2" y="14.5" fontSize="7" fill="currentColor" stroke="none">
          2
        </text>
        <text x="2" y="20.5" fontSize="7" fill="currentColor" stroke="none">
          3
        </text>
      </>,
    ),
  },
  {
    action: 'quote',
    label: 'Quote',
    icon: svg(
      <>
        <line x1="5" y1="5" x2="5" y2="19" />
        <line x1="9" y1="8" x2="19" y2="8" />
        <line x1="9" y1="12" x2="19" y2="12" />
        <line x1="9" y1="16" x2="16" y2="16" />
      </>,
    ),
  },
  {
    action: 'inlineCode',
    label: 'Inline code',
    icon: svg(
      <>
        <polyline points="16 18 22 12 16 6" />
        <polyline points="8 6 2 12 8 18" />
      </>,
    ),
  },
  {
    action: 'codeBlock',
    label: 'Code block',
    icon: svg(
      <>
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <polyline points="9 10 7 13 9 16" />
        <polyline points="14 10 16 13 14 16" />
      </>,
    ),
  },
  {
    action: 'link',
    label: 'Link',
    icon: svg(
      <>
        <path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1" />
        <path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1" />
      </>,
    ),
  },
];

type MarkdownToolbarProps = {
  readonly onAction: (action: MdAction) => void;
  readonly className?: string;
};

export function MarkdownToolbar({ onAction, className }: MarkdownToolbarProps) {
  const [focusIndex, setFocusIndex] = useState(0);
  const btnRefs = useRef<Array<HTMLButtonElement | null>>([]);

  const moveFocus = (next: number) => {
    const clamped = (next + ITEMS.length) % ITEMS.length;
    setFocusIndex(clamped);
    btnRefs.current[clamped]?.focus();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      moveFocus(focusIndex + 1);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      moveFocus(focusIndex - 1);
    } else if (e.key === 'Home') {
      e.preventDefault();
      moveFocus(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      moveFocus(ITEMS.length - 1);
    }
  };

  return (
    <div
      role="toolbar"
      aria-label="Markdown formatting"
      aria-orientation="horizontal"
      onKeyDown={onKeyDown}
      className={cn(
        'flex flex-wrap items-center gap-0.5 rounded-lg border border-aiki-border-subtle bg-white/[0.02] p-1',
        className,
      )}
    >
      {ITEMS.map((item, i) => (
        <button
          key={item.action}
          ref={(el) => {
            btnRefs.current[i] = el;
          }}
          type="button"
          aria-label={item.label}
          title={item.label}
          tabIndex={i === focusIndex ? 0 : -1}
          onFocus={() => setFocusIndex(i)}
          onClick={() => onAction(item.action)}
          className={cn(
            'inline-flex h-8 w-8 items-center justify-center rounded-md',
            'text-aiki-text-tertiary',
            'transition-colors duration-200',
            'hover:bg-white/[0.05] hover:text-aiki-text-secondary',
          )}
        >
          {item.icon}
        </button>
      ))}
    </div>
  );
}
