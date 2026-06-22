'use client';

import { type ReactNode, useCallback, useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { cn } from '@/lib/cn';

type CommandItem = {
  readonly id: string;
  readonly label: string;
  readonly description?: string;
  readonly icon?: ReactNode;
  readonly onSelect: () => void;
};

type HudCommandPaletteProps = {
  readonly open: boolean;
  readonly onClose: () => void;
  readonly items: readonly CommandItem[];
  readonly placeholder?: string;
};

export function HudCommandPalette({
  open,
  onClose,
  items,
  placeholder = 'Type a command…',
}: HudCommandPaletteProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const listboxId = useId();
  const optionPrefix = useId();

  const filtered = items.filter(
    (item) =>
      item.label.toLowerCase().includes(query.toLowerCase()) ||
      (item.description?.toLowerCase().includes(query.toLowerCase()) ?? false),
  );

  // Reset state on open
  useEffect(() => {
    if (open) {
      setQuery('');
      setActiveIndex(0);
      // Focus input after portal render
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  // Clamp active index when filtered list changes
  useEffect(() => {
    setActiveIndex((prev) => Math.min(prev, Math.max(filtered.length - 1, 0)));
  }, [filtered.length]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
        case 'ArrowDown':
          e.preventDefault();
          setActiveIndex((prev) => (prev + 1) % Math.max(filtered.length, 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setActiveIndex((prev) => (prev - 1 + filtered.length) % Math.max(filtered.length, 1));
          break;
        case 'Enter':
          e.preventDefault();
          if (filtered[activeIndex]) {
            filtered[activeIndex].onSelect();
            onClose();
          }
          break;
      }
    },
    [filtered, activeIndex, onClose],
  );

  // Scroll active item into view
  useEffect(() => {
    const activeEl = listRef.current?.children[activeIndex] as HTMLElement | undefined;
    activeEl?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex]);

  if (!open) return null;

  return createPortal(
    <div className="fixed inset-0 z-[60] flex items-start justify-center pt-[15vh]">
      {/* Backdrop */}
      <button
        type="button"
        tabIndex={-1}
        aria-label="Close command palette"
        className="absolute inset-0 bg-black/40 backdrop-blur-sm border-none cursor-default"
        onClick={onClose}
      />

      <div
        role="combobox"
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-controls={listboxId}
        aria-owns={listboxId}
        className={cn(
          'relative w-full max-w-lg',
          'backdrop-blur-sm rounded-xl',
          'border border-aiki-accent-border',
          'bg-aiki-panel',
          'shadow-[0_0_40px_rgba(184,115,51,0.15),inset_0_1px_0_rgba(255,255,255,0.05)]',
          'animate-scale-in',
        )}
      >
        {/* Search input */}
        <div className="flex items-center gap-2 px-3 py-2 border-b border-white/[0.06]">
          {/* Search icon */}
          <svg
            aria-hidden="true"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--color-aiki-accent)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            aria-expanded={open}
            aria-controls={listboxId}
            aria-activedescendant={
              filtered[activeIndex] ? `${optionPrefix}-${activeIndex}` : undefined
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className={cn(
              'flex-1 bg-transparent border-none outline-none',
              'text-sm font-mono text-aiki-text-secondary',
              'placeholder:text-aiki-text-tertiary/40',
            )}
          />
          {/* Cmd+K hint */}
          <kbd className="text-[9px] font-mono text-aiki-text-tertiary/50 px-1.5 py-0.5 rounded border border-white/[0.08] bg-white/[0.03]">
            ⌘K
          </kbd>
        </div>

        {/* Results list */}
        <ul ref={listRef} id={listboxId} role="listbox" className="max-h-64 overflow-y-auto p-1">
          {filtered.length === 0 ? (
            <li className="px-3 py-4 text-center text-xs text-aiki-text-tertiary/50 font-mono">
              No results
            </li>
          ) : (
            filtered.map((item, i) => (
              <li
                key={item.id}
                id={`${optionPrefix}-${i}`}
                role="option"
                aria-selected={i === activeIndex}
                className={cn(
                  'flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer',
                  'text-sm font-mono text-aiki-text-secondary',
                  'transition-colors duration-150',
                  i === activeIndex ? 'bg-aiki-accent/15 text-aiki-text' : 'hover:bg-white/[0.04]',
                )}
                onClick={() => {
                  item.onSelect();
                  onClose();
                }}
                onMouseEnter={() => setActiveIndex(i)}
              >
                {item.icon && <span className="flex-shrink-0 text-aiki-accent">{item.icon}</span>}
                <div className="flex flex-col min-w-0">
                  <span className="truncate">{item.label}</span>
                  {item.description && (
                    <span className="text-[10px] text-aiki-text-tertiary truncate">
                      {item.description}
                    </span>
                  )}
                </div>
              </li>
            ))
          )}
        </ul>
      </div>
    </div>,
    document.body,
  );
}
