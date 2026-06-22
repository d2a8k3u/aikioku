'use client';

import { type KeyboardEvent, useLayoutEffect, useRef } from 'react';

import { applyMarkdown, type MdAction } from './markdown-actions';
import { MarkdownPreview } from './MarkdownPreview';
import { MarkdownToolbar } from './MarkdownToolbar';

const SHORTCUTS: Record<string, MdAction> = { b: 'bold', i: 'italic', k: 'link' };

type MarkdownEditorProps = {
  readonly value: string;
  readonly onChange: (value: string) => void;
  readonly mode: 'edit' | 'preview';
  readonly ariaLabelledBy?: string;
  readonly textareaId?: string;
};

export function MarkdownEditor({
  value,
  onChange,
  mode,
  ariaLabelledBy,
  textareaId,
}: MarkdownEditorProps) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const pendingSelection = useRef<{ start: number; end: number } | null>(null);

  // Restore focus + selection after a toolbar/shortcut edit replaces the value.
  useLayoutEffect(() => {
    const sel = pendingSelection.current;
    if (sel && ref.current) {
      ref.current.focus();
      ref.current.setSelectionRange(sel.start, sel.end);
      pendingSelection.current = null;
    }
  });

  const runAction = (action: MdAction) => {
    const ta = ref.current;
    if (!ta) return;
    const result = applyMarkdown(value, ta.selectionStart, ta.selectionEnd, action);
    pendingSelection.current = { start: result.selStart, end: result.selEnd };
    onChange(result.value);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (!(e.metaKey || e.ctrlKey)) return;
    const action = SHORTCUTS[e.key.toLowerCase()];
    if (action) {
      e.preventDefault();
      runAction(action);
    }
  };

  if (mode === 'preview') {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto rounded-xl border border-aiki-border-subtle bg-[rgba(8,11,16,0.4)] p-5">
        <MarkdownPreview content={value} />
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <MarkdownToolbar onAction={runAction} />
      <textarea
        ref={ref}
        id={textareaId}
        aria-labelledby={ariaLabelledBy}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        spellCheck
        placeholder="Write in Markdown…"
        className="min-h-0 flex-1 resize-none rounded-xl border border-aiki-border-subtle bg-[rgba(8,11,16,0.4)] p-5 font-mono text-sm leading-7 text-aiki-text-secondary outline-none transition-colors placeholder:text-aiki-text-muted focus:border-aiki-accent/40"
      />
    </div>
  );
}
