'use client';

import { useState, useId, useRef } from 'react';
import { usePathname } from 'next/navigation';
import ErrorBoundary from '@/components/ErrorBoundary';
import ReembedProvider from '@/components/ReembedProvider';
import ReembedBanner from '@/components/ReembedBanner';
import BudgetProvider from '@/components/BudgetProvider';
import BudgetBanner from '@/components/BudgetBanner';
import { notesApi } from '@/lib/api';
import { AuthGuard } from '@/components/auth';
import { RailLayout } from '@/components/layout/RailLayout';
import { HudModal } from '@/components/hud/HudModal';
import { HudInput } from '@/components/hud/HudInput';
import { HudTextarea } from '@/components/hud/HudTextarea';
import { HudButton } from '@/components/hud/HudButton';
import { HudToast, useToast } from '@/components/hud/HudToast';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useHealth } from '@/hooks/useHealth';

// ── Inline SVG Icons ───────────────────────────────────

function PlusIcon() {
  return (
    <svg
      aria-hidden="true"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg
      aria-hidden="true"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

// ── QuickCapture ────────────────────────────────────────

function QuickCapture() {
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { addToast } = useToast();

  const titleId = useId();
  const contentId = useId();
  const hintId = useId();
  const errorId = useId();
  const contentRef = useRef<HTMLTextAreaElement>(null);

  const close = () => {
    setOpen(false);
    setError(null);
  };

  const handleSubmit = async () => {
    if (saving) return;
    if (!content.trim()) {
      setError('Note content is required.');
      contentRef.current?.focus();
      return;
    }
    setError(null);
    try {
      setSaving(true);
      await notesApi.quickCreate({ title: title.trim() || undefined, content: content.trim() });
      addToast('Note saved', 'success');
      setTitle('');
      setContent('');
      setOpen(false);
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Failed to save', 'error');
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="fixed bottom-6 right-6 z-50 rounded-full bg-aiki-accent p-3.5 text-white shadow-lg transition-colors hover:bg-aiki-accent-hover"
        aria-label="Quick capture"
      >
        <PlusIcon />
      </button>

      <HudModal title="Quick Capture" open={open} onClose={close}>
        <form
          className="space-y-3"
          noValidate
          onSubmit={(e) => {
            e.preventDefault();
            handleSubmit();
          }}
        >
          <div>
            <label htmlFor={titleId} className="sr-only">
              Title (optional)
            </label>
            <HudInput
              id={titleId}
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Title (optional)"
            />
          </div>

          <div>
            <label htmlFor={contentId} className="sr-only">
              Note content (required)
            </label>
            <HudTextarea
              ref={contentRef}
              id={contentId}
              value={content}
              onChange={(e) => {
                setContent(e.target.value);
                if (error) setError(null);
              }}
              placeholder="What's on your mind?"
              minRows={4}
              maxRows={8}
              required
              aria-required="true"
              aria-invalid={error ? true : undefined}
              aria-describedby={error ? errorId : hintId}
              onKeyDown={(e: React.KeyboardEvent) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
            />
            {error ? (
              <p id={errorId} role="alert" className="mt-1 text-[10px] font-mono text-aiki-danger">
                {error}
              </p>
            ) : (
              <p id={hintId} className="mt-1 text-[10px] font-mono text-aiki-text-tertiary">
                Press ⌘/Ctrl + Enter to save.
              </p>
            )}
          </div>

          <div className="flex items-center justify-end">
            <HudButton type="submit" disabled={saving}>
              <SendIcon />
              {saving ? 'Saving...' : 'Save'}
            </HudButton>
          </div>
        </form>
      </HudModal>
    </>
  );
}

// ── Main layout ────────────────────────────────────────

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  useWebSocket();
  useHealth();

  // The graph view has its own full-bleed HUD; the quick-capture FAB would overlap its controls.
  const showQuickCapture = pathname !== '/graph';

  return (
    <BudgetProvider>
      <ReembedProvider>
        <HudToast>
          <AuthGuard>
            <RailLayout>
              <BudgetBanner />
              <ReembedBanner />
              <ErrorBoundary>{children}</ErrorBoundary>
            </RailLayout>
            {showQuickCapture && <QuickCapture />}
          </AuthGuard>
        </HudToast>
      </ReembedProvider>
    </BudgetProvider>
  );
}
