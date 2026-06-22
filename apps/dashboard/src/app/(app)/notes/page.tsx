'use client';

import { type KeyboardEvent, useCallback, useEffect, useId, useRef, useState } from 'react';

import { notesApi } from '@/lib/api';
import type { Note } from '@/types';
import { useMediaQuery } from '@/hooks/useMediaQuery';
import { cn } from '@/lib/cn';
import { HudButton } from '@/components/hud/HudButton';
import { HudInput } from '@/components/hud/HudInput';
import { HudBadge } from '@/components/hud/HudBadge';
import { HudSpinner } from '@/components/hud/HudSpinner';
import { HudSegmented, type SegmentedOption } from '@/components/hud/HudSegmented';
import { HudModal } from '@/components/hud/HudModal';
import { useToast } from '@/components/hud/HudToast';
import { MarkdownEditor } from '@/components/markdown/MarkdownEditor';

// ── Helpers ─────────────────────────────────────────────

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'now';
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs} h ago`;
  const days = Math.floor(hrs / 24);
  return `${days} d ago`;
}

function extractTags(note: Note): string[] {
  const fm = note.frontmatter || {};
  if (Array.isArray(fm.tags)) return fm.tags as string[];
  if (typeof fm.tags === 'string') return (fm.tags as string).split(',').map((t) => t.trim());
  return [];
}

type EditorMode = 'edit' | 'preview';

const VIEW_OPTIONS: SegmentedOption<EditorMode>[] = [
  { value: 'preview', label: 'Preview', icon: <EyeIcon /> },
  { value: 'edit', label: 'Edit', icon: <PencilIcon /> },
];

// ── Icons ───────────────────────────────────────────────

function PlusIcon() {
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
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

function BackIcon() {
  return (
    <svg
      aria-hidden="true"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="15 18 9 12 15 6" />
    </svg>
  );
}

function EyeIcon() {
  return (
    <svg
      aria-hidden="true"
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function PencilIcon() {
  return (
    <svg
      aria-hidden="true"
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4z" />
    </svg>
  );
}

function SaveIcon() {
  return (
    <svg
      aria-hidden="true"
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
      <polyline points="17 21 17 13 7 13 7 21" />
      <polyline points="7 3 7 8 15 8" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg
      aria-hidden="true"
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}

function BulbIcon() {
  return (
    <svg
      aria-hidden="true"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      className="flex-none"
    >
      <path d="M12 3a4 4 0 0 0-4 4 3.5 3.5 0 0 0-1 6.8V18a3 3 0 0 0 5 2 3 3 0 0 0 5-2v-4.2A3.5 3.5 0 0 0 16 7a4 4 0 0 0-4-4z" />
    </svg>
  );
}

// ── Page ────────────────────────────────────────────────

export default function NotesPage() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [title, setTitle] = useState('');
  const [body, setBody] = useState('');
  const [insight, setInsight] = useState('');
  const [mode, setMode] = useState<EditorMode>('preview');
  const [mobileView, setMobileView] = useState<'list' | 'detail'>('list');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const isDesktop = useMediaQuery('(min-width: 768px)');
  const { addToast } = useToast();

  const titleHeadingId = useId();
  const editorId = useId();

  const newDraftId = useRef<string | null>(null);
  const activeNoteId = useRef<string | null>(null);
  const titleRef = useRef<HTMLInputElement>(null);
  const backRef = useRef<HTMLButtonElement>(null);
  const optionRefs = useRef<Record<string, HTMLLIElement | null>>({});

  const selectedNote = notes.find((n) => n.id === selectedId) ?? null;
  const tags = selectedNote ? extractTags(selectedNote) : [];
  const isDirty =
    selectedNote != null && (body !== selectedNote.content || title !== selectedNote.title);

  const loadInsight = useCallback((id: string) => {
    setInsight('');
    notesApi
      .getRelated(id)
      .then((r) => {
        // Ignore late responses for a note the user has already navigated away from.
        if (r.insight && activeNoteId.current === id) setInsight(r.insight);
      })
      .catch(() => {});
  }, []);

  const selectNote = useCallback(
    (note: Note, opts?: { mode?: EditorMode; openDetail?: boolean }) => {
      setSelectedId(note.id);
      activeNoteId.current = note.id;
      setTitle(note.title || '');
      setBody(note.content || '');
      setMode(opts?.mode ?? (note.id === newDraftId.current ? 'edit' : 'preview'));
      // A freshly created draft has no related insight yet — don't fetch or show one.
      if (note.id === newDraftId.current) setInsight('');
      else loadInsight(note.id);
      if (!isDesktop && (opts?.openDetail ?? true)) setMobileView('detail');
    },
    [isDesktop, loadInsight],
  );

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const list = await notesApi.list({ limit: 50 });
        if (cancelled) return;
        setNotes(list);
        if (list.length > 0) selectNote(list[0], { openDetail: false });
      } catch {
        // degrade to empty state
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // Initial load only; intentionally not re-running when selectNote changes.
  }, []);

  const handleSave = useCallback(async () => {
    if (!selectedNote || saving || !isDirty) return;
    setSaving(true);
    try {
      const saved = await notesApi.update(selectedNote.id, {
        ...selectedNote,
        title: title.trim() || 'Untitled',
        content: body,
        modified: new Date().toISOString(),
      });
      setNotes((prev) => prev.map((n) => (n.id === saved.id ? saved : n)));
      setTitle(saved.title);
      newDraftId.current = null;
      addToast('Note saved', 'success');
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Failed to save note', 'error');
    } finally {
      setSaving(false);
    }
  }, [selectedNote, saving, isDirty, title, body, addToast]);

  const handleNew = useCallback(async () => {
    const now = new Date().toISOString();
    try {
      const created = await notesApi.create({
        title: 'Untitled',
        content: '',
        frontmatter: {},
        links: [],
        path: `note-${Date.now()}.md`,
        created: now,
        modified: now,
      });
      newDraftId.current = created.id;
      setNotes((prev) => [created, ...prev]);
      selectNote(created, { mode: 'edit', openDetail: true });
      requestAnimationFrame(() => titleRef.current?.focus());
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Failed to create note', 'error');
    }
  }, [selectNote, addToast]);

  const handleDelete = useCallback(async () => {
    if (!selectedNote || deleting) return;
    setDeleting(true);
    try {
      await notesApi.delete(selectedNote.id);
      if (selectedNote.id === newDraftId.current) newDraftId.current = null;
      const remaining = notes.filter((n) => n.id !== selectedNote.id);
      setNotes(remaining);
      setConfirmOpen(false);
      setMobileView('list');
      addToast('Note deleted', 'success');
      if (remaining.length > 0) {
        selectNote(remaining[0], { openDetail: false });
      } else {
        setSelectedId(null);
        activeNoteId.current = null;
        setTitle('');
        setBody('');
        setInsight('');
      }
    } catch (err) {
      addToast(err instanceof Error ? err.message : 'Failed to delete note', 'error');
    } finally {
      setDeleting(false);
    }
  }, [selectedNote, deleting, notes, selectNote, addToast]);

  const handleBack = () => {
    setMobileView('list');
    if (selectedId) requestAnimationFrame(() => optionRefs.current[selectedId]?.focus());
  };

  // Cmd/Ctrl+S saves the current note.
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        handleSave();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleSave]);

  // Move focus into the detail pane when it opens on mobile.
  useEffect(() => {
    if (!isDesktop && mobileView === 'detail') {
      requestAnimationFrame(() => backRef.current?.focus());
    }
  }, [mobileView, isDesktop, selectedId]);

  const focusOption = (note: Note) => {
    selectNote(note, { openDetail: false });
    requestAnimationFrame(() => optionRefs.current[note.id]?.focus());
  };

  const onListKeyDown = (e: KeyboardEvent<HTMLUListElement>) => {
    const idx = notes.findIndex((n) => n.id === selectedId);
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      focusOption(notes[Math.min(idx + 1, notes.length - 1)]);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      focusOption(notes[Math.max(idx - 1, 0)]);
    } else if (e.key === 'Home') {
      e.preventDefault();
      if (notes[0]) focusOption(notes[0]);
    } else if (e.key === 'End') {
      e.preventDefault();
      if (notes.length) focusOption(notes[notes.length - 1]);
    } else if ((e.key === 'Enter' || e.key === ' ') && !isDesktop) {
      e.preventDefault();
      setMobileView('detail');
    }
  };

  const listHidden = !isDesktop && mobileView === 'detail';
  const detailHidden = !isDesktop && mobileView === 'list';

  return (
    <div className="flex min-h-0 flex-1 justify-center overflow-hidden">
      <div className="flex min-h-0 w-full max-w-[1280px]">
        {/* List pane */}
        <aside
          className={cn(
            'flex min-h-0 w-full flex-col border-r border-aiki-border-subtle md:w-[260px] md:shrink-0 lg:w-[300px] xl:w-[340px]',
            listHidden && 'hidden',
          )}
        >
          <div className="flex-none p-4 md:p-5">
            <h1 className="font-mono text-xl font-light text-aiki-text">Thoughts</h1>
            <p className="mt-2 font-mono text-[10px] text-aiki-text-muted">
              {notes.length} {notes.length === 1 ? 'note' : 'notes'} you have gathered with KIO
            </p>
            <button
              type="button"
              onClick={handleNew}
              className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl border border-aiki-accent-border bg-aiki-accent-bg py-2.5 font-mono text-[10px] font-semibold tracking-[0.14em] text-aiki-accent transition-all duration-200 hover:border-aiki-accent/55 hover:bg-aiki-accent/15"
            >
              <PlusIcon />
              NEW THOUGHT
            </button>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-5">
            {loading ? (
              <div className="flex justify-center py-10">
                <HudSpinner />
              </div>
            ) : notes.length === 0 ? (
              <p className="px-2 py-8 text-center font-mono text-xs text-aiki-text-muted">
                No thoughts yet. Create your first one.
              </p>
            ) : (
              <ul
                role="listbox"
                aria-label="Notes"
                onKeyDown={onListKeyDown}
                className="flex flex-col gap-1.5"
              >
                {notes.map((note) => {
                  const noteTags = extractTags(note);
                  const isSelected = note.id === selectedId;
                  return (
                    <li
                      key={note.id}
                      ref={(el) => {
                        optionRefs.current[note.id] = el;
                      }}
                      role="option"
                      aria-selected={isSelected}
                      tabIndex={isSelected ? 0 : -1}
                      onClick={() => selectNote(note)}
                      className={cn(
                        'cursor-pointer rounded-xl border p-3.5 transition-colors duration-200',
                        isSelected
                          ? 'border-aiki-accent/30 bg-aiki-accent/[0.06]'
                          : 'border-aiki-border-subtle bg-white/[0.01] hover:border-aiki-border-strong',
                      )}
                    >
                      <div className="mb-1.5 truncate font-mono text-[13px] font-medium text-aiki-text-secondary">
                        {note.title || 'Untitled'}
                      </div>
                      <p className="mb-2.5 truncate font-mono text-[10px] leading-relaxed text-aiki-text-muted">
                        {note.content?.replace(/\n/g, ' ').slice(0, 80) || 'Empty'}
                      </p>
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex min-w-0 gap-1.5">
                          {noteTags.slice(0, 2).map((t) => (
                            <span
                              key={t}
                              className="truncate rounded-md border border-aiki-accent/15 bg-aiki-accent/[0.08] px-2 py-0.5 font-mono text-[8px] tracking-wider text-aiki-accent/80"
                            >
                              {t}
                            </span>
                          ))}
                        </div>
                        <span className="flex-none font-mono text-[8px] text-aiki-text-muted">
                          {timeAgo(note.modified)}
                        </span>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </aside>

        {/* Detail pane */}
        <section
          aria-labelledby={titleHeadingId}
          className={cn('flex min-w-0 flex-1 flex-col', detailHidden && 'hidden')}
        >
          {selectedNote ? (
            <div className="flex min-h-0 flex-1 flex-col p-4 md:px-6 md:py-6 lg:px-8">
              <div className="mx-auto flex min-h-0 w-full max-w-[820px] flex-1 flex-col">
                <div className="@container mb-4 flex items-center justify-between gap-3">
                  <div className="flex min-w-0 flex-1 items-center gap-2">
                    {!isDesktop && (
                      <button
                        ref={backRef}
                        type="button"
                        onClick={handleBack}
                        aria-label="Back to notes list"
                        className="flex-none rounded-lg border border-aiki-border-subtle p-1.5 text-aiki-text-tertiary transition-colors hover:text-aiki-text-secondary"
                      >
                        <BackIcon />
                      </button>
                    )}
                    <div className="min-w-0 flex-1">
                      <h2 id={titleHeadingId} className="sr-only">
                        {title || 'Untitled'}
                      </h2>
                      <HudInput
                        ref={titleRef}
                        aria-label="Note title"
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        placeholder="Untitled"
                        className="border-transparent bg-transparent px-0 py-1 text-lg text-aiki-text focus:bg-transparent focus:shadow-none"
                      />
                      {tags.length > 0 && (
                        <div className="mt-1.5 flex flex-wrap gap-1.5">
                          {tags.map((t) => (
                            <span
                              key={t}
                              className="rounded-md border border-aiki-accent/18 bg-aiki-accent/[0.08] px-2.5 py-1 font-mono text-[9px] tracking-wider text-aiki-accent/80"
                            >
                              {t}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex shrink-0 items-center gap-2">
                    {isDirty && <HudBadge status="active" label="Unsaved" />}
                    <HudSegmented<EditorMode>
                      ariaLabel="Editor view mode"
                      options={VIEW_OPTIONS}
                      value={mode}
                      onChange={setMode}
                      labelClassName="hidden @lg:inline"
                    />
                    <div className="flex items-center gap-2">
                      <HudButton
                        onClick={handleSave}
                        disabled={!isDirty || saving}
                        aria-label="Save note"
                      >
                        <SaveIcon />
                        <span className="hidden @lg:inline">{saving ? 'Saving…' : 'Save'}</span>
                      </HudButton>
                      <HudButton
                        variant="danger"
                        onClick={() => setConfirmOpen(true)}
                        disabled={deleting}
                        aria-label="Delete note"
                      >
                        <TrashIcon />
                        <span className="hidden @lg:inline">Delete</span>
                      </HudButton>
                    </div>
                  </div>
                </div>

                {insight && (
                  <div className="mb-3.5 flex items-center gap-2.5 rounded-xl border border-aiki-accent/14 bg-aiki-accent/[0.05] px-3.5 py-2.5">
                    <span className="text-aiki-accent">
                      <BulbIcon />
                    </span>
                    <span className="font-mono text-[11px] leading-relaxed text-aiki-text-secondary">
                      {insight}
                    </span>
                  </div>
                )}

                <MarkdownEditor
                  value={body}
                  onChange={setBody}
                  mode={mode}
                  ariaLabelledBy={titleHeadingId}
                  textareaId={editorId}
                />
              </div>
            </div>
          ) : (
            <div className="flex flex-1 items-center justify-center p-6">
              <span className="font-mono text-[13px] text-aiki-text-muted">
                {notes.length === 0
                  ? 'No notes yet. Create your first one.'
                  : 'Select a note on the left.'}
              </span>
            </div>
          )}
        </section>
      </div>

      <HudModal title="Delete note" open={confirmOpen} onClose={() => setConfirmOpen(false)}>
        <div className="space-y-4">
          <p className="text-sm text-aiki-text-secondary">
            Delete “{selectedNote?.title || 'Untitled'}”? This cannot be undone.
          </p>
          <div className="flex justify-end gap-2">
            <HudButton variant="ghost" onClick={() => setConfirmOpen(false)} disabled={deleting}>
              Cancel
            </HudButton>
            <HudButton variant="danger" onClick={handleDelete} disabled={deleting}>
              {deleting ? 'Deleting…' : 'Delete'}
            </HudButton>
          </div>
        </div>
      </HudModal>
    </div>
  );
}
