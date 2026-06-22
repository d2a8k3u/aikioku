'use client';

import { useState, useEffect, useCallback } from 'react';
import { reviewApi, notesApi } from '@/lib/api';
import type { ReviewStats, Card, Note } from '@/types';
import { HudModal } from '@/components/hud/HudModal';
import { HudInput } from '@/components/hud/HudInput';
import { HudButton } from '@/components/hud/HudButton';
import { HudSpinner } from '@/components/hud/HudSpinner';
import { useToast } from '@/components/hud/HudToast';
import { cn } from '@/lib/cn';

// ── Animated counter ────────────────────────────────────

function useAnimatedValue(target: number, duration = 600): number {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (target === 0) {
      setValue(0);
      return;
    }
    const steps = 20;
    const ms = duration / steps;
    let step = 0;
    const timer = setInterval(() => {
      step++;
      if (step >= steps) {
        setValue(target);
        clearInterval(timer);
        return;
      }
      const t = step / steps;
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(value + (target - value) * eased));
    }, ms);
    return () => clearInterval(timer);
  }, [target, duration]);
  return value;
}

// ── Make-cards-from-notes picker ────────────────────────

function MakeCardsModal({
  open,
  onClose,
  onGenerated,
}: {
  open: boolean;
  onClose: () => void;
  onGenerated: () => void;
}) {
  const { addToast } = useToast();
  const [notes, setNotes] = useState<Note[]>([]);
  const [notesLoading, setNotesLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setNotesLoading(true);
    setLoadError(null);
    setSelected(new Set());
    setQuery('');
    notesApi
      .list({ limit: 200 })
      .then((data) => {
        if (!cancelled) setNotes(data);
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : 'Failed to load notes');
      })
      .finally(() => {
        if (!cancelled) setNotesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const q = query.trim().toLowerCase();
  const filtered = q ? notes.filter((n) => n.title.toLowerCase().includes(q)) : notes;

  const handleGenerate = async () => {
    if (generating || selected.size === 0) return;
    setGenerating(true);
    let created = 0;
    let ok = 0;
    let failed = 0;
    // Sequential — gentle on the LLM and the daily generation budget.
    for (const id of selected) {
      try {
        const cards = await reviewApi.generateCards(id);
        created += cards.length;
        ok += 1;
      } catch {
        failed += 1;
      }
    }
    setGenerating(false);

    if (ok === 0) {
      addToast('Card generation failed. Check the LLM provider and retry.', 'error');
      return; // keep the modal open so the user can retry
    }
    const suffix = failed > 0 ? ` (${failed} failed)` : '';
    addToast(
      `Created ${created} card${created === 1 ? '' : 's'} from ${ok} note${ok === 1 ? '' : 's'}${suffix}`,
      created > 0 ? 'success' : 'warning',
    );
    setSelected(new Set());
    onClose();
    onGenerated();
  };

  return (
    <HudModal title="Make cards from notes" open={open} onClose={onClose}>
      <div className="flex flex-col gap-3">
        <HudInput
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search notes…"
          aria-label="Search notes"
        />

        {notesLoading ? (
          <div className="flex items-center justify-center py-10">
            <HudSpinner />
          </div>
        ) : loadError ? (
          <p className="py-6 text-center text-xs text-aiki-danger">{loadError}</p>
        ) : notes.length === 0 ? (
          <p className="py-6 text-center text-xs text-aiki-text-tertiary">
            No notes yet. Create a note first, then generate cards from it.
          </p>
        ) : filtered.length === 0 ? (
          <p className="py-6 text-center text-xs text-aiki-text-tertiary">
            No notes match your search.
          </p>
        ) : (
          <ul className="flex max-h-[42vh] flex-col gap-1.5 overflow-y-auto pr-1">
            {filtered.map((note) => {
              const isSel = selected.has(note.id);
              return (
                <li key={note.id}>
                  <button
                    type="button"
                    aria-pressed={isSel}
                    onClick={() => toggle(note.id)}
                    className={cn(
                      'flex w-full items-start gap-2.5 rounded-lg border px-3 py-2 text-left transition-colors',
                      isSel
                        ? 'border-aiki-accent/50 bg-aiki-accent/10'
                        : 'border-aiki-border bg-white/[0.02] hover:bg-white/[0.04]',
                    )}
                  >
                    <span
                      aria-hidden="true"
                      className={cn(
                        'mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded border text-[9px] leading-none',
                        isSel
                          ? 'border-aiki-accent bg-aiki-accent text-white'
                          : 'border-aiki-border',
                      )}
                    >
                      {isSel ? '✓' : ''}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-xs text-aiki-text-secondary">
                        {note.title}
                      </span>
                      {note.path ? (
                        <span className="mt-0.5 block truncate text-[10px] text-aiki-text-tertiary/70">
                          {note.path}
                        </span>
                      ) : null}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}

        <div className="flex items-center justify-between pt-1">
          <span className="text-[10px] text-aiki-text-tertiary">
            {selected.size > 0 ? `${selected.size} selected` : 'Select notes to generate cards'}
          </span>
          <HudButton onClick={handleGenerate} disabled={selected.size === 0 || generating}>
            {generating ? (
              <>
                <HudSpinner size={14} /> Generating…
              </>
            ) : (
              `Generate (${selected.size})`
            )}
          </HudButton>
        </div>
      </div>
    </HudModal>
  );
}

// ── Main Page ───────────────────────────────────────────

export default function ReviewPage() {
  const [stats, setStats] = useState<ReviewStats | null>(null);
  const [dueCards, setDueCards] = useState<Card[]>([]);
  const [loading, setLoading] = useState(true);
  const [reviewing, setReviewing] = useState<string | null>(null);
  const [showAnswer, setShowAnswer] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [s, cards] = await Promise.all([reviewApi.stats(), reviewApi.due()]);
      setStats(s);
      setDueCards(cards);
    } catch {
      // degrade
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const animTotal = useAnimatedValue(stats?.total ?? 0);
  const animDue = useAnimatedValue(stats?.due ?? 0);
  const animReview = useAnimatedValue(stats?.review ?? 0);

  const handleRate = async (cardId: string, rating: number) => {
    try {
      await reviewApi.reviewCard(cardId, rating);
      setShowAnswer(false);
      setReviewing(null);
      fetchData();
    } catch {
      // degrade
    }
  };

  const currentCard = dueCards.find((c) => c.id === reviewing);

  if (loading && !stats) {
    return (
      <div className="flex h-full items-center justify-center">
        <HudSpinner size={32} />
      </div>
    );
  }

  return (
    <>
      <div
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: 'auto',
          display: 'flex',
          justifyContent: 'center',
          padding: '40px clamp(16px, 4vw, 32px)',
        }}
      >
        <div
          style={{
            width: '100%',
            maxWidth: 1280,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <div
            style={{
              width: '100%',
              maxWidth: 640,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              textAlign: 'center',
            }}
          >
            {/* Orb */}
            <div
              style={{
                position: 'relative',
                width: 96,
                height: 96,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: 26,
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  width: 96,
                  height: 96,
                  borderRadius: '50%',
                  background: 'radial-gradient(circle, rgba(184,115,51,0.12), transparent 70%)',
                  animation: 'auraPulse 4.5s ease-in-out infinite',
                }}
              />
              <svg
                width="96"
                height="96"
                style={{ position: 'absolute', animation: 'spinSlow 26s linear infinite' }}
              >
                <circle
                  cx="48"
                  cy="48"
                  r="44"
                  fill="none"
                  stroke="rgba(184,115,51,0.12)"
                  strokeWidth="1"
                  strokeDasharray="3 8"
                />
              </svg>
              <div
                style={{
                  width: 60,
                  height: 60,
                  borderRadius: '50%',
                  border: '1px solid rgba(184,115,51,0.22)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(184,115,51,0.04)',
                  animation: 'floatY 4s ease-in-out infinite',
                }}
              >
                <svg
                  width="28"
                  height="28"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#cf9a5e"
                  strokeWidth="1.4"
                >
                  <rect x="4" y="6" width="13" height="14" rx="2" />
                  <path d="M8 6V4h12v14h-2" />
                </svg>
              </div>
            </div>

            {/* Active review card */}
            {currentCard ? (
              <div style={{ width: '100%', marginBottom: 28 }}>
                <div
                  style={{
                    padding: 'clamp(16px, 4vw, 24px)',
                    borderRadius: 15,
                    border: '1px solid rgba(184,115,51,0.2)',
                    background: 'rgba(13,17,23,0.5)',
                    marginBottom: 16,
                  }}
                >
                  <div
                    style={{
                      font: '500 16px/1.4 "JetBrains Mono"',
                      color: '#eef0f3',
                      marginBottom: 16,
                    }}
                  >
                    {currentCard.front}
                  </div>
                  {showAnswer && (
                    <div
                      style={{
                        padding: 16,
                        borderRadius: 11,
                        background: 'rgba(52,214,196,0.06)',
                        border: '1px solid rgba(52,214,196,0.15)',
                        font: '400 14px/1.5 "JetBrains Mono"',
                        color: '#9fe3ec',
                        marginBottom: 16,
                      }}
                    >
                      {currentCard.back}
                    </div>
                  )}
                  {!showAnswer ? (
                    <button
                      onClick={() => setShowAnswer(true)}
                      style={{
                        width: '100%',
                        padding: '12px',
                        borderRadius: 11,
                        border: '1px solid rgba(52,214,196,0.3)',
                        background: 'rgba(52,214,196,0.08)',
                        color: '#4cc9bb',
                        font: '600 11px/1 "JetBrains Mono"',
                        letterSpacing: '0.14em',
                        cursor: 'pointer',
                        transition: 'all .2s',
                      }}
                    >
                      SHOW ANSWER
                    </button>
                  ) : (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
                      {[
                        { label: 'FORGOT', rating: 1, color: '#f87171' },
                        { label: 'HARD', rating: 2, color: '#fbbf24' },
                        { label: 'GOOD', rating: 3, color: '#5fd3e0' },
                        { label: 'EASY', rating: 4, color: '#34d399' },
                      ].map((r) => (
                        <button
                          key={r.rating}
                          onClick={() => handleRate(currentCard.id, r.rating)}
                          style={{
                            padding: '10px',
                            borderRadius: 9,
                            border: `1px solid ${r.color}40`,
                            background: `${r.color}10`,
                            color: r.color,
                            font: '600 10px/1 "JetBrains Mono"',
                            letterSpacing: '0.1em',
                            cursor: 'pointer',
                            transition: 'all .2s',
                          }}
                        >
                          {r.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => {
                    setReviewing(null);
                    setShowAnswer(false);
                  }}
                  style={{
                    padding: '8px 16px',
                    borderRadius: 9,
                    border: '1px solid rgba(255,255,255,0.08)',
                    background: 'transparent',
                    color: '#6b707a',
                    font: '500 9px/1 "JetBrains Mono"',
                    letterSpacing: '0.1em',
                    cursor: 'pointer',
                  }}
                >
                  SKIP
                </button>
              </div>
            ) : dueCards.length > 0 ? (
              <>
                <h1
                  style={{
                    margin: 0,
                    font: '300 24px/1.3 "JetBrains Mono"',
                    fontSize: 'clamp(20px, 4vw, 24px)',
                    color: '#eef0f3',
                  }}
                >
                  {dueCards.length} cards waiting
                </h1>
                <p
                  style={{
                    margin: '16px 0 28px',
                    font: '400 12px/1.7 "JetBrains Mono"',
                    color: '#8a8f99',
                  }}
                >
                  KIO reminds you of exactly what you&apos;d otherwise forget.
                </p>
                <button
                  onClick={() => setReviewing(dueCards[0].id)}
                  style={{
                    padding: '14px 28px',
                    borderRadius: 13,
                    border: '1px solid rgba(184,115,51,0.3)',
                    background:
                      'linear-gradient(180deg, rgba(184,115,51,0.13), rgba(184,115,51,0.04))',
                    color: '#e7b876',
                    font: '600 11px/1 "JetBrains Mono"',
                    letterSpacing: '0.16em',
                    cursor: 'pointer',
                    transition: 'all .25s',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(184,115,51,0.6)';
                    e.currentTarget.style.boxShadow = '0 0 24px rgba(184,115,51,0.18)';
                    e.currentTarget.style.transform = 'translateY(-1px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(184,115,51,0.3)';
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  START REVIEW
                </button>
              </>
            ) : (
              <>
                <h1
                  style={{
                    margin: 0,
                    font: '300 24px/1.3 "JetBrains Mono"',
                    fontSize: 'clamp(20px, 4vw, 24px)',
                    color: '#eef0f3',
                  }}
                >
                  Nothing to review right now
                </h1>
                <p
                  style={{
                    margin: '16px 0 28px',
                    font: '400 12px/1.7 "JetBrains Mono"',
                    color: '#8a8f99',
                  }}
                >
                  KIO can turn your notes into cards and remind you of them exactly when you&apos;d
                  otherwise forget.
                </p>
                <button
                  onClick={() => setPickerOpen(true)}
                  style={{
                    padding: '14px 28px',
                    borderRadius: 13,
                    border: '1px solid rgba(184,115,51,0.3)',
                    background:
                      'linear-gradient(180deg, rgba(184,115,51,0.13), rgba(184,115,51,0.04))',
                    color: '#e7b876',
                    font: '600 11px/1 "JetBrains Mono"',
                    letterSpacing: '0.16em',
                    cursor: 'pointer',
                    transition: 'all .25s',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(184,115,51,0.6)';
                    e.currentTarget.style.boxShadow = '0 0 24px rgba(184,115,51,0.18)';
                    e.currentTarget.style.transform = 'translateY(-1px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(184,115,51,0.3)';
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  MAKE CARDS FROM NOTES
                </button>
              </>
            )}

            {/* Stats */}
            <div
              style={{
                display: 'flex',
                gap: 12,
                marginTop: 36,
                width: '100%',
                flexWrap: 'wrap',
                justifyContent: 'center',
              }}
            >
              <div
                style={{
                  flex: '1 1 0',
                  minWidth: 88,
                  maxWidth: 132,
                  padding: 16,
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: 12,
                  background: 'rgba(255,255,255,0.015)',
                }}
              >
                <div style={{ font: '300 24px/1 "JetBrains Mono"', color: '#cfd2d8' }}>
                  {animTotal}
                </div>
                <div
                  style={{
                    font: '400 8px/1 "JetBrains Mono"',
                    letterSpacing: '0.16em',
                    color: '#5a5f69',
                    marginTop: 8,
                  }}
                >
                  TOTAL
                </div>
              </div>
              <div
                style={{
                  flex: '1 1 0',
                  minWidth: 88,
                  maxWidth: 132,
                  padding: 16,
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: 12,
                  background: 'rgba(255,255,255,0.015)',
                }}
              >
                <div style={{ font: '300 24px/1 "JetBrains Mono"', color: '#e7b876' }}>
                  {animDue}
                </div>
                <div
                  style={{
                    font: '400 8px/1 "JetBrains Mono"',
                    letterSpacing: '0.16em',
                    color: '#5a5f69',
                    marginTop: 8,
                  }}
                >
                  LEARNING
                </div>
              </div>
              <div
                style={{
                  flex: '1 1 0',
                  minWidth: 88,
                  maxWidth: 132,
                  padding: 16,
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: 12,
                  background: 'rgba(255,255,255,0.015)',
                }}
              >
                <div style={{ font: '300 24px/1 "JetBrains Mono"', color: '#5fd3e0' }}>
                  {animReview}
                </div>
                <div
                  style={{
                    font: '400 8px/1 "JetBrains Mono"',
                    letterSpacing: '0.16em',
                    color: '#5a5f69',
                    marginTop: 8,
                  }}
                >
                  TO REVIEW
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <MakeCardsModal
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onGenerated={fetchData}
      />
    </>
  );
}
