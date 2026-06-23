'use client';

import { useState, useRef, useEffect, useLayoutEffect, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';
import { chatApi, buddyApi, conversationsApi, statsApi } from '@/lib/api';
import { MarkdownPreview } from '@/components/markdown/MarkdownPreview';
import type { BuddyProfile, ConversationMessage } from '@/types';

// ── Types ──────────────────────────────────────────────

type Message = {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  time: string;
  chips?: string[];
  inProgress?: boolean;
};

// ── Helpers ─────────────────────────────────────────────

const PAGE_SIZE = 10;
const LOAD_OLDER_THRESHOLD_PX = 80;

function nowTime(): string {
  const d = new Date();
  return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
}

function convToMessage(c: ConversationMessage): Message {
  const chips = c.citations.map((ct) => `note: ${ct.title || ct.note_id}`);
  return {
    id: c.id,
    role: c.role,
    text: c.content,
    time: formatTime(c.created),
    inProgress: c.role === 'assistant' ? c.in_progress : false,
    ...(chips.length > 0 ? { chips } : {}),
  };
}

// ── Main Page ───────────────────────────────────────────

export default function ChatPage() {
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get('q') || '';

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<'simple' | 'multi_hop'>('simple');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPreparing, setIsPreparing] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [profile, setProfile] = useState<BuddyProfile | null>(null);
  const [noteCount, setNoteCount] = useState(0);
  const [memoryCount, setMemoryCount] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [snippets, setSnippets] = useState<{ note_id: string; title: string; snippet: string }[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef<(() => void) | null>(null);
  const oldestCreatedRef = useRef<string | null>(null);
  const isPrependingRef = useRef(false);
  const prevHeightRef = useRef(0);

  // Load buddy profile
  useEffect(() => {
    buddyApi
      .getProfile()
      .then(setProfile)
      .catch(() => {});
  }, []);

  useEffect(() => {
    statsApi
      .get()
      .then((s) => {
        setNoteCount(s.notes);
        setMemoryCount(s.memories);
      })
      .catch(() => {});
  }, []);

  // Keep view pinned to the newest message; preserve position when older
  // history is prepended on scroll-up.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (isPrependingRef.current) {
      el.scrollTop = el.scrollHeight - prevHeightRef.current;
      isPrependingRef.current = false;
      return;
    }
    el.scrollTop = el.scrollHeight;
  }, [messages, isGenerating]);

  const sendMessage = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || isGenerating) return;

      const userMsg: Message = {
        id: `u_${Date.now()}`,
        role: 'user',
        text: trimmed,
        time: nowTime(),
      };

      const assistantId = `a_${Date.now()}`;
      let fullText = '';
      const startTime = Date.now();

      // Placeholder shown immediately so the "preparing context…" text lives
      // inside the assistant bubble until the first chunk arrives.
      setMessages((prev) => [
        ...prev,
        userMsg,
        { id: assistantId, role: 'assistant', text: '', time: nowTime() },
      ]);
      setInput('');
      setIsGenerating(true);
      setIsPreparing(true);
      setElapsedSeconds(0);
      setSnippets([]);

      // Tick elapsed seconds every second while generating
      const elapsedInterval = setInterval(() => {
        setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);

      const { cancel } = chatApi.chatStream(
        { query: trimmed, mode },
        {
          onChunk: (chunk) => {
            if (isPreparing) setIsPreparing(false);
            fullText += chunk;
            setMessages((prev) => {
              const existing = prev.find((m) => m.id === assistantId);
              if (existing) {
                return prev.map((m) => (m.id === assistantId ? { ...m, text: fullText } : m));
              }
              return [
                ...prev,
                {
                  id: assistantId,
                  role: 'assistant',
                  text: fullText,
                  time: nowTime(),
                },
              ];
            });
          },
          onCitations: (event) => {
            if (event.citations.length > 0) {
              const chips = event.citations.map((c) => `note: ${c.title || c.note_id}`);
              setMessages((prev) => {
                const existing = prev.find((m) => m.id === assistantId);
                if (existing) {
                  return prev.map((m) => (m.id === assistantId ? { ...m, chips } : m));
                }
                return [
                  ...prev,
                  {
                    id: assistantId,
                    role: 'assistant',
                    text: fullText,
                    time: nowTime(),
                    chips,
                  },
                ];
              });
            }
          },
          onSnippet: (snippet) => {
            setSnippets((prev) => [...prev, snippet]);
          },
          onDone: () => {
            clearInterval(elapsedInterval);
            setMessages((prev) =>
              prev.map((m) => {
                if (m.id !== assistantId) return m;
                // Drop placeholder with no content at all
                if (m.text === '' && !m.chips?.length) return null;
                // Citations arrived but no answer — show error
                if (m.text === '') return { ...m, text: 'No response received. The backend may be busy. Try again.' };
                return m;
              }).filter(Boolean) as Message[]
            );
            setIsGenerating(false);
            setIsPreparing(false);
            setElapsedSeconds(0);
            setSnippets([]);
            cancelRef.current = null;
          },
          onError: (err) => {
            clearInterval(elapsedInterval);
            const errText = `Error: ${err.message}`;
            setMessages((prev) => {
              const existing = prev.find((m) => m.id === assistantId);
              if (existing) {
                return prev.map((m) => (m.id === assistantId ? { ...m, text: errText } : m));
              }
              return [
                ...prev,
                { id: assistantId, role: 'assistant', text: errText, time: nowTime() },
              ];
            });
            setIsGenerating(false);
            setIsPreparing(false);
            setElapsedSeconds(0);
            setSnippets([]);
            cancelRef.current = null;
          },
        },
      );

      cancelRef.current = cancel;
    },
    [mode, isGenerating, isPreparing],
  );

  // Initial history load: fetch the newest page, render chronologically,
  // then (if arriving with ?q=) send that query — order matters so the
  // history load does not overwrite the freshly-sent message.
  //
  // After load, if any messages are in-progress (mid-generation reload),
  // show the preparing state. The WebSocket `chat.message_updated` event
  // will resolve them in-place — no polling needed.
  useEffect(() => {
    let cancelled = false;

    conversationsApi
      .listMessages({ limit: PAGE_SIZE })
      .then((rows) => {
        if (cancelled) return;
        const chronological = rows.slice().reverse();
        const msgs = chronological.map(convToMessage);
        setMessages(msgs);
        oldestCreatedRef.current = rows.length ? rows[rows.length - 1].created : null;
        setHasMore(rows.length === PAGE_SIZE);

        // If any messages are in-progress (mid-generation reload), show
        // the preparing state. The WebSocket will resolve them.
        if (msgs.some((m) => m.inProgress)) {
          setIsGenerating(true);
          setIsPreparing(true);
        }

        if (initialQuery) sendMessage(initialQuery);
      })
      .catch(() => {
        if (!cancelled && initialQuery) sendMessage(initialQuery);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Listen for WebSocket `chat.message_updated` events — the backend
  // broadcasts these when a placeholder is promoted to final content.
  // This resolves in-progress messages after a mid-generation reload
  // without polling.
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as {
        message_id: string;
        content: string;
        citations?: { note_id: string; title?: string; snippet?: string }[];
        sub_questions?: string[];
        in_progress: boolean;
      };
      setMessages((prev) => {
        const updated = prev.map((m) => {
          if (m.id !== detail.message_id) return m;
          const chips = (detail.citations || []).map(
            (c) => `note: ${c.title || c.note_id}`,
          );
          return {
            ...m,
            text: detail.content,
            inProgress: detail.in_progress,
            ...(chips.length > 0 ? { chips } : {}),
          };
        });
        // If no more in-progress messages, exit preparing state.
        if (!updated.some((m) => m.inProgress)) {
          setIsGenerating(false);
          setIsPreparing(false);
        }
        return updated;
      });
    };
    window.addEventListener('aikioku:message_updated', handler);
    return () => window.removeEventListener('aikioku:message_updated', handler);
  }, []);

  const loadOlder = useCallback(() => {
    const cursor = oldestCreatedRef.current;
    if (!hasMore || loadingOlder || !cursor) return;
    setLoadingOlder(true);
    conversationsApi
      .listMessages({ limit: PAGE_SIZE, before: cursor })
      .then((rows) => {
        if (rows.length > 0) {
          const older = rows.slice().reverse().map(convToMessage);
          const el = scrollRef.current;
          isPrependingRef.current = true;
          prevHeightRef.current = el ? el.scrollHeight : 0;
          setMessages((prev) => [...older, ...prev]);
          oldestCreatedRef.current = rows[rows.length - 1].created;
        }
        setHasMore(rows.length === PAGE_SIZE);
      })
      .catch(() => {})
      .finally(() => setLoadingOlder(false));
  }, [hasMore, loadingOlder]);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    if (e.currentTarget.scrollTop < LOAD_OLDER_THRESHOLD_PX) loadOlder();
  };

  const handleStop = () => {
    cancelRef.current?.();
    chatApi.stopGeneration().catch(() => {});
    setIsGenerating(false);
    setIsPreparing(false);
    setElapsedSeconds(0);
    setSnippets([]);
    cancelRef.current = null;
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const name = profile?.name || 'KIO';
  const stateText = isGenerating ? 'thinking' : 'listening';

  return (
    <div className="flex min-h-0 flex-1 justify-center overflow-hidden">
      <div className="flex min-h-0 w-full max-w-[1280px] flex-col">
        {/* Header */}
        <div
          className="flex-none px-4 pt-6 pb-4 md:px-6 lg:px-8"
          style={{ display: 'flex', alignItems: 'center', gap: 13 }}
        >
          <div
            style={{
              position: 'relative',
              width: 36,
              height: 36,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <div
              style={{
                position: 'absolute',
                width: 36,
                height: 36,
                borderRadius: '50%',
                background:
                  'radial-gradient(circle at 38% 32%, rgba(184,115,51,0.6), rgba(52,214,196,0.2) 70%, transparent)',
                animation: 'auraPulse 4s ease-in-out infinite',
              }}
            />
            <div
              style={{
                width: 17,
                height: 17,
                borderRadius: '50%',
                background: 'radial-gradient(circle at 36% 30%, #ffe9cf, #B87333 55%, #2aa8b8)',
                boxShadow: '0 0 12px rgba(184,115,51,0.6)',
              }}
            />
          </div>
          <div style={{ flex: 1 }}>
            <div
              style={{
                font: '500 14px/1 "JetBrains Mono"',
                letterSpacing: '0.08em',
                color: '#eef0f3',
              }}
            >
              {name}
            </div>
            <div style={{ font: '400 10px/1 "JetBrains Mono"', color: '#6b707a', marginTop: 6 }}>
              {stateText} · drawing from {noteCount} notes &amp; {memoryCount} memories
            </div>
          </div>
          <div
            style={{
              display: 'flex',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: 9,
              overflow: 'hidden',
            }}
          >
            <span
              onClick={() => setMode('simple')}
              style={{
                padding: '7px 13px',
                font: '500 9px/1 "JetBrains Mono"',
                letterSpacing: '0.1em',
                cursor: 'pointer',
                transition: 'all .2s',
                background: mode === 'simple' ? 'rgba(95,211,224,0.12)' : 'transparent',
                color: mode === 'simple' ? '#5fd3e0' : '#5a5f69',
              }}
            >
              DIRECT
            </span>
            <span
              onClick={() => setMode('multi_hop')}
              style={{
                padding: '7px 13px',
                font: '500 9px/1 "JetBrains Mono"',
                letterSpacing: '0.1em',
                cursor: 'pointer',
                transition: 'all .2s',
                background: mode === 'multi_hop' ? 'rgba(95,211,224,0.12)' : 'transparent',
                color: mode === 'multi_hop' ? '#5fd3e0' : '#5a5f69',
              }}
            >
              IN DEPTH
            </span>
          </div>
        </div>

        {/* Messages */}
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="min-h-0 w-full flex-1 overflow-y-auto px-4 py-3.5 md:px-6 lg:px-8"
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 18,
          }}
        >
          {loadingOlder && (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '4px 0' }}>
              <span style={{ font: '400 11px/1 "JetBrains Mono"', color: '#6b707a' }}>
                loading earlier…
              </span>
            </div>
          )}

          {messages.length === 0 && !isGenerating && (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '60px 0' }}>
              <span style={{ font: '400 13px/1 "JetBrains Mono"', color: '#6b707a' }}>
                Write KIO a message...
              </span>
            </div>
          )}

          {messages.map((m) => {
            const isUser = m.role === 'user';
            return (
              <div
                key={m.id}
                style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start' }}
              >
                <div style={{ maxWidth: '74%' }}>
                  <div
                    style={{
                      padding: '14px 17px',
                      borderRadius: isUser ? '15px 15px 4px 15px' : '15px 15px 15px 4px',
                      background: isUser ? 'rgba(184,115,51,0.08)' : 'rgba(20,24,31,0.7)',
                      border: `1px solid ${isUser ? 'rgba(184,115,51,0.2)' : 'rgba(255,255,255,0.06)'}`,
                    }}
                  >
                    {!isUser && (m.inProgress || (m.text === '' && isGenerating)) ? (
                      <div>
                        <div
                          style={{
                            font: '400 13px/1.65 "JetBrains Mono"',
                            color: '#6b707a',
                            fontStyle: 'italic',
                          }}
                        >
                          preparing context…
                        </div>
                        {(m.inProgress || elapsedSeconds > 5) && (
                          <div
                            style={{
                              font: '400 9px/1 "JetBrains Mono"',
                              color: '#4a4f59',
                              marginTop: 8,
                            }}
                          >
                            {elapsedSeconds}s elapsed
                          </div>
                        )}
                      </div>
                    ) : isUser ? (
                      <div
                        style={{
                          font: '400 13px/1.65 "JetBrains Mono"',
                          color: '#dde0e6',
                          whiteSpace: 'pre-wrap',
                        }}
                      >
                        {m.text}
                      </div>
                    ) : (
                      <MarkdownPreview content={m.text} />
                    )}
                  </div>
                  {m.chips && m.chips.length > 0 && (
                    <div
                      style={{
                        display: 'flex',
                        gap: 6,
                        marginTop: 8,
                        flexWrap: 'nowrap',
                        overflow: 'hidden',
                        alignItems: 'center',
                      }}
                    >
                      {m.chips.slice(0, 2).map((ch, i) => (
                        <span
                          key={i}
                          title={ch}
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 5,
                            minWidth: 0,
                            maxWidth: 170,
                            padding: '4px 9px',
                            borderRadius: 7,
                            background: 'rgba(52,214,196,0.06)',
                            border: '1px solid rgba(52,214,196,0.15)',
                            font: '400 8px/1 "JetBrains Mono"',
                            letterSpacing: '0.05em',
                            color: '#4cc9bb',
                          }}
                        >
                          <svg
                            width="9"
                            height="9"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            style={{ flexShrink: 0 }}
                          >
                            <path d="M12 3a4 4 0 0 0-4 4 3.5 3.5 0 0 0-1 6.8V18a3 3 0 0 0 5 2 3 3 0 0 0 5-2v-4.2A3.5 3.5 0 0 0 16 7a4 4 0 0 0-4-4z" />
                          </svg>
                          <span
                            style={{
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {ch}
                          </span>
                        </span>
                      ))}
                      {m.chips.length > 2 && (
                        <span
                          style={{
                            flexShrink: 0,
                            padding: '4px 9px',
                            borderRadius: 7,
                            background: 'rgba(52,214,196,0.06)',
                            border: '1px solid rgba(52,214,196,0.15)',
                            font: '400 8px/1 "JetBrains Mono"',
                            letterSpacing: '0.05em',
                            color: '#4cc9bb',
                          }}
                        >
                          +{m.chips.length - 2}
                        </span>
                      )}
                    </div>
                  )}
                  {!isUser && snippets.length > 0 && m.text === '' && (
                    <div
                      style={{
                        display: 'flex',
                        gap: 6,
                        marginTop: 8,
                        flexWrap: 'wrap',
                        alignItems: 'center',
                      }}
                    >
                      {snippets.map((s, i) => (
                        <span
                          key={s.note_id}
                          title={s.snippet}
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 5,
                            minWidth: 0,
                            maxWidth: 200,
                            padding: '4px 9px',
                            borderRadius: 7,
                            background: 'rgba(184,115,51,0.06)',
                            border: '1px solid rgba(184,115,51,0.15)',
                            font: '400 8px/1 "JetBrains Mono"',
                            letterSpacing: '0.05em',
                            color: '#b87333',
                          }}
                        >
                          <svg
                            width="9"
                            height="9"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            style={{ flexShrink: 0 }}
                          >
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                            <polyline points="14 2 14 8 20 8" />
                            <line x1="16" y1="13" x2="8" y2="13" />
                            <line x1="16" y1="17" x2="8" y2="17" />
                            <polyline points="10 9 9 9 8 9" />
                          </svg>
                          <span
                            style={{
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {s.title || s.note_id}
                          </span>
                        </span>
                      ))}
                    </div>
                  )}
                  <div
                    style={{
                      font: '400 8px/1 "JetBrains Mono"',
                      color: '#4a4f59',
                      marginTop: 8,
                      textAlign: isUser ? 'right' : 'left',
                    }}
                  >
                    {m.time}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Input area */}
        <div className="flex-none px-4 pb-6 pt-3.5 md:px-6 lg:px-8">
          {!isGenerating && messages.length > 0 && messages[messages.length - 1].inProgress && (
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 13 }}>
              <button
                onClick={handleStop}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '8px 18px',
                  borderRadius: 9,
                  border: '1px solid rgba(248,113,113,0.4)',
                  background: 'rgba(248,113,113,0.08)',
                  color: '#f87171',
                  font: '600 10px/1 "JetBrains Mono"',
                  letterSpacing: '0.16em',
                  cursor: 'pointer',
                  transition: 'all .2s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(248,113,113,0.16)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(248,113,113,0.08)';
                }}
              >
                <span style={{ width: 8, height: 8, borderRadius: 2, background: '#f87171' }} />
                STOP
              </button>
            </div>
          )}
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Write KIO…"
              disabled={isGenerating}
              style={{
                flex: 1,
                padding: '15px 18px',
                borderRadius: 14,
                border: '1px solid rgba(255,255,255,0.08)',
                background: 'rgba(255,255,255,0.02)',
                color: '#dde0e6',
                font: '400 13px/1 "JetBrains Mono"',
                transition: 'all .2s',
                outline: 'none',
                opacity: isGenerating ? 0.5 : 1,
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = 'rgba(95,211,224,0.4)';
                e.currentTarget.style.background = 'rgba(95,211,224,0.03)';
                e.currentTarget.style.boxShadow = '0 0 22px rgba(95,211,224,0.08)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
                e.currentTarget.style.background = 'rgba(255,255,255,0.02)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            />
            <button
              onClick={() => sendMessage(input)}
              disabled={isGenerating || !input.trim()}
              style={{
                width: 50,
                height: 50,
                borderRadius: 14,
                border: 'none',
                background: 'radial-gradient(circle at 38% 32%, #ffd9a8, #B87333)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: isGenerating ? 'not-allowed' : 'pointer',
                transition: 'transform .2s',
                opacity: isGenerating || !input.trim() ? 0.5 : 1,
              }}
              onMouseEnter={(e) => {
                if (!isGenerating && input.trim()) e.currentTarget.style.transform = 'scale(1.06)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'scale(1)';
              }}
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#2a1d0c"
                strokeWidth="2.2"
              >
                <path d="M5 12h13M12 5l7 7-7 7" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
