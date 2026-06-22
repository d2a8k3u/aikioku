'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { buddyApi } from '@/lib/api';
import type { BuddyGreeting, BuddyCards, BuddyCard } from '@/types';

// ── KIO Orb Canvas ──────────────────────────────────────

function KioOrb() {
  useEffect(() => {
    const cv = document.getElementById('kio-orb') as HTMLCanvasElement | null;
    if (!cv) return;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const S = 210;
    cv.width = S * dpr;
    cv.height = S * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cx = S / 2,
      cy = S / 2,
      R = 50;
    const blobs = [
      { a: 0, sp: 0.6, r: 22, c: [184, 115, 51] },
      { a: 2.1, sp: -0.42, r: 26, c: [52, 214, 196] },
      { a: 4.2, sp: 0.82, r: 17, c: [127, 227, 238] },
    ];
    const parts: { a: number; rad: number; sp: number; sz: number; tilt: number }[] = [];
    for (let i = 0; i < 16; i++) {
      parts.push({
        a: Math.random() * 7,
        rad: 64 + Math.random() * 30,
        sp: (0.18 + Math.random() * 0.45) * (Math.random() < 0.5 ? -1 : 1),
        sz: Math.random() * 1.6 + 0.7,
        tilt: 0.45 + Math.random() * 0.45,
      });
    }
    let t = 0;
    let raf: number;
    const draw = () => {
      t += 0.012;
      ctx.clearRect(0, 0, S, S);
      const br = 0.5 + 0.5 * Math.sin(t * 0.8);
      const g = ctx.createRadialGradient(cx, cy, 8, cx, cy, 100);
      g.addColorStop(0, `rgba(184,115,51,${0.18 + 0.1 * br})`);
      g.addColorStop(0.5, 'rgba(52,214,196,0.05)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(cx, cy, 100, 0, 7);
      ctx.fill();
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, 7);
      ctx.clip();
      const cg = ctx.createRadialGradient(cx - 14, cy - 16, 4, cx, cy, R + 6);
      cg.addColorStop(0, 'rgba(38,29,18,0.92)');
      cg.addColorStop(1, 'rgba(7,11,15,0.96)');
      ctx.fillStyle = cg;
      ctx.fillRect(cx - R, cy - R, 2 * R, 2 * R);
      blobs.forEach((b) => {
        const ang = b.a + t * b.sp;
        const bx = cx + Math.cos(ang) * 17;
        const by = cy + Math.sin(ang * 1.2) * 17;
        const rr = b.r + 8 * br;
        const bg = ctx.createRadialGradient(bx, by, 0, bx, by, rr);
        bg.addColorStop(0, `rgba(${b.c.join(',')},0.7)`);
        bg.addColorStop(1, `rgba(${b.c.join(',')},0)`);
        ctx.fillStyle = bg;
        ctx.beginPath();
        ctx.arc(bx, by, rr, 0, 7);
        ctx.fill();
      });
      const hg = ctx.createRadialGradient(cx - 17, cy - 19, 0, cx - 17, cy - 19, 24);
      hg.addColorStop(0, 'rgba(255,242,224,0.55)');
      hg.addColorStop(1, 'rgba(255,242,224,0)');
      ctx.fillStyle = hg;
      ctx.beginPath();
      ctx.arc(cx - 17, cy - 19, 24, 0, 7);
      ctx.fill();
      ctx.restore();
      ctx.strokeStyle = `rgba(184,115,51,${0.3 + 0.2 * br})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, 7);
      ctx.stroke();
      ctx.strokeStyle = 'rgba(95,211,224,0.28)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 6]);
      ctx.beginPath();
      ctx.arc(cx, cy, R + 11, t * 0.5, t * 0.5 + Math.PI * 1.55);
      ctx.stroke();
      ctx.setLineDash([]);
      parts.forEach((p) => {
        const ang = p.a + t * p.sp;
        const px = cx + Math.cos(ang) * p.rad;
        const py = cy + Math.sin(ang) * p.rad * p.tilt;
        const fade = 0.4 + 0.5 * (0.5 + 0.5 * Math.sin(ang * 2));
        ctx.fillStyle = `rgba(127,227,238,${fade})`;
        ctx.beginPath();
        ctx.arc(px, py, p.sz, 0, 7);
        ctx.fill();
      });
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, []);

  return <canvas id="kio-orb" style={{ width: 210, height: 210 }} />;
}

// ── Card Component ──────────────────────────────────────

function HomeCard({ card, onClick }: { card: BuddyCard; onClick: () => void }) {
  return (
    <div
      onClick={onClick}
      style={{
        height: '100%',
        minWidth: 0,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        padding: 18,
        borderRadius: 15,
        border: '1px solid rgba(255,255,255,0.07)',
        background: 'rgba(13,17,23,0.45)',
        cursor: 'pointer',
        transition: 'all .25s',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.16)';
        e.currentTarget.style.transform = 'translateY(-2px)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)';
        e.currentTarget.style.transform = 'translateY(0)';
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 11 }}>
        <span
          style={{
            width: 22,
            height: 22,
            borderRadius: 7,
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: `rgba(${card.rgb},0.1)`,
            border: `1px solid rgba(${card.rgb},0.2)`,
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: `rgb(${card.rgb})`,
              boxShadow: `0 0 6px rgb(${card.rgb})`,
            }}
          />
        </span>
        <span
          style={{ font: '500 9px/1 "JetBrains Mono"', letterSpacing: '0.16em', color: '#6b707a' }}
        >
          {card.kicker}
        </span>
      </div>
      <div
        style={{
          font: '500 13px/1.4 "JetBrains Mono"',
          color: '#dfe2e7',
          marginBottom: 8,
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {card.title}
      </div>
      <p
        className="line-clamp-3"
        style={{
          flex: 1,
          margin: '0 0 13px',
          font: '400 11px/1.55 "JetBrains Mono"',
          color: '#7d828c',
          overflowWrap: 'anywhere',
        }}
      >
        {card.body}
      </p>
      <span
        style={{
          font: '500 10px/1 "JetBrains Mono"',
          letterSpacing: '0.08em',
          color: `rgb(${card.rgb})`,
        }}
      >
        {card.action} →
      </span>
    </div>
  );
}

// ── Main Page ───────────────────────────────────────────

export default function DashboardPage() {
  const [greeting, setGreeting] = useState<BuddyGreeting | null>(null);
  const [cards, setCards] = useState<BuddyCards | null>(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [g, c] = await Promise.all([buddyApi.getGreeting(), buddyApi.getCards()]);
      setGreeting(g);
      setCards(c);
    } catch {
      // degrade gracefully
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleSend = () => {
    const text = input.trim();
    if (!text) return;
    router.push(`/chat?q=${encodeURIComponent(text)}`);
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSend();
    }
  };

  const prompts = [
    { label: 'What do I have to do today?', q: 'What do I have to do today?' },
    { label: 'Remind me of my projects', q: 'Remind me of my projects' },
    { label: 'What do you remember about me?', q: 'What do you remember about me?' },
  ];

  const goMap: Record<string, string> = {
    talk: '/chat',
    thoughts: '/notes',
    recall: '/review',
  };

  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        overflowY: 'auto',
        display: 'flex',
        justifyContent: 'center',
        padding: '40px clamp(16px, 4vw, 32px) 60px',
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
        {/* Hero stays a centered narrow column; cards/about-you below use the full xl width. */}
        <div
          style={{
            width: '100%',
            maxWidth: 680,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          {/* Orb */}
          <div
            style={{
              position: 'relative',
              width: 210,
              height: 210,
              marginBottom: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <KioOrb />
          </div>

          {/* Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 26 }}>
            <span
              style={{
                width: 5,
                height: 5,
                borderRadius: '50%',
                background: '#34d6c4',
                boxShadow: '0 0 8px #34d6c4',
                animation: 'pulse-subtle 2.6s ease-in-out infinite',
              }}
            />
            <span
              style={{
                font: '500 9px/1 "JetBrains Mono"',
                letterSpacing: '0.22em',
                color: '#6b707a',
              }}
            >
              KIO · {loading ? '...' : 'is here'}
            </span>
          </div>

          {/* Greeting */}
          <h1
            style={{
              margin: 0,
              textAlign: 'center',
              font: '300 30px/1.25 "JetBrains Mono"',
              letterSpacing: '0.01em',
              color: '#eef0f3',
            }}
          >
            {greeting?.greeting || 'Welcome.'}
          </h1>
          <p
            style={{
              margin: '16px 0 32px',
              textAlign: 'center',
              maxWidth: 480,
              font: '400 13px/1.7 "JetBrains Mono"',
              color: '#8a8f99',
            }}
          >
            {greeting?.buddy_line || "I'm here for you."}
          </p>

          {/* Input */}
          <div style={{ width: '100%', position: 'relative', marginBottom: 14 }}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="What's on your mind?"
              style={{
                width: '100%',
                padding: '18px 58px 18px 22px',
                borderRadius: 16,
                border: '1px solid rgba(255,255,255,0.09)',
                background: 'rgba(13,17,23,0.6)',
                backdropFilter: 'blur(10px)',
                color: '#dde0e6',
                font: '400 14px/1 "JetBrains Mono"',
                transition: 'all .25s',
                outline: 'none',
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = 'rgba(184,115,51,0.4)';
                e.currentTarget.style.boxShadow = '0 0 30px rgba(184,115,51,0.1)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = 'rgba(255,255,255,0.09)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            />
            <button
              onClick={handleSend}
              style={{
                position: 'absolute',
                right: 9,
                top: '50%',
                transform: 'translateY(-50%)',
                width: 40,
                height: 40,
                borderRadius: 11,
                border: 'none',
                background: 'radial-gradient(circle at 38% 32%, #7fe3ee, #2aa8b8)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                transition: 'transform .2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-50%) scale(1.08)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(-50%) scale(1)';
              }}
            >
              <svg
                width="17"
                height="17"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#06242a"
                strokeWidth="2.2"
              >
                <path d="M5 12h13M12 5l7 7-7 7" />
              </svg>
            </button>
          </div>

          {/* Quick prompts */}
          <div
            style={{
              display: 'flex',
              gap: 8,
              flexWrap: 'wrap',
              justifyContent: 'center',
              marginBottom: 40,
            }}
          >
            {prompts.map((p) => (
              <span
                key={p.label}
                onClick={() => router.push(`/chat?q=${encodeURIComponent(p.q)}`)}
                style={{
                  padding: '7px 13px',
                  borderRadius: 18,
                  border: '1px solid rgba(255,255,255,0.08)',
                  background: 'rgba(255,255,255,0.02)',
                  font: '400 10px/1 "JetBrains Mono"',
                  color: '#8a8f99',
                  cursor: 'pointer',
                  transition: 'all .2s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(95,211,224,0.3)';
                  e.currentTarget.style.color = '#9fe3ec';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
                  e.currentTarget.style.color = '#8a8f99';
                }}
              >
                {p.label}
              </span>
            ))}
          </div>
        </div>

        {/* Cards — column count tracks the cards area width (container query), not the viewport. */}
        {cards && cards.cards.length > 0 && (
          <div className="@container w-full">
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                marginBottom: 16,
              }}
            >
              <span
                style={{
                  width: 5,
                  height: 5,
                  borderRadius: '50%',
                  background: '#B87333',
                  boxShadow: '0 0 8px #B87333',
                }}
              />
              <span
                style={{
                  font: '500 10px/1 "JetBrains Mono"',
                  letterSpacing: '0.2em',
                  color: '#6b707a',
                }}
              >
                KIO PREPARED FOR YOU
              </span>
            </div>
            <div className="flex flex-wrap justify-center gap-[14px] mb-[34px]">
              {cards.cards.map((card, i) => (
                <div
                  key={i}
                  className="min-w-0 grow-0 shrink-0 basis-full @xl:basis-[calc(50%_-_7px)] @4xl:basis-[calc((100%_-_28px)/3)] @6xl:basis-[calc((100%_-_42px)/4)]"
                >
                  <HomeCard
                    card={card}
                    onClick={() => {
                      const path = goMap[card.go] || '/';
                      router.push(path);
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* About you */}
        {greeting && greeting.about_you.length > 0 && (
          <div
            style={{
              width: '100%',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: 15,
              padding: '18px 20px',
              background: 'rgba(255,255,255,0.012)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 14 }}>
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#5fd3e0"
                strokeWidth="1.6"
              >
                <path d="M12 3a4 4 0 0 0-4 4 3.5 3.5 0 0 0-1 6.8V18a3 3 0 0 0 5 2 3 3 0 0 0 5-2v-4.2A3.5 3.5 0 0 0 16 7a4 4 0 0 0-4-4z" />
              </svg>
              <span style={{ font: '400 11px/1 "JetBrains Mono"', color: '#8a8f99' }}>
                What KIO remembers about you
              </span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {greeting.about_you.map((fact, i) => (
                <span
                  key={i}
                  style={{
                    padding: '7px 12px',
                    borderRadius: 9,
                    background: 'rgba(52,214,196,0.06)',
                    border: '1px solid rgba(52,214,196,0.14)',
                    font: '400 11px/1.3 "JetBrains Mono"',
                    color: '#aeb3bc',
                  }}
                >
                  {fact}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
