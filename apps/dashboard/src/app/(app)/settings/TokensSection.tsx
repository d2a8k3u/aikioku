'use client';

import { useEffect, useState } from 'react';
import {
  HudPanel,
  HudButton,
  HudInput,
  HudBadge,
  HudSpinner,
  HudErrorBanner,
} from '@/components/hud';
import { tokensApi } from '@/lib/api';
import type { AccessToken } from '@/types';

/* ─── Inline SVG icons ─── */
function PlusIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
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

function TrashIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 6h18" />
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}

function CopyIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function AlertTriangleIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8869';
const MCP_URL = `${API_BASE}/mcp`;

export default function TokensSection() {
  const [tokens, setTokens] = useState<AccessToken[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [name, setName] = useState('');
  const [scope, setScope] = useState<'read' | 'full'>('full');
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  async function refresh() {
    try {
      setTokens(await tokensApi.list());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleCreate() {
    if (!name.trim()) return;
    setCreating(true);
    setError(null);
    try {
      const res = await tokensApi.create(name.trim(), scope);
      setCreated(res.token);
      setCopied(false);
      setName('');
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCreating(false);
    }
  }

  async function handleRevoke(id: string) {
    try {
      await tokensApi.remove(id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function copy(text: string) {
    navigator.clipboard?.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <HudPanel title="API Access (MCP)" glow className="p-4">
      <p className="text-xs font-mono text-aiki-text-tertiary mb-4">
        Generate a token to let other apps (Claude Desktop, Cursor, your own agents) use this brain
        over MCP. Endpoint:{' '}
        <code className="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] text-aiki-accent">
          {MCP_URL}
        </code>
      </p>

      {error && <HudErrorBanner message={error} onRetry={refresh} />}

      {/* Freshly created token — shown exactly once */}
      {created && (
        <div className="mb-4 rounded-lg border border-aiki-accent/40 bg-aiki-accent/10 p-3">
          <div className="flex items-center gap-2 mb-2 text-xs font-sans font-medium text-aiki-accent">
            <AlertTriangleIcon className="h-3.5 w-3.5" />
            Copy this token now — it won&apos;t be shown again.
          </div>
          <div className="flex items-center gap-2">
            <code className="flex-1 overflow-x-auto rounded bg-white/[0.04] px-3 py-2 text-[10px] font-mono text-aiki-text-secondary">
              {created}
            </code>
            <HudButton size="sm" variant="default" onClick={() => copy(created)}>
              {copied ? <CheckIcon className="h-3 w-3" /> : <CopyIcon className="h-3 w-3" />}
              {copied ? 'Copied' : 'Copy'}
            </HudButton>
          </div>
          <button
            onClick={() => setCreated(null)}
            className="mt-2 text-[10px] text-aiki-text-tertiary/50 hover:text-aiki-text-tertiary transition-colors"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Create form */}
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <div className="flex-1 min-w-[180px]">
          <label className="block text-[10px] font-sans font-medium text-aiki-text-tertiary mb-1.5">
            Token name
          </label>
          <HudInput
            value={name}
            placeholder="e.g. Claude Desktop"
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <div className="min-w-[140px]">
          <label className="block text-[10px] font-sans font-medium text-aiki-text-tertiary mb-1.5">
            Scope
          </label>
          <select
            value={scope}
            onChange={(e) => setScope(e.target.value as 'read' | 'full')}
            className="w-full bg-white/[0.03] backdrop-blur-sm border border-white/[0.08] rounded-lg px-3 py-2 text-sm font-mono text-aiki-text-secondary focus:border-aiki-accent/50 focus:shadow-[0_0_16px_rgba(184,115,51,0.1)] focus:outline-none transition-all duration-300"
          >
            <option value="full">Full (read + write)</option>
            <option value="read">Read-only</option>
          </select>
        </div>
        <HudButton
          size="md"
          variant="default"
          onClick={handleCreate}
          disabled={creating || !name.trim()}
        >
          {creating ? <HudSpinner size={14} /> : <PlusIcon className="h-3.5 w-3.5" />}
          Generate
        </HudButton>
      </div>

      {/* Token list */}
      {loading ? (
        <div className="flex items-center gap-2 text-xs text-aiki-text-tertiary">
          <HudSpinner size={14} /> Loading tokens…
        </div>
      ) : tokens.length === 0 ? (
        <p className="text-xs font-mono text-aiki-text-tertiary/50">No tokens yet.</p>
      ) : (
        <div className="space-y-1.5">
          {tokens.map((t) => (
            <div
              key={t.id}
              className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="truncate text-xs font-mono text-aiki-text-secondary">
                    {t.name}
                  </span>
                  <HudBadge
                    status={t.scope === 'full' ? 'active' : 'queued'}
                    label={t.scope === 'full' ? 'Full' : 'Read'}
                  />
                </div>
                <div className="mt-0.5 text-[10px] font-mono text-aiki-text-tertiary/50">
                  <code>{t.prefix}…</code> · created {new Date(t.created).toLocaleDateString()}
                  {t.last_used
                    ? ` · last used ${new Date(t.last_used).toLocaleDateString()}`
                    : ' · never used'}
                </div>
              </div>
              <HudButton
                size="sm"
                variant="danger"
                onClick={() => handleRevoke(t.id)}
                title="Revoke token"
              >
                <TrashIcon className="h-3.5 w-3.5" />
              </HudButton>
            </div>
          ))}
        </div>
      )}
    </HudPanel>
  );
}
