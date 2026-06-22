'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { buddyApi, settingsApi, tokensApi } from '@/lib/api';
import { HudSpinner } from '@/components/hud/HudSpinner';
import type {
  BuddyProfile,
  AppSettings,
  AccessToken,
  ProviderModel,
  EmbeddingModel,
} from '@/types';

// ── Selection card ──────────────────────────────────────

function SelCard({
  active,
  rgb,
  onClick,
  title,
  desc,
}: {
  active: boolean;
  rgb: string;
  onClick: () => void;
  title: string;
  desc: string;
}) {
  const base: React.CSSProperties = {
    padding: 14,
    borderRadius: 11,
    cursor: 'pointer',
    transition: 'all .2s',
    border: '1px solid rgba(255,255,255,0.07)',
    background: 'rgba(255,255,255,0.015)',
  };
  const activeStyle: React.CSSProperties = {
    ...base,
    border: `1px solid rgba(${rgb},0.45)`,
    background: `rgba(${rgb},0.07)`,
    boxShadow: `0 0 18px rgba(${rgb},0.1)`,
  };
  return (
    <div onClick={onClick} style={active ? activeStyle : base}>
      <div style={{ font: '500 12px/1 "JetBrains Mono"', color: '#dfe2e7', marginBottom: 7 }}>
        {title}
      </div>
      <div style={{ font: '400 9px/1.4 "JetBrains Mono"', color: '#6b707a' }}>{desc}</div>
    </div>
  );
}

// ── Glass panel wrapper ─────────────────────────────────

function GlassPanel({
  children,
  style,
}: {
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  return (
    <div
      style={{
        background: 'rgba(12,15,21,0.5)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: 16,
        padding: 24,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function SectionDot({ color }: { color: string }) {
  return (
    <span
      style={{
        width: 5,
        height: 5,
        borderRadius: '50%',
        background: color,
        boxShadow: `0 0 8px ${color}`,
      }}
    />
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span style={{ font: '500 11px/1 "JetBrains Mono"', letterSpacing: '0.2em', color: '#9aa0aa' }}>
      {children}
    </span>
  );
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        font: '500 9px/1 "JetBrains Mono"',
        letterSpacing: '0.16em',
        color: '#6b707a',
        marginBottom: 9,
      }}
    >
      {children}
    </div>
  );
}

function TextInput({
  value,
  onChange,
  readOnly,
  placeholder,
  type,
  list,
}: {
  value: string;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  readOnly?: boolean;
  placeholder?: string;
  type?: string;
  list?: string;
}) {
  return (
    <input
      value={value}
      onChange={onChange}
      readOnly={readOnly}
      placeholder={placeholder}
      type={type}
      list={list}
      style={{
        width: '100%',
        padding: '13px 15px',
        borderRadius: 10,
        border: '1px solid rgba(255,255,255,0.08)',
        background: 'rgba(255,255,255,0.02)',
        color: '#cfd2d8',
        font: '400 12px/1 "JetBrains Mono"',
        marginBottom: 20,
        outline: 'none',
        transition: 'all .2s',
      }}
      onFocus={(e) => {
        e.currentTarget.style.borderColor = 'rgba(184,115,51,0.4)';
      }}
      onBlur={(e) => {
        e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)';
      }}
    />
  );
}

function ToggleRow({
  label,
  desc,
  value,
  onChange,
}: {
  label: string;
  desc: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div
      onClick={() => onChange(!value)}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 14px',
        borderRadius: 10,
        border: '1px solid rgba(255,255,255,0.06)',
        background: 'rgba(255,255,255,0.015)',
        cursor: 'pointer',
        transition: 'all .2s',
        marginBottom: 10,
      }}
    >
      <div>
        <div style={{ font: '500 11px/1 "JetBrains Mono"', color: '#cfd2d8', marginBottom: 4 }}>
          {label}
        </div>
        <div style={{ font: '400 9px/1.4 "JetBrains Mono"', color: '#6b707a' }}>{desc}</div>
      </div>
      <div
        style={{
          width: 40,
          height: 22,
          borderRadius: 11,
          background: value ? 'rgba(52,214,196,0.3)' : 'rgba(255,255,255,0.08)',
          border: `1px solid ${value ? 'rgba(52,214,196,0.5)' : 'rgba(255,255,255,0.12)'}`,
          position: 'relative',
          transition: 'all .2s',
        }}
      >
        <div
          style={{
            width: 16,
            height: 16,
            borderRadius: '50%',
            background: value ? '#34d6c4' : '#6b707a',
            position: 'absolute',
            top: 2,
            left: value ? 21 : 2,
            transition: 'all .2s',
          }}
        />
      </div>
    </div>
  );
}

// ── Icons ────────────────────────────────────────────────

function KeyIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4Z" />
      <circle cx="16.5" cy="7.5" r=".5" fill="currentColor" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg
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

function TrashIcon() {
  return (
    <svg
      width="14"
      height="14"
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

function CopyIcon() {
  return (
    <svg
      width="14"
      height="14"
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

function CheckIcon() {
  return (
    <svg
      width="14"
      height="14"
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

function AlertIcon() {
  return (
    <svg
      width="14"
      height="14"
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

// ── LLM provider draft ──────────────────────────────────

interface LlmDraft {
  provider: string;
  model: string;
  ollamaUrl: string;
  openrouterUrl: string;
  apiKey: string; // new value to set; empty = keep existing secret
  dailyBudget: string; // USD/day spend cap; processing pauses when reached
}

const DEFAULT_DAILY_BUDGET = '5';

const llmFromSettings = (s: AppSettings): LlmDraft => ({
  provider: s.llm_provider,
  model: s.llm_model ?? '',
  ollamaUrl: s.ollama_base_url ?? '',
  openrouterUrl: s.openrouter_base_url ?? '',
  apiKey: '',
  dailyBudget: String(s.llm_daily_budget_usd ?? DEFAULT_DAILY_BUDGET),
});

// Secret key each provider authenticates with (local Ollama needs none).
const apiKeySecretFor = (provider: string): string | null =>
  provider === 'openrouter'
    ? 'openrouter_api_key'
    : provider === 'ollama_remote'
      ? 'ollama_api_key'
      : null;

// ── Embedding draft ─────────────────────────────────────

interface EmbDraft {
  provider: string; // ollama | openai | huggingface
  ollamaModel: string;
  openaiModel: string;
  hfModel: string;
  ollamaUrl: string;
  openaiUrl: string;
  apiKey: string; // new value to set; empty = keep existing secret
  dimension: string;
  strict: boolean;
}

const embFromSettings = (s: AppSettings): EmbDraft => ({
  provider: s.embedding_provider || 'ollama',
  ollamaModel: s.ollama_embedding_model ?? '',
  openaiModel: s.openai_embedding_model ?? '',
  hfModel: s.hf_embedding_model ?? '',
  ollamaUrl: s.ollama_embedding_base_url ?? '',
  openaiUrl: s.openai_base_url ?? '',
  apiKey: '',
  dimension: String(s.embedding_dimension ?? 1024),
  strict: s.embedding_strict ?? true,
});

const embApiKeySecretFor = (provider: string): string | null =>
  provider === 'openai'
    ? 'openai_api_key'
    : provider === 'huggingface'
      ? 'hf_api_key'
      : provider === 'ollama_remote'
        ? 'ollama_api_key'
        : null;

const embModelOf = (d: EmbDraft): string =>
  d.provider === 'openai'
    ? d.openaiModel
    : d.provider === 'huggingface'
      ? d.hfModel
      : d.ollamaModel;

const embStoredModel = (s: AppSettings, provider: string): string =>
  provider === 'openai'
    ? (s.openai_embedding_model ?? '')
    : provider === 'huggingface'
      ? (s.hf_embedding_model ?? '')
      : (s.ollama_embedding_model ?? '');

// ── Main Page ───────────────────────────────────────────

export default function SettingsPage() {
  const [profile, setProfile] = useState<BuddyProfile>({
    name: 'KIO',
    tone: 'warm',
    lm_provider: 'remote',
  });
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // LLM provider section — local draft so changes are explicit-save (not live).
  const [llm, setLlm] = useState<LlmDraft>({
    provider: 'ollama',
    model: '',
    ollamaUrl: '',
    openrouterUrl: '',
    apiKey: '',
    dailyBudget: DEFAULT_DAILY_BUDGET,
  });
  const [savingLlm, setSavingLlm] = useState(false);
  const [llmModels, setLlmModels] = useState<ProviderModel[]>([]);
  const [llmModelsError, setLlmModelsError] = useState<string | null>(null);
  const modelFetchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const modelReqId = useRef(0); // discards out-of-order responses while typing

  // Embedding section — same explicit-save draft pattern as the LLM section.
  const [emb, setEmb] = useState<EmbDraft>({
    provider: 'ollama',
    ollamaModel: '',
    openaiModel: '',
    hfModel: '',
    ollamaUrl: '',
    openaiUrl: '',
    apiKey: '',
    dimension: '1024',
    strict: true,
  });
  const [savingEmb, setSavingEmb] = useState(false);
  const [embModels, setEmbModels] = useState<EmbeddingModel[]>([]);
  const [embModelsError, setEmbModelsError] = useState<string | null>(null);
  const embFetchTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const embReqId = useRef(0);

  // Tokens
  const [tokens, setTokens] = useState<AccessToken[]>([]);
  const [tokenName, setTokenName] = useState('');
  const [tokenScope, setTokenScope] = useState<'read' | 'full'>('full');
  const [creating, setCreating] = useState(false);
  const [createdToken, setCreatedToken] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Secrets (read-only list; keys are set per-provider in the LLM/Embedding sections)
  const [secretKeys, setSecretKeys] = useState<string[]>([]);

  const fetchData = useCallback(async () => {
    try {
      const [p, s, sec, tok] = await Promise.all([
        buddyApi.getProfile(),
        settingsApi.get(),
        settingsApi.listSecrets(),
        tokensApi.list(),
      ]);
      setProfile(p);
      setSettings(s);
      setLlm(llmFromSettings(s));
      setEmb(embFromSettings(s));
      setSecretKeys(sec.keys);
      setTokens(tok);
    } catch {
      // degrade
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const llmBaseUrl = llm.provider === 'openrouter' ? llm.openrouterUrl : llm.ollamaUrl;

  // Models are NOT pre-loaded. They are fetched as the user types in the model
  // field (search-as-you-type), debounced, using the URL/key currently typed so
  // the results reflect unsaved config.
  const searchModels = useCallback(
    (provider: string, baseUrl: string, apiKey: string, query: string) => {
      if (modelFetchTimer.current) clearTimeout(modelFetchTimer.current);
      if (!query.trim()) {
        modelReqId.current++;
        setLlmModels([]);
        setLlmModelsError(null);
        return;
      }
      modelFetchTimer.current = setTimeout(() => {
        const reqId = ++modelReqId.current;
        settingsApi
          .listModels(provider, {
            baseUrl: baseUrl || undefined,
            apiKey: apiKey || undefined,
            q: query,
          })
          .then((r) => {
            if (reqId !== modelReqId.current) return;
            setLlmModels(r.models.filter((m) => m.type === 'chat'));
            setLlmModelsError(r.error);
          })
          .catch(() => {
            if (reqId !== modelReqId.current) return;
            setLlmModels([]);
            setLlmModelsError('Could not load models');
          });
      }, 400);
    },
    [],
  );

  const resetLlmModels = () => {
    if (modelFetchTimer.current) clearTimeout(modelFetchTimer.current);
    modelReqId.current++;
    setLlmModels([]);
    setLlmModelsError(null);
  };

  const updateProfile = async (patch: Partial<BuddyProfile>) => {
    const updated = { ...profile, ...patch };
    setProfile(updated);
    try {
      await buddyApi.updateProfile(updated);
    } catch {}
  };

  const updateSetting = async (key: string, value: string | boolean | number) => {
    if (!settings) return;
    const val = typeof value === 'boolean' ? String(value) : String(value);
    setSaving(true);
    try {
      await settingsApi.update({ [key]: val } as Partial<AppSettings>);
      const s = await settingsApi.get();
      setSettings(s);
    } catch {
    } finally {
      setSaving(false);
    }
  };

  const apiKeySecret = apiKeySecretFor(llm.provider);
  const llmDirty =
    !!settings &&
    (llm.provider !== settings.llm_provider ||
      llm.model !== (settings.llm_model ?? '') ||
      (llm.provider === 'openrouter'
        ? llm.openrouterUrl !== (settings.openrouter_base_url ?? '')
        : llm.ollamaUrl !== (settings.ollama_base_url ?? '')) ||
      llm.apiKey.trim() !== '' ||
      llm.dailyBudget !== String(settings.llm_daily_budget_usd ?? DEFAULT_DAILY_BUDGET));

  const setLlmProvider = (provider: string) => {
    setLlm((d) => ({ ...d, provider, apiKey: '' }));
    resetLlmModels();
  };

  const cancelLlm = () => {
    if (!settings) return;
    setLlm(llmFromSettings(settings));
    resetLlmModels();
  };

  const saveLlm = async () => {
    if (!settings || !llmDirty || savingLlm) return;
    setSavingLlm(true);
    try {
      const payload: Partial<AppSettings> = {
        llm_provider: llm.provider,
        llm_model: llm.model,
      };
      const budget = Number(llm.dailyBudget);
      if (Number.isFinite(budget) && budget >= 0) payload.llm_daily_budget_usd = budget;
      if (llm.provider === 'openrouter') payload.openrouter_base_url = llm.openrouterUrl;
      else payload.ollama_base_url = llm.ollamaUrl;
      await settingsApi.update(payload);
      if (apiKeySecret && llm.apiKey.trim()) {
        await settingsApi.setSecret(apiKeySecret, llm.apiKey.trim());
      }
      const [s, sec] = await Promise.all([settingsApi.get(), settingsApi.listSecrets()]);
      setSettings(s);
      setSecretKeys(sec.keys);
      setLlm(llmFromSettings(s));
      resetLlmModels();
    } catch {
    } finally {
      setSavingLlm(false);
    }
  };

  // ── Embedding section logic (mirrors the LLM section) ──
  const embApiKeySecret = embApiKeySecretFor(emb.provider);
  const embUsesOllama = emb.provider === 'ollama' || emb.provider === 'ollama_remote';
  const embBaseUrl = emb.provider === 'openai' ? emb.openaiUrl : embUsesOllama ? emb.ollamaUrl : '';
  const embDirty =
    !!settings &&
    (emb.provider !== (settings.embedding_provider || 'ollama') ||
      embModelOf(emb) !== embStoredModel(settings, emb.provider) ||
      (emb.provider === 'openai'
        ? emb.openaiUrl !== (settings.openai_base_url ?? '')
        : embUsesOllama
          ? emb.ollamaUrl !== (settings.ollama_embedding_base_url ?? '')
          : false) ||
      emb.dimension !== String(settings.embedding_dimension ?? 1024) ||
      emb.strict !== (settings.embedding_strict ?? true) ||
      emb.apiKey.trim() !== '');

  const searchEmbeddingModels = useCallback(
    (provider: string, baseUrl: string, apiKey: string, query: string) => {
      if (embFetchTimer.current) clearTimeout(embFetchTimer.current);
      if (!query.trim()) {
        embReqId.current++;
        setEmbModels([]);
        setEmbModelsError(null);
        return;
      }
      embFetchTimer.current = setTimeout(() => {
        const reqId = ++embReqId.current;
        settingsApi
          .listEmbeddingModels(provider, {
            baseUrl: baseUrl || undefined,
            apiKey: apiKey || undefined,
            q: query,
          })
          .then((r) => {
            if (reqId !== embReqId.current) return;
            setEmbModels(r.models);
            setEmbModelsError(r.error);
          })
          .catch(() => {
            if (reqId !== embReqId.current) return;
            setEmbModels([]);
            setEmbModelsError('Could not load models');
          });
      }, 400);
    },
    [],
  );

  const resetEmbModels = () => {
    if (embFetchTimer.current) clearTimeout(embFetchTimer.current);
    embReqId.current++;
    setEmbModels([]);
    setEmbModelsError(null);
  };

  const setEmbProvider = (provider: string) => {
    setEmb((d) => ({ ...d, provider, apiKey: '' }));
    resetEmbModels();
  };

  const onEmbModelChange = (value: string) => {
    setEmb((d) => {
      const next =
        d.provider === 'openai'
          ? { ...d, openaiModel: value }
          : d.provider === 'huggingface'
            ? { ...d, hfModel: value }
            : { ...d, ollamaModel: value };
      const known = embModels.find((m) => m.id === value);
      if (known?.dimensions) next.dimension = String(known.dimensions);
      return next;
    });
    searchEmbeddingModels(emb.provider, embBaseUrl, emb.apiKey, value);
  };

  const cancelEmb = () => {
    if (!settings) return;
    setEmb(embFromSettings(settings));
    resetEmbModels();
  };

  const saveEmb = async () => {
    if (!settings || !embDirty || savingEmb) return;
    setSavingEmb(true);
    try {
      const payload: Partial<AppSettings> = {
        embedding_provider: emb.provider,
        embedding_dimension: Number(emb.dimension) || 1024,
        embedding_strict: emb.strict,
      };
      if (emb.provider === 'openai') {
        payload.openai_embedding_model = emb.openaiModel;
        payload.openai_base_url = emb.openaiUrl;
      } else if (emb.provider === 'huggingface') {
        payload.hf_embedding_model = emb.hfModel;
      } else {
        payload.ollama_embedding_model = emb.ollamaModel;
        payload.ollama_embedding_base_url = emb.ollamaUrl;
      }
      await settingsApi.update(payload);
      if (embApiKeySecret && emb.apiKey.trim()) {
        await settingsApi.setSecret(embApiKeySecret, emb.apiKey.trim());
      }
      const [s, sec] = await Promise.all([settingsApi.get(), settingsApi.listSecrets()]);
      setSettings(s);
      setSecretKeys(sec.keys);
      setEmb(embFromSettings(s));
      resetEmbModels();
    } catch {
    } finally {
      setSavingEmb(false);
    }
  };

  const handleDeleteSecret = async (key: string) => {
    try {
      await settingsApi.deleteSecret(key);
      const sec = await settingsApi.listSecrets();
      setSecretKeys(sec.keys);
    } catch {}
  };

  const handleCreateToken = async () => {
    if (!tokenName.trim()) return;
    setCreating(true);
    try {
      const res = await tokensApi.create(tokenName.trim(), tokenScope);
      setCreatedToken(res.token);
      setCopied(false);
      setTokenName('');
      const tok = await tokensApi.list();
      setTokens(tok);
    } catch {
    } finally {
      setCreating(false);
    }
  };

  const handleRevokeToken = async (id: string) => {
    try {
      await tokensApi.remove(id);
      const tok = await tokensApi.list();
      setTokens(tok);
    } catch {}
  };

  const copy = (text: string) => {
    navigator.clipboard?.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8869';

  if (loading && !settings) {
    return (
      <div className="flex h-full items-center justify-center">
        <HudSpinner size={32} />
      </div>
    );
  }

  return (
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
      <div style={{ width: '100%', maxWidth: 1280 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, margin: '0 0 26px' }}>
          <h1 style={{ margin: 0, font: '300 22px/1 "JetBrains Mono"', color: '#eef0f3' }}>
            Settings
          </h1>
          {saving && (
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                font: '400 11px/1 "JetBrains Mono"',
                color: '#8a8f99',
              }}
            >
              <HudSpinner size={12} /> Saving…
            </span>
          )}
        </div>

        {/* Full-width panels stacked; only LLM + Embedding sit side-by-side on
            desktop (see the nested grid below). Capped at the xl (1280px) container. */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          {/* ════ KIO PERSONALITY ════ */}
          <GlassPanel>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 18 }}>
              <SectionDot color="#B87333" />
              <SectionLabel>KIO PERSONALITY</SectionLabel>
            </div>

            <FieldLabel>BUDDY NAME</FieldLabel>
            <TextInput
              value={profile.name}
              onChange={(e) => updateProfile({ name: e.target.value })}
            />

            <FieldLabel>TONE</FieldLabel>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 11 }}>
              <SelCard
                active={profile.tone === 'warm'}
                rgb="184,115,51"
                onClick={() => updateProfile({ tone: 'warm' })}
                title="Warm"
                desc="Kind, encouraging"
              />
              <SelCard
                active={profile.tone === 'focused'}
                rgb="95,211,224"
                onClick={() => updateProfile({ tone: 'focused' })}
                title="Focused"
                desc="Concise, to the point"
              />
              <SelCard
                active={profile.tone === 'playful'}
                rgb="184,115,51"
                onClick={() => updateProfile({ tone: 'playful' })}
                title="Playful"
                desc="Light, curious"
              />
            </div>
          </GlassPanel>

          {/* LLM provider + Embedding side-by-side on desktop, stacked on narrow. */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 420px), 1fr))',
              gap: 18,
              alignItems: 'start',
            }}
          >
            {/* ════ LLM PROVIDER ════ */}
            <GlassPanel>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 7 }}>
                <SectionDot color="#34d6c4" />
                <SectionLabel>LLM PROVIDER</SectionLabel>
              </div>
              <p
                style={{
                  margin: '0 0 16px',
                  font: '400 10px/1.5 "JetBrains Mono"',
                  color: '#6b707a',
                }}
              >
                Where your buddy&apos;s brain runs.
              </p>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: 11,
                  marginBottom: 20,
                }}
              >
                <SelCard
                  active={llm.provider === 'ollama'}
                  rgb="184,115,51"
                  onClick={() => setLlmProvider('ollama')}
                  title="Ollama (local)"
                  desc="Models run on your machine"
                />
                <SelCard
                  active={llm.provider === 'ollama_remote'}
                  rgb="95,211,224"
                  onClick={() => setLlmProvider('ollama_remote')}
                  title="Ollama (remote)"
                  desc="Remote instance"
                />
                <SelCard
                  active={llm.provider === 'openrouter'}
                  rgb="184,115,51"
                  onClick={() => setLlmProvider('openrouter')}
                  title="OpenRouter"
                  desc="100+ models"
                />
              </div>

              {apiKeySecret && (
                <>
                  <FieldLabel>
                    {apiKeySecret === 'openrouter_api_key'
                      ? 'OPENROUTER API KEY'
                      : 'OLLAMA API KEY'}
                  </FieldLabel>
                  <TextInput
                    type="password"
                    value={llm.apiKey}
                    onChange={(e) => setLlm((d) => ({ ...d, apiKey: e.target.value }))}
                    placeholder={
                      secretKeys.includes(apiKeySecret)
                        ? '•••••••• saved — leave blank to keep'
                        : 'Enter API key'
                    }
                  />
                </>
              )}

              {llm.provider === 'openrouter' ? (
                <>
                  <FieldLabel>OPENROUTER BASE URL</FieldLabel>
                  <TextInput
                    value={llm.openrouterUrl}
                    onChange={(e) => setLlm((d) => ({ ...d, openrouterUrl: e.target.value }))}
                    placeholder="https://openrouter.ai"
                  />
                </>
              ) : (
                <>
                  <FieldLabel>OLLAMA BASE URL</FieldLabel>
                  <TextInput
                    value={llm.ollamaUrl}
                    onChange={(e) => setLlm((d) => ({ ...d, ollamaUrl: e.target.value }))}
                    placeholder="https://api.ollama.com"
                  />
                </>
              )}
              <p
                style={{
                  margin: '-14px 0 20px',
                  font: '400 9px/1.4 "JetBrains Mono"',
                  color: '#4a4f59',
                }}
              >
                Paste the host URL — a trailing /api or /v1 is handled automatically.
              </p>

              <FieldLabel>MODEL</FieldLabel>
              <TextInput
                value={llm.model}
                onChange={(e) => {
                  const value = e.target.value;
                  setLlm((d) => ({ ...d, model: value }));
                  searchModels(llm.provider, llmBaseUrl, llm.apiKey, value);
                }}
                placeholder={
                  llm.provider === 'openrouter'
                    ? 'Type to search — e.g. owl-alpha'
                    : 'Type to search — e.g. kimi'
                }
                list="llm-models"
              />
              <datalist id="llm-models">
                {llmModels.map((m) => (
                  <option
                    key={m.id}
                    value={m.id}
                    label={m.name && m.name !== m.id ? m.name : undefined}
                  />
                ))}
              </datalist>
              {llmModelsError ? (
                <p
                  style={{
                    margin: '-14px 0 20px',
                    font: '400 9px/1.4 "JetBrains Mono"',
                    color: '#e7b876',
                  }}
                >
                  {llmModelsError}
                </p>
              ) : llmModels.length > 0 ? (
                <p
                  style={{
                    margin: '-14px 0 20px',
                    font: '400 9px/1.4 "JetBrains Mono"',
                    color: '#6b707a',
                  }}
                >
                  {llmModels.length} match{llmModels.length === 1 ? '' : 'es'} — pick one or keep
                  your own
                </p>
              ) : (
                <p
                  style={{
                    margin: '-14px 0 20px',
                    font: '400 9px/1.4 "JetBrains Mono"',
                    color: '#4a4f59',
                  }}
                >
                  Type to search available models, or enter any name.
                </p>
              )}

              <FieldLabel>DAILY BUDGET (USD)</FieldLabel>
              <TextInput
                type="number"
                value={llm.dailyBudget}
                onChange={(e) => setLlm((d) => ({ ...d, dailyBudget: e.target.value }))}
                placeholder="5"
              />
              <p
                style={{
                  margin: '-14px 0 20px',
                  font: '400 9px/1.4 "JetBrains Mono"',
                  color: '#4a4f59',
                }}
              >
                Daily LLM spend cap. At the limit, processing pauses and new notes &amp; memories
                queue until the budget resets at 00:00 UTC (or you raise it). 0 disables the cap.
              </p>

              {llmDirty && (
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
                  <button
                    onClick={cancelLlm}
                    disabled={savingLlm}
                    style={{
                      padding: '10px 18px',
                      borderRadius: 8,
                      border: '1px solid rgba(255,255,255,0.1)',
                      background: 'rgba(255,255,255,0.02)',
                      color: '#9aa0aa',
                      font: '600 10px/1 "JetBrains Mono"',
                      letterSpacing: '0.1em',
                      cursor: 'pointer',
                      transition: 'all .2s',
                      opacity: savingLlm ? 0.5 : 1,
                    }}
                  >
                    CANCEL
                  </button>
                  <button
                    onClick={saveLlm}
                    disabled={savingLlm}
                    style={{
                      padding: '10px 18px',
                      borderRadius: 8,
                      border: '1px solid rgba(52,214,196,0.3)',
                      background: 'rgba(52,214,196,0.08)',
                      color: '#34d6c4',
                      font: '600 10px/1 "JetBrains Mono"',
                      letterSpacing: '0.1em',
                      cursor: 'pointer',
                      transition: 'all .2s',
                      opacity: savingLlm ? 0.5 : 1,
                    }}
                  >
                    {savingLlm ? 'SAVING…' : 'SAVE CHANGES'}
                  </button>
                </div>
              )}
            </GlassPanel>

            {/* ════ EMBEDDING ════ */}
            <GlassPanel>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 7 }}>
                <SectionDot color="#a78bfa" />
                <SectionLabel>EMBEDDING</SectionLabel>
              </div>
              <p
                style={{
                  margin: '0 0 16px',
                  font: '400 10px/1.5 "JetBrains Mono"',
                  color: '#6b707a',
                }}
              >
                How text is converted into vectors for search.
              </p>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, 1fr)',
                  gap: 11,
                  marginBottom: 20,
                }}
              >
                <SelCard
                  active={emb.provider === 'openai'}
                  rgb="167,139,250"
                  onClick={() => setEmbProvider('openai')}
                  title="OpenAI"
                  desc="text-embedding-3"
                />
                <SelCard
                  active={emb.provider === 'ollama'}
                  rgb="167,139,250"
                  onClick={() => setEmbProvider('ollama')}
                  title="Ollama (local)"
                  desc="Models on your machine"
                />
                <SelCard
                  active={emb.provider === 'ollama_remote'}
                  rgb="167,139,250"
                  onClick={() => setEmbProvider('ollama_remote')}
                  title="Ollama (remote)"
                  desc="Remote instance"
                />
                <SelCard
                  active={emb.provider === 'huggingface'}
                  rgb="167,139,250"
                  onClick={() => setEmbProvider('huggingface')}
                  title="HuggingFace"
                  desc="Inference API"
                />
              </div>

              {embApiKeySecret && (
                <>
                  <FieldLabel>
                    {embApiKeySecret === 'openai_api_key'
                      ? 'OPENAI API KEY'
                      : embApiKeySecret === 'ollama_api_key'
                        ? 'OLLAMA API KEY'
                        : 'HUGGINGFACE API KEY'}
                  </FieldLabel>
                  <TextInput
                    type="password"
                    value={emb.apiKey}
                    onChange={(e) => setEmb((d) => ({ ...d, apiKey: e.target.value }))}
                    placeholder={
                      secretKeys.includes(embApiKeySecret)
                        ? '•••••••• saved — leave blank to keep'
                        : 'Enter API key'
                    }
                  />
                </>
              )}

              {emb.provider === 'openai' ? (
                <>
                  <FieldLabel>OPENAI BASE URL</FieldLabel>
                  <TextInput
                    value={emb.openaiUrl}
                    onChange={(e) => setEmb((d) => ({ ...d, openaiUrl: e.target.value }))}
                    placeholder="https://api.openai.com"
                  />
                  <p
                    style={{
                      margin: '-14px 0 20px',
                      font: '400 9px/1.4 "JetBrains Mono"',
                      color: '#4a4f59',
                    }}
                  >
                    Paste the host URL — a trailing /v1 is handled automatically.
                  </p>
                </>
              ) : embUsesOllama ? (
                <>
                  <FieldLabel>OLLAMA EMBEDDING BASE URL</FieldLabel>
                  <TextInput
                    value={emb.ollamaUrl}
                    onChange={(e) => setEmb((d) => ({ ...d, ollamaUrl: e.target.value }))}
                    placeholder={
                      emb.provider === 'ollama_remote'
                        ? 'https://api.ollama.com'
                        : 'http://host.docker.internal:11434'
                    }
                  />
                  <p
                    style={{
                      margin: '-14px 0 20px',
                      font: '400 9px/1.4 "JetBrains Mono"',
                      color: '#4a4f59',
                    }}
                  >
                    Paste the host URL — a trailing /api or /v1 is handled automatically.
                  </p>
                </>
              ) : null}

              <FieldLabel>EMBEDDING MODEL</FieldLabel>
              <TextInput
                value={embModelOf(emb)}
                onChange={(e) => onEmbModelChange(e.target.value)}
                placeholder={
                  emb.provider === 'openai'
                    ? 'Type to search — e.g. text-embedding-3'
                    : emb.provider === 'huggingface'
                      ? 'Type to search — e.g. bge'
                      : 'Type to search — e.g. mxbai'
                }
                list="emb-models"
              />
              <datalist id="emb-models">
                {embModels.map((m) => (
                  <option
                    key={m.id}
                    value={m.id}
                    label={m.name && m.name !== m.id ? m.name : undefined}
                  />
                ))}
              </datalist>
              {embModelsError ? (
                <p
                  style={{
                    margin: '-14px 0 20px',
                    font: '400 9px/1.4 "JetBrains Mono"',
                    color: '#e7b876',
                  }}
                >
                  {embModelsError}
                </p>
              ) : embModels.length > 0 ? (
                <p
                  style={{
                    margin: '-14px 0 20px',
                    font: '400 9px/1.4 "JetBrains Mono"',
                    color: '#6b707a',
                  }}
                >
                  {embModels.length} match{embModels.length === 1 ? '' : 'es'} — pick one or keep
                  your own
                </p>
              ) : (
                <p
                  style={{
                    margin: '-14px 0 20px',
                    font: '400 9px/1.4 "JetBrains Mono"',
                    color: '#4a4f59',
                  }}
                >
                  Type to search available models, or enter any name.
                </p>
              )}

              <FieldLabel>DIMENSION</FieldLabel>
              <TextInput
                value={emb.dimension}
                onChange={(e) => setEmb((d) => ({ ...d, dimension: e.target.value }))}
              />

              <ToggleRow
                label="Strict dimension check"
                desc="Rejects embeddings with a different dimension"
                value={emb.strict}
                onChange={(v) => setEmb((d) => ({ ...d, strict: v }))}
              />

              {embDirty && (
                <div
                  style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 10 }}
                >
                  <button
                    onClick={cancelEmb}
                    disabled={savingEmb}
                    style={{
                      padding: '10px 18px',
                      borderRadius: 8,
                      border: '1px solid rgba(255,255,255,0.1)',
                      background: 'rgba(255,255,255,0.02)',
                      color: '#9aa0aa',
                      font: '600 10px/1 "JetBrains Mono"',
                      letterSpacing: '0.1em',
                      cursor: 'pointer',
                      transition: 'all .2s',
                      opacity: savingEmb ? 0.5 : 1,
                    }}
                  >
                    CANCEL
                  </button>
                  <button
                    onClick={saveEmb}
                    disabled={savingEmb}
                    style={{
                      padding: '10px 18px',
                      borderRadius: 8,
                      border: '1px solid rgba(167,139,250,0.3)',
                      background: 'rgba(167,139,250,0.08)',
                      color: '#a78bfa',
                      font: '600 10px/1 "JetBrains Mono"',
                      letterSpacing: '0.1em',
                      cursor: 'pointer',
                      transition: 'all .2s',
                      opacity: savingEmb ? 0.5 : 1,
                    }}
                  >
                    {savingEmb ? 'SAVING…' : 'SAVE CHANGES'}
                  </button>
                </div>
              )}
            </GlassPanel>
          </div>

          {/* ════ SECURITY ════ */}
          <GlassPanel>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 7 }}>
              <SectionDot color="#f87171" />
              <SectionLabel>SECURITY</SectionLabel>
            </div>

            <ToggleRow
              label="Require authentication"
              desc="When off, the API is accessible without a token (local-trust)"
              value={settings?.auth_required ?? true}
              onChange={(v) => updateSetting('auth_required', v)}
            />

            <FieldLabel>CORS ORIGINS</FieldLabel>
            <TextInput
              value={settings?.cors_origins || ''}
              onChange={(e) => updateSetting('cors_origins', e.target.value)}
            />

            <div style={{ marginTop: 20 }}>
              <FieldLabel>API KEYS (encrypted)</FieldLabel>
              {secretKeys.length === 0 ? (
                <p style={{ font: '400 10px/1 "JetBrains Mono"', color: '#6b707a' }}>
                  No saved keys.
                </p>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 14 }}>
                  {secretKeys.map((k) => (
                    <div
                      key={k}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '8px 12px',
                        borderRadius: 8,
                        border: '1px solid rgba(255,255,255,0.06)',
                        background: 'rgba(255,255,255,0.015)',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <KeyIcon />
                        <span style={{ font: '400 11px/1 "JetBrains Mono"', color: '#cfd2d8' }}>
                          {k}
                        </span>
                      </div>
                      <button
                        onClick={() => handleDeleteSecret(k)}
                        style={{
                          padding: '4px 8px',
                          borderRadius: 6,
                          border: '1px solid rgba(248,113,113,0.3)',
                          background: 'rgba(248,113,113,0.08)',
                          color: '#f87171',
                          font: '500 9px/1 "JetBrains Mono"',
                          cursor: 'pointer',
                          transition: 'all .2s',
                        }}
                      >
                        <TrashIcon />
                      </button>
                    </div>
                  ))}
                </div>
              )}
              <p style={{ marginTop: 10, font: '400 9px/1.4 "JetBrains Mono"', color: '#4a4f59' }}>
                API keys are set in the LLM provider and Embedding sections above.
              </p>
            </div>
          </GlassPanel>

          {/* ════ AUTOMATION ════ */}
          <GlassPanel>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 7 }}>
              <SectionDot color="#fbbf24" />
              <SectionLabel>AUTOMATION</SectionLabel>
            </div>
            <p
              style={{
                margin: '0 0 16px',
                font: '400 10px/1.5 "JetBrains Mono"',
                color: '#6b707a',
              }}
            >
              Everything KIO does automatically in the background.
            </p>

            <ToggleRow
              label="Auto-extract entities"
              desc="Automatically extracts entities and relationships from notes"
              value={settings?.auto_extract ?? true}
              onChange={(v) => updateSetting('auto_extract', v)}
            />
            <ToggleRow
              label="Auto-consolidation"
              desc="Regularly consolidates memories (dedup, merge, tiering)"
              value={settings?.auto_consolidation ?? true}
              onChange={(v) => updateSetting('auto_consolidation', v)}
            />
          </GlassPanel>

          {/* ════ API TOKENS ════ */}
          <GlassPanel>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 7 }}>
              <SectionDot color="#5fd3e0" />
              <SectionLabel>API TOKENS (MCP)</SectionLabel>
            </div>
            <p
              style={{
                margin: '0 0 16px',
                font: '400 10px/1.5 "JetBrains Mono"',
                color: '#6b707a',
              }}
            >
              Tokens for external applications (Claude Desktop, Cursor, custom agents) to access KIO
              over MCP. Endpoint:{' '}
              <code
                style={{
                  background: 'rgba(255,255,255,0.04)',
                  padding: '2px 6px',
                  borderRadius: 4,
                  font: '400 10px/1 "JetBrains Mono"',
                  color: '#e7b876',
                }}
              >
                {API_BASE}/mcp
              </code>
            </p>

            {/* Freshly created token */}
            {createdToken && (
              <div
                style={{
                  marginBottom: 14,
                  padding: 14,
                  borderRadius: 11,
                  border: '1px solid rgba(184,115,51,0.4)',
                  background: 'rgba(184,115,51,0.1)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <AlertIcon />
                  <span style={{ font: '500 10px/1 "JetBrains Mono"', color: '#e7b876' }}>
                    Copy your token — you will never see it again.
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <code
                    style={{
                      flex: 1,
                      overflowX: 'auto',
                      padding: '10px 14px',
                      borderRadius: 8,
                      background: 'rgba(255,255,255,0.04)',
                      font: '400 10px/1 "JetBrains Mono"',
                      color: '#cfd2d8',
                    }}
                  >
                    {createdToken}
                  </code>
                  <button
                    onClick={() => copy(createdToken)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      padding: '8px 12px',
                      borderRadius: 8,
                      border: '1px solid rgba(184,115,51,0.3)',
                      background: 'rgba(184,115,51,0.08)',
                      color: '#e7b876',
                      font: '500 9px/1 "JetBrains Mono"',
                      cursor: 'pointer',
                      transition: 'all .2s',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {copied ? <CheckIcon /> : <CopyIcon />}
                    {copied ? 'Copied' : 'Copy'}
                  </button>
                </div>
                <button
                  onClick={() => setCreatedToken(null)}
                  style={{
                    marginTop: 8,
                    background: 'none',
                    border: 'none',
                    font: '400 9px/1 "JetBrains Mono"',
                    color: '#6b707a',
                    cursor: 'pointer',
                  }}
                >
                  Close
                </button>
              </div>
            )}

            {/* Create form */}
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                alignItems: 'flex-end',
                gap: 10,
                marginBottom: 16,
              }}
            >
              <div style={{ flex: '1 1 180px' }}>
                <div
                  style={{ font: '400 8px/1 "JetBrains Mono"', color: '#6b707a', marginBottom: 6 }}
                >
                  NAME
                </div>
                <input
                  value={tokenName}
                  onChange={(e) => setTokenName(e.target.value)}
                  placeholder="e.g. Claude Desktop"
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: 8,
                    border: '1px solid rgba(255,255,255,0.08)',
                    background: 'rgba(255,255,255,0.02)',
                    color: '#cfd2d8',
                    font: '400 11px/1 "JetBrains Mono"',
                    outline: 'none',
                  }}
                />
              </div>
              <div style={{ minWidth: 140 }}>
                <div
                  style={{ font: '400 8px/1 "JetBrains Mono"', color: '#6b707a', marginBottom: 6 }}
                >
                  SCOPE
                </div>
                <select
                  value={tokenScope}
                  onChange={(e) => setTokenScope(e.target.value as 'read' | 'full')}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: 8,
                    border: '1px solid rgba(255,255,255,0.08)',
                    background: 'rgba(255,255,255,0.02)',
                    color: '#cfd2d8',
                    font: '400 11px/1 "JetBrains Mono"',
                    outline: 'none',
                  }}
                >
                  <option value="full">Full (read + write)</option>
                  <option value="read">Read-only</option>
                </select>
              </div>
              <button
                onClick={handleCreateToken}
                disabled={creating || !tokenName.trim()}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '10px 16px',
                  borderRadius: 8,
                  border: '1px solid rgba(95,211,224,0.3)',
                  background: 'rgba(95,211,224,0.08)',
                  color: '#5fd3e0',
                  font: '600 10px/1 "JetBrains Mono"',
                  letterSpacing: '0.1em',
                  cursor: 'pointer',
                  transition: 'all .2s',
                  whiteSpace: 'nowrap',
                  opacity: creating || !tokenName.trim() ? 0.5 : 1,
                }}
              >
                <PlusIcon />
                GENERATE
              </button>
            </div>

            {/* Token list */}
            {tokens.length === 0 ? (
              <p style={{ font: '400 10px/1 "JetBrains Mono"', color: '#6b707a' }}>
                No tokens yet.
              </p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {tokens.map((t) => (
                  <div
                    key={t.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '10px 14px',
                      borderRadius: 9,
                      border: '1px solid rgba(255,255,255,0.06)',
                      background: 'rgba(255,255,255,0.015)',
                    }}
                  >
                    <div style={{ minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ font: '400 11px/1 "JetBrains Mono"', color: '#cfd2d8' }}>
                          {t.name}
                        </span>
                        <span
                          style={{
                            padding: '2px 7px',
                            borderRadius: 5,
                            background:
                              t.scope === 'full' ? 'rgba(52,214,196,0.1)' : 'rgba(167,139,250,0.1)',
                            border: `1px solid ${t.scope === 'full' ? 'rgba(52,214,196,0.2)' : 'rgba(167,139,250,0.2)'}`,
                            font: '400 8px/1 "JetBrains Mono"',
                            color: t.scope === 'full' ? '#34d6c4' : '#a78bfa',
                          }}
                        >
                          {t.scope === 'full' ? 'Full' : 'Read'}
                        </span>
                      </div>
                      <div
                        style={{
                          marginTop: 4,
                          font: '400 8px/1 "JetBrains Mono"',
                          color: '#4a4f59',
                        }}
                      >
                        <code>{t.prefix}…</code> · {new Date(t.created).toLocaleDateString()}
                        {t.last_used
                          ? ` · last used ${new Date(t.last_used).toLocaleDateString()}`
                          : ' · unused'}
                      </div>
                    </div>
                    <button
                      onClick={() => handleRevokeToken(t.id)}
                      style={{
                        padding: '6px 10px',
                        borderRadius: 6,
                        border: '1px solid rgba(248,113,113,0.3)',
                        background: 'rgba(248,113,113,0.08)',
                        color: '#f87171',
                        font: '500 9px/1 "JetBrains Mono"',
                        cursor: 'pointer',
                        transition: 'all .2s',
                      }}
                    >
                      <TrashIcon />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </GlassPanel>
        </div>

        <div style={{ height: 80 }} />
      </div>
    </div>
  );
}
