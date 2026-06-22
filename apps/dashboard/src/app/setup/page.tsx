'use client';

import { useState } from 'react';
import { setupApi, authApi } from '@/lib/api';
import type { SetupPayload } from '@/types';
import { HudButton, HudInput, HudSpinner, HudErrorBanner } from '@/components/hud';

const STEPS = ['Welcome', 'AI Provider', 'Embeddings', 'Account', 'Finish'] as const;

type Form = {
  llm_provider: string;
  llm_model: string;
  ollama_base_url: string;
  ollama_api_key: string;
  openrouter_base_url: string;
  openrouter_api_key: string;
  embedding_provider: string;
  ollama_embedding_base_url: string;
  ollama_embedding_model: string;
  openai_base_url: string;
  openai_embedding_model: string;
  openai_api_key: string;
  hf_embedding_model: string;
  hf_api_key: string;
  embedding_dimension: string;
  embedding_strict: boolean;
  llm_daily_budget_usd: string;
  username: string;
  password: string;
  confirm: string;
};

const INITIAL: Form = {
  llm_provider: 'openrouter',
  llm_model: 'openrouter/owl-alpha',
  ollama_base_url: 'http://host.docker.internal:11434',
  ollama_api_key: '',
  openrouter_base_url: 'https://openrouter.ai/api/v1',
  openrouter_api_key: '',
  embedding_provider: 'huggingface',
  ollama_embedding_base_url: 'http://host.docker.internal:11434',
  ollama_embedding_model: 'mxbai-embed-large',
  openai_base_url: 'https://api.openai.com',
  openai_embedding_model: 'text-embedding-3-small',
  openai_api_key: '',
  hf_embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
  hf_api_key: '',
  embedding_dimension: '384',
  embedding_strict: true,
  llm_daily_budget_usd: '5',
  username: '',
  password: '',
  confirm: '',
};

// Provider-specific embedding dimension defaults, applied on provider switch.
const EMB_DIMENSION: Record<string, string> = {
  ollama: '1024',
  ollama_remote: '1024',
  openai: '1536',
  huggingface: '384',
};

const embNeedsSecret = (p: string): boolean =>
  p === 'openai' || p === 'huggingface' || p === 'ollama_remote';

const embModelValueOf = (f: Form): string =>
  f.embedding_provider === 'openai'
    ? f.openai_embedding_model
    : f.embedding_provider === 'huggingface'
      ? f.hf_embedding_model
      : f.ollama_embedding_model;

const embSecretValueOf = (f: Form): string =>
  f.embedding_provider === 'openai'
    ? f.openai_api_key
    : f.embedding_provider === 'huggingface'
      ? f.hf_api_key
      : f.embedding_provider === 'ollama_remote'
        ? f.ollama_api_key
        : '';

function BrainIcon() {
  return (
    <svg
      aria-hidden="true"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-aiki-accent"
    >
      <path d="M12 4.5a2.5 2.5 0 0 0-4.96-.46 2.5 2.5 0 0 0-1.98 3.46 2.5 2.5 0 0 0-1.32 4.24 3 3 0 0 0 .34 5.58 2.5 2.5 0 0 0 2.96 3.08 2.5 2.5 0 0 0 4.91.05L12 20V4.5Z" />
      <path d="M16 8V5c0-1.1.9-2 2-2" />
      <path d="M12 4h4a2 2 0 0 1 2 2v2M12 12h4a2 2 0 0 1 2 2v2M12 20h4a2 2 0 0 0 2-2v-2" />
    </svg>
  );
}

function CpuIcon() {
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
      className="text-aiki-accent"
    >
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <line x1="9" y1="1" x2="9" y2="4" />
      <line x1="15" y1="1" x2="15" y2="4" />
      <line x1="9" y1="20" x2="9" y2="23" />
      <line x1="15" y1="20" x2="15" y2="23" />
      <line x1="20" y1="9" x2="23" y2="9" />
      <line x1="20" y1="14" x2="23" y2="14" />
      <line x1="1" y1="9" x2="4" y2="9" />
      <line x1="1" y1="14" x2="4" y2="14" />
    </svg>
  );
}
function DatabaseIcon() {
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
      className="text-aiki-accent"
    >
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
    </svg>
  );
}
function ShieldIcon() {
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
      className="text-aiki-accent"
    >
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}
function CheckIcon() {
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
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}
function ArrowLeftIcon() {
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
      <line x1="19" y1="12" x2="5" y2="12" />
      <polyline points="12 19 5 12 12 5" />
    </svg>
  );
}
function ArrowRightIcon() {
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
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  );
}

export default function SetupWizard() {
  const [step, setStep] = useState(0);
  const [form, setForm] = useState<Form>(INITIAL);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);

  const set = (k: keyof Form) => (v: string) => setForm((f) => ({ ...f, [k]: v }));
  const isRemote = form.llm_provider === 'ollama_remote';
  const isOpenRouter = form.llm_provider === 'openrouter';
  const embProvider = form.embedding_provider;

  const onEmbProviderChange = (p: string) =>
    setForm((f) => ({
      ...f,
      embedding_provider: p,
      embedding_dimension: EMB_DIMENSION[p] ?? f.embedding_dimension,
    }));

  const stepValid = (): boolean => {
    if (step === 1) {
      if (isOpenRouter) return !!form.llm_model.trim() && !!form.openrouter_api_key.trim();
      return (
        !!form.llm_model.trim() &&
        !!form.ollama_base_url.trim() &&
        (!isRemote || !!form.ollama_api_key.trim())
      );
    }
    if (step === 2) {
      if (!embModelValueOf(form).trim()) return false;
      if (embNeedsSecret(embProvider) && !embSecretValueOf(form).trim()) return false;
      return true;
    }
    if (step === 3)
      return (
        form.username.trim().length >= 3 &&
        form.password.length >= 8 &&
        form.password === form.confirm
      );
    return true;
  };

  const runTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const r = await setupApi.test({
        llm_provider: form.llm_provider,
        ollama_base_url: form.ollama_base_url,
        ollama_api_key: form.ollama_api_key || undefined,
        openrouter_base_url: form.openrouter_base_url,
        openrouter_api_key: form.openrouter_api_key || undefined,
      });
      setTestResult(r.ok ? `Reachable (${r.detail})` : `Unreachable: ${r.detail}`);
    } catch (e) {
      setTestResult(e instanceof Error ? e.message : String(e));
    } finally {
      setTesting(false);
    }
  };

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    const payload: SetupPayload = {
      llm_provider: form.llm_provider,
      llm_model: form.llm_model,
      ollama_base_url: form.ollama_base_url,
      embedding_provider: embProvider,
      embedding_dimension: Number(form.embedding_dimension) || 384,
      embedding_strict: form.embedding_strict,
      llm_daily_budget_usd: Number(form.llm_daily_budget_usd) || 5,
      account: { username: form.username.trim(), password: form.password },
    };
    if (isOpenRouter) {
      payload.openrouter_base_url = form.openrouter_base_url;
      payload.openrouter_api_key = form.openrouter_api_key || undefined;
    } else if (isRemote) {
      payload.ollama_api_key = form.ollama_api_key || undefined;
    }
    if (embProvider === 'openai') {
      payload.openai_embedding_model = form.openai_embedding_model;
      payload.openai_base_url = form.openai_base_url;
      payload.openai_api_key = form.openai_api_key || undefined;
    } else if (embProvider === 'huggingface') {
      payload.hf_embedding_model = form.hf_embedding_model;
      payload.hf_api_key = form.hf_api_key || undefined;
    } else {
      payload.ollama_embedding_model = form.ollama_embedding_model;
      payload.ollama_embedding_base_url = form.ollama_embedding_base_url;
      if (embProvider === 'ollama_remote')
        payload.ollama_api_key = form.ollama_api_key || undefined;
    }
    try {
      await setupApi.submit(payload);
      await authApi.login(form.username.trim(), form.password);
      window.location.href = '/';
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSubmitting(false);
    }
  };

  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="w-full max-w-lg rounded-xl border border-white/[0.06] bg-aiki-panel backdrop-blur-sm p-6 sm:p-8 relative overflow-hidden">
        {/* Inner glow */}
        <div className="pointer-events-none absolute inset-0 rounded-xl bg-gradient-to-b from-white/[0.03] to-transparent" />
        {/* Corner accents */}
        <div className="absolute top-0 left-0 w-4 h-4 border-t border-l border-aiki-accent/40 rounded-tl-xl" />
        <div className="absolute top-0 right-0 w-4 h-4 border-t border-r border-aiki-accent/30 rounded-tr-xl" />
        <div className="absolute bottom-0 left-0 w-4 h-4 border-b border-l border-aiki-accent/30 rounded-bl-xl" />
        <div className="absolute bottom-0 right-0 w-4 h-4 border-b border-r border-aiki-accent/40 rounded-br-xl" />

        {/* Header + progress */}
        <div className="mb-6 flex items-center gap-3 relative z-10">
          <BrainIcon />
          <div>
            <h1 className="text-xl font-sans font-bold tracking-[0.15em] text-aiki-text">
              Set up Aikioku
            </h1>
            <p className="text-xs font-mono text-aiki-text-tertiary">
              Step {step + 1} of {STEPS.length} · {STEPS[step]}
            </p>
          </div>
        </div>
        <div className="mb-6 flex gap-1.5 relative z-10">
          {STEPS.map((s, i) => (
            <div
              key={s}
              className={`h-1 flex-1 rounded-full transition-colors ${i <= step ? 'bg-aiki-accent' : 'bg-white/[0.06]'}`}
            />
          ))}
        </div>

        {/* Steps */}
        <div className="space-y-4 relative z-10">
          {step === 0 && (
            <div className="space-y-3 text-sm text-aiki-text-secondary">
              <p>
                Welcome. This wizard configures your AI provider, embeddings, and admin account.
              </p>
              <p className="text-aiki-text-tertiary">
                Settings and API keys are stored encrypted in your local database — no{' '}
                <code className="font-mono text-xs bg-white/[0.05] px-1.5 py-0.5 rounded ">
                  .env
                </code>{' '}
                file. You can change everything later in Settings.
              </p>
            </div>
          )}

          {step === 1 && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-aiki-text">
                <CpuIcon />
                <span className="font-sans text-xs uppercase ">AI Provider</span>
              </div>
              <div>
                <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                  Provider
                </label>
                <select
                  value={form.llm_provider}
                  onChange={(e) => {
                    const p = e.target.value;
                    set('llm_provider')(p);
                    if (
                      p === 'openrouter' &&
                      (form.llm_model === '' || form.llm_model === 'kimi-k2.6:cloud')
                    ) {
                      set('llm_model')('openrouter/owl-alpha');
                    } else if (p !== 'openrouter' && form.llm_model === 'openrouter/owl-alpha') {
                      set('llm_model')('kimi-k2.6:cloud');
                    }
                  }}
                  className="w-full rounded-lg border border-white/[0.08] bg-white/[0.03] px-3 py-2 text-sm font-mono text-aiki-text-secondary focus:border-aiki-accent/50 focus:outline-none focus:shadow-[0_0_16px_rgba(184,115,51,0.1)]"
                >
                  <option value="ollama">Ollama (Local)</option>
                  <option value="ollama_remote">Ollama (Remote / Cloud)</option>
                  <option value="openrouter">OpenRouter</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                  Model
                </label>
                <HudInput
                  value={form.llm_model}
                  onChange={(e) => set('llm_model')(e.target.value)}
                  placeholder={isOpenRouter ? 'openrouter/owl-alpha' : 'kimi-k2.6:cloud'}
                />
              </div>
              {isOpenRouter ? (
                <>
                  <div>
                    <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                      OpenRouter Base URL
                    </label>
                    <HudInput
                      value={form.openrouter_base_url}
                      onChange={(e) => set('openrouter_base_url')(e.target.value)}
                      placeholder="https://openrouter.ai"
                    />
                    <p className="mt-1 text-[10px] font-mono text-aiki-text-tertiary">
                      Host URL — a trailing /api or /v1 is handled automatically.
                    </p>
                  </div>
                  <div>
                    <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                      API Key
                    </label>
                    <HudInput
                      type="password"
                      value={form.openrouter_api_key}
                      onChange={(e) => set('openrouter_api_key')(e.target.value)}
                    />
                  </div>
                </>
              ) : (
                <>
                  <div>
                    <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                      Base URL
                    </label>
                    <HudInput
                      value={form.ollama_base_url}
                      onChange={(e) => set('ollama_base_url')(e.target.value)}
                      placeholder={isRemote ? 'https://api.ollama.com' : 'http://localhost:11434'}
                    />
                    <p className="mt-1 text-[10px] font-mono text-aiki-text-tertiary">
                      Host URL — a trailing /api or /v1 is handled automatically.
                    </p>
                  </div>
                  {isRemote && (
                    <div>
                      <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                        API Key
                      </label>
                      <HudInput
                        type="password"
                        value={form.ollama_api_key}
                        onChange={(e) => set('ollama_api_key')(e.target.value)}
                      />
                    </div>
                  )}
                </>
              )}
              <div>
                <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                  Daily budget (USD)
                </label>
                <HudInput
                  type="number"
                  min="0"
                  step="0.5"
                  value={form.llm_daily_budget_usd}
                  onChange={(e) => set('llm_daily_budget_usd')(e.target.value)}
                  placeholder="5"
                />
                <p className="mt-1 text-[10px] font-mono text-aiki-text-tertiary">
                  Daily LLM spend cap. At the limit, processing pauses and new notes &amp; memories
                  queue until it resets at 00:00 UTC. 0 disables the cap.
                </p>
              </div>
              <div className="flex items-center gap-3">
                <HudButton onClick={runTest} disabled={testing} variant="ghost" size="sm">
                  {testing && <HudSpinner size={14} />} Test connection
                </HudButton>
                {testResult && (
                  <span className="text-[10px] font-mono text-aiki-text-tertiary">
                    {testResult}
                  </span>
                )}
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-aiki-text">
                <DatabaseIcon />
                <span className="font-sans text-xs uppercase ">Embeddings</span>
              </div>
              <div>
                <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                  Provider
                </label>
                <select
                  value={embProvider}
                  onChange={(e) => onEmbProviderChange(e.target.value)}
                  className="w-full rounded-lg border border-white/[0.08] bg-white/[0.03] px-3 py-2 text-sm font-mono text-aiki-text-secondary focus:border-aiki-accent/50 focus:outline-none focus:shadow-[0_0_16px_rgba(184,115,51,0.1)]"
                >
                  <option value="huggingface">HuggingFace (Inference API)</option>
                  <option value="openai">OpenAI</option>
                  <option value="ollama">Ollama (Local)</option>
                  <option value="ollama_remote">Ollama (Remote / Cloud)</option>
                </select>
              </div>

              {embProvider === 'openai' && (
                <>
                  <div>
                    <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                      Embedding model
                    </label>
                    <HudInput
                      value={form.openai_embedding_model}
                      onChange={(e) => set('openai_embedding_model')(e.target.value)}
                      placeholder="text-embedding-3-small"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                      OpenAI base URL
                    </label>
                    <HudInput
                      value={form.openai_base_url}
                      onChange={(e) => set('openai_base_url')(e.target.value)}
                      placeholder="https://api.openai.com"
                    />
                    <p className="mt-1 text-[10px] font-mono text-aiki-text-tertiary">
                      Host URL — a trailing /v1 is handled automatically.
                    </p>
                  </div>
                  <div>
                    <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                      OpenAI API key
                    </label>
                    <HudInput
                      type="password"
                      value={form.openai_api_key}
                      onChange={(e) => set('openai_api_key')(e.target.value)}
                    />
                  </div>
                </>
              )}

              {embProvider === 'huggingface' && (
                <>
                  <div>
                    <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                      Embedding model
                    </label>
                    <HudInput
                      value={form.hf_embedding_model}
                      onChange={(e) => set('hf_embedding_model')(e.target.value)}
                      placeholder="sentence-transformers/all-MiniLM-L6-v2"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                      HuggingFace API key
                    </label>
                    <HudInput
                      type="password"
                      value={form.hf_api_key}
                      onChange={(e) => set('hf_api_key')(e.target.value)}
                    />
                  </div>
                </>
              )}

              {(embProvider === 'ollama' || embProvider === 'ollama_remote') && (
                <>
                  <div>
                    <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                      Embedding model
                    </label>
                    <HudInput
                      value={form.ollama_embedding_model}
                      onChange={(e) => set('ollama_embedding_model')(e.target.value)}
                      placeholder="mxbai-embed-large"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                      Embedding base URL
                    </label>
                    <HudInput
                      value={form.ollama_embedding_base_url}
                      onChange={(e) => set('ollama_embedding_base_url')(e.target.value)}
                      placeholder={
                        embProvider === 'ollama_remote'
                          ? 'https://api.ollama.com'
                          : 'http://host.docker.internal:11434'
                      }
                    />
                  </div>
                  {embProvider === 'ollama_remote' && (
                    <div>
                      <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                        Ollama API key
                      </label>
                      <HudInput
                        type="password"
                        value={form.ollama_api_key}
                        onChange={(e) => set('ollama_api_key')(e.target.value)}
                        placeholder="Enter API key"
                      />
                    </div>
                  )}
                </>
              )}

              <div>
                <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                  Embedding dimension
                </label>
                <HudInput
                  type="number"
                  value={form.embedding_dimension}
                  onChange={(e) => set('embedding_dimension')(e.target.value)}
                />
                <p className="mt-1 text-[10px] font-mono text-aiki-text-tertiary">
                  Must match the model — e.g. all-MiniLM-L6-v2 → 384, mxbai-embed-large → 1024.
                </p>
              </div>

              <label className="flex items-center gap-2 text-xs font-sans text-aiki-text-secondary cursor-pointer">
                <input
                  type="checkbox"
                  checked={form.embedding_strict}
                  onChange={(e) => setForm((f) => ({ ...f, embedding_strict: e.target.checked }))}
                  className="accent-aiki-accent"
                />
                Strict mode — fail loudly instead of writing a placeholder vector when the model is
                unreachable
              </label>
            </div>
          )}

          {step === 3 && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-aiki-text">
                <ShieldIcon />
                <span className="font-sans text-xs uppercase ">Admin Account</span>
              </div>
              <div>
                <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                  Username
                </label>
                <HudInput
                  value={form.username}
                  onChange={(e) => set('username')(e.target.value)}
                  placeholder="admin"
                />
              </div>
              <div>
                <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                  Password
                </label>
                <HudInput
                  type="password"
                  value={form.password}
                  onChange={(e) => set('password')(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-xs font-sans font-medium text-aiki-text-tertiary mb-1.5">
                  Confirm password
                </label>
                <HudInput
                  type="password"
                  value={form.confirm}
                  onChange={(e) => set('confirm')(e.target.value)}
                />
              </div>
              {form.confirm && form.password !== form.confirm && (
                <p className="text-[10px] text-aiki-danger">Passwords do not match.</p>
              )}
            </div>
          )}

          {step === 4 && (
            <div className="space-y-2 text-sm text-aiki-text-secondary">
              <p className="text-aiki-text-tertiary">Review and finish:</p>
              <ul className="space-y-1 text-aiki-text-secondary font-mono text-xs">
                <li>
                  Provider: <span className="text-aiki-text">{form.llm_provider}</span> · model{' '}
                  <span className="text-aiki-text">{form.llm_model}</span>
                </li>
                <li>
                  Embedding: <span className="text-aiki-text">{embProvider}</span> · model{' '}
                  <span className="text-aiki-text">{embModelValueOf(form)}</span> (
                  {form.embedding_dimension}d)
                </li>
                <li>
                  Account: <span className="text-aiki-text">{form.username || '—'}</span>
                </li>
                <li>
                  Secrets set:{' '}
                  <span className="text-aiki-text">
                    {[
                      (isRemote || embProvider === 'ollama_remote') &&
                        form.ollama_api_key &&
                        'ollama_api_key',
                      isOpenRouter && form.openrouter_api_key && 'openrouter_api_key',
                      embProvider === 'openai' && form.openai_api_key && 'openai_api_key',
                      embProvider === 'huggingface' && form.hf_api_key && 'hf_api_key',
                    ]
                      .filter(Boolean)
                      .join(', ') || 'none'}
                  </span>
                </li>
              </ul>
              <p className="text-[10px] text-aiki-text-tertiary">
                Finishing enables the login wall and starts the AI runtime.
              </p>
            </div>
          )}
        </div>

        {error && (
          <div className="mt-4 relative z-10">
            <HudErrorBanner message={error} />
          </div>
        )}

        {/* Nav */}
        <div className="mt-8 flex items-center justify-between relative z-10">
          <HudButton
            onClick={() => setStep((s) => Math.max(0, s - 1))}
            disabled={step === 0 || submitting}
            variant="ghost"
            size="sm"
          >
            <ArrowLeftIcon /> Back
          </HudButton>
          {step < STEPS.length - 1 ? (
            <HudButton onClick={() => setStep((s) => s + 1)} disabled={!stepValid()} size="sm">
              Next <ArrowRightIcon />
            </HudButton>
          ) : (
            <HudButton onClick={submit} disabled={submitting} size="sm">
              {submitting ? <HudSpinner size={14} /> : <CheckIcon />} Finish setup
            </HudButton>
          )}
        </div>
      </div>
    </div>
  );
}
