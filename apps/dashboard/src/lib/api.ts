// API client for backend (fetch wrapper)
import { normalizeCitationsEvent } from '@/lib/chat-events';
import type { Citation } from '@/lib/chat-events';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8869';

// --- auth token (persisted in localStorage; attached as a bearer header) -------
const TOKEN_KEY = 'sb_token';

export function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return window.localStorage.getItem(TOKEN_KEY);
}
export function setToken(token: string): void {
  if (typeof window !== 'undefined') window.localStorage.setItem(TOKEN_KEY, token);
}
export function clearToken(): void {
  if (typeof window !== 'undefined') window.localStorage.removeItem(TOKEN_KEY);
}

export function authHeaders(extra?: HeadersInit): Record<string, string> {
  const token = getToken();
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(extra as Record<string, string> | undefined),
  };
}

function handleUnauthorized(): void {
  clearToken();
  if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
    window.location.href = '/login';
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: authHeaders(options?.headers),
  });
  if (res.status === 401) {
    handleUnauthorized();
    throw new Error('Unauthorized');
  }
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// Notes
export const notesApi = {
  list: (params?: { tag?: string; search?: string; skip?: number; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params?.tag) qs.set('tag', params.tag);
    if (params?.search) qs.set('search', params.search);
    if (params?.skip !== undefined) qs.set('skip', String(params.skip));
    if (params?.limit !== undefined) qs.set('limit', String(params.limit));
    const suffix = qs.toString() ? `?${qs}` : '';
    return request<import('@/types').Note[]>(`/api/notes/${suffix}`);
  },
  get: (id: string) => request<import('@/types').Note>(`/api/notes/${id}`),
  create: (note: Omit<import('@/types').Note, 'id'>) =>
    request<import('@/types').Note>('/api/notes/', {
      method: 'POST',
      body: JSON.stringify(note),
    }),
  update: (id: string, note: import('@/types').Note) =>
    request<import('@/types').Note>(`/api/notes/${id}`, {
      method: 'PUT',
      body: JSON.stringify(note),
    }),
  delete: (id: string) =>
    request<{ status: string; id: string }>(`/api/notes/${id}`, {
      method: 'DELETE',
    }),
  getBacklinks: (id: string) =>
    request<import('@/types').BacklinksResponse>(`/api/notes/${id}/backlinks`),
  getRelated: (id: string) =>
    request<import('@/types').RelatedResponse>(`/api/notes/${id}/related`),
  quickCreate: (note: { title?: string; content: string }) =>
    request<import('@/types').Note>('/api/notes/', {
      method: 'POST',
      body: JSON.stringify({
        title: note.title || 'Quick Note',
        content: note.content,
        frontmatter: {},
        links: [],
        path: `quick-${Date.now()}.md`,
        created: new Date().toISOString(),
        modified: new Date().toISOString(),
      }),
    }),
};

// Search
export const searchApi = {
  search: (q: string, limit = 20) => {
    const qs = new URLSearchParams({ q, limit: String(limit) });
    return request<import('@/types').Note[]>(`/api/search/?${qs}`);
  },
};

// Health
export const healthApi = {
  check: () => request<{ status: string; version: string }>('/health'),
};

// Chat
export const chatApi = {
  chat: async (body: import('@/types').ChatRequest) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 120s timeout for Ollama
    try {
      const res = await fetch(`${API_BASE}/api/chat/`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (res.status === 401) {
        handleUnauthorized();
        throw new Error('Unauthorized');
      }
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`API ${res.status}: ${text}`);
      }
      return (await res.json()) as import('@/types').ChatResponse;
    } catch (err) {
      clearTimeout(timeoutId);
      if (err instanceof Error && err.name === 'AbortError') {
        throw new Error('Chat request timed out (120s). The backend may be busy.');
      }
      throw err;
    }
  },
  stopGeneration: () => request<{ status: string }>('/api/chat/stop', { method: 'POST' }),
  chatStream: (
    body: import('@/types').ChatRequest,
    callbacks: {
      onChunk: (chunk: string) => void;
      onDone: () => void;
      onError: (err: Error) => void;
      // Receives the normalized citations event. `citations` is always an array;
      // `sub_questions` is populated in multi_hop mode (empty otherwise).
      onCitations?: (event: { citations: Citation[]; sub_questions: string[] }) => void;
      onSnippet?: (snippet: { note_id: string; title: string; snippet: string }) => void;
    },
  ): { cancel: () => void } => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180_000); // 180s stream timeout

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/chat/stream`, {
          method: 'POST',
          headers: authHeaders(),
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        if (res.status === 401) {
          handleUnauthorized();
          throw new Error('Unauthorized');
        }
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`API ${res.status}: ${text}`);
        }
        const reader = res.body?.getReader();
        if (!reader) throw new Error('No response body');

        const decoder = new TextDecoder();
        let buffer = '';
        let eventName = 'message';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventName = line.slice(7);
            } else if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (eventName === 'citations') {
                try {
                  // Simple mode sends an array; multi_hop sends
                  // { citations, sub_questions }. Normalize both shapes.
                  const parsed = JSON.parse(data);
                  callbacks.onCitations?.(normalizeCitationsEvent(parsed));
                } catch {
                  // ignore malformed citations
                }
              } else if (eventName === 'message') {
                try {
                  const parsed = JSON.parse(data);
                  if (parsed.chunk !== undefined) {
                    callbacks.onChunk(parsed.chunk);
                  }
                } catch {
                  // ignore malformed message
                }
              } else if (eventName === 'snippet') {
                try {
                  const parsed = JSON.parse(data);
                  callbacks.onSnippet?.(parsed);
                } catch {
                  // ignore malformed snippet
                }
              }
              eventName = 'message';
            } else if (line === '') {
              eventName = 'message';
            }
          }
        }

        clearTimeout(timeoutId);
        callbacks.onDone();
      } catch (err) {
        clearTimeout(timeoutId);
        if (err instanceof Error && err.name === 'AbortError') {
          callbacks.onDone();
        } else {
          callbacks.onError(err instanceof Error ? err : new Error(String(err)));
        }
      }
    })();

    return {
      cancel: () => {
        clearTimeout(timeoutId);
        controller.abort();
      },
    };
  },
};

// Conversations (persisted chat history; single continuous thread per user)
export const conversationsApi = {
  // Returns a page of messages newest-first. Pass the oldest loaded message's
  // `created` as `before` to page further back (scroll-up loading).
  listMessages: (params?: { limit?: number; before?: string }) => {
    const qs = new URLSearchParams();
    if (params?.limit !== undefined) qs.set('limit', String(params.limit));
    if (params?.before) qs.set('before', params.before);
    const suffix = qs.toString() ? `?${qs}` : '';
    return request<import('@/types').ConversationMessage[]>(`/api/conversations/messages${suffix}`);
  },
  // Fetch a single message by id — used to poll in-progress placeholders
  // after a mid-generation reload.
  getMessage: (id: string) =>
    request<import('@/types').ConversationMessage>(`/api/conversations/messages/${id}`),
  clear: () =>
    request<{ status: string; deleted: number }>(`/api/conversations/messages`, {
      method: 'DELETE',
    }),
};

// Graph
export const graphApi = {
  listEntities: (params?: { type?: string; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params?.type) qs.set('type', params.type);
    if (params?.limit !== undefined) qs.set('limit', String(params.limit));
    const suffix = qs.toString() ? `?${qs}` : '';
    return request<import('@/types').Entity[]>(`/api/graph/entities${suffix}`);
  },
  getEntity: (id: string) =>
    request<import('@/types').Entity & { relations: import('@/types').Relation[] }>(
      `/api/graph/entities/${id}`,
    ),
  getEntityRelations: (id: string) =>
    request<import('@/types').Relation[]>(`/api/graph/entities/${id}/relations`),
  listRelations: (limit?: number) => {
    const qs = new URLSearchParams();
    if (limit !== undefined) qs.set('limit', String(limit));
    const suffix = qs.toString() ? `?${qs}` : '';
    return request<import('@/types').Relation[]>(`/api/graph/relations${suffix}`);
  },
  findPaths: (source: string, target: string, max_depth = 3) => {
    const qs = new URLSearchParams({ source, target, max_depth: String(max_depth) });
    return request<import('@/types').GraphPathsResponse>(`/api/graph/paths?${qs}`);
  },
  stats: () => request<import('@/types').GraphStats>('/api/graph/stats'),
  full: (params?: { limit?: number; relationLimit?: number }) => {
    const qs = new URLSearchParams();
    if (params?.limit !== undefined) qs.set('limit', String(params.limit));
    if (params?.relationLimit !== undefined) qs.set('relation_limit', String(params.relationLimit));
    const suffix = qs.toString() ? `?${qs}` : '';
    return request<import('@/types').GraphFullResponse>(`/api/graph/full${suffix}`);
  },
};

// Memory
export const memoryApi = {
  list: (entity?: string) => {
    const qs = new URLSearchParams();
    if (entity) qs.set('entity', entity);
    const suffix = qs.toString() ? `?${qs}` : '';
    return request<import('@/types').Memory[]>(`/api/memory/${suffix}`);
  },
  stats: () => request<import('@/types').MemoryStats>('/api/memory/stats'),
  extract: (note_id: string) =>
    request<import('@/types').Memory[]>('/api/memory/extract', {
      method: 'POST',
      body: JSON.stringify({ note_id }),
    }),
  consolidate: () =>
    request<Record<string, unknown>>('/api/memory/consolidate', { method: 'POST' }),
};

// Review
export const reviewApi = {
  due: () => request<import('@/types').Card[]>('/api/review/due'),
  generateCards: (note_id: string) =>
    request<import('@/types').Card[]>('/api/review/cards', {
      method: 'POST',
      body: JSON.stringify({ note_id }),
    }),
  reviewCard: (card_id: string, rating: number) =>
    request<import('@/types').Card>(`/api/review/cards/${card_id}/review`, {
      method: 'POST',
      body: JSON.stringify({ rating }),
    }),
  stats: () => request<import('@/types').ReviewStats>('/api/review/stats'),
};

// Settings
export const settingsApi = {
  get: () => request<import('@/types').AppSettings>('/api/settings/'),
  update: (body: Partial<import('@/types').AppSettings>) =>
    request<import('@/types').AppSettings>('/api/settings/', {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  listSecrets: () => request<{ keys: string[] }>('/api/settings/secrets'),
  setSecret: (key: string, value: string) =>
    request<{ keys: string[] }>('/api/settings/secrets', {
      method: 'PUT',
      body: JSON.stringify({ key, value }),
    }),
  deleteSecret: (key: string) =>
    request<{ keys: string[] }>(`/api/settings/secrets/${key}`, {
      method: 'DELETE',
    }),
  // Fetch available models for a provider. Pass baseUrl/apiKey to preview
  // models for config the user has typed but not yet saved.
  listModels: (provider: string, opts?: { baseUrl?: string; apiKey?: string; q?: string }) => {
    const qs = new URLSearchParams({ provider });
    if (opts?.baseUrl) qs.set('base_url', opts.baseUrl);
    if (opts?.apiKey) qs.set('api_key', opts.apiKey);
    if (opts?.q) qs.set('q', opts.q);
    return request<import('@/types').ModelsResponse>(`/api/settings/models?${qs}`);
  },
  // Embedding model search per provider. Pass baseUrl/apiKey/q to preview models
  // for unsaved config and to search-as-you-type.
  listEmbeddingModels: (
    provider: string,
    opts?: { baseUrl?: string; apiKey?: string; q?: string },
  ) => {
    const qs = new URLSearchParams({ provider });
    if (opts?.baseUrl) qs.set('base_url', opts.baseUrl);
    if (opts?.apiKey) qs.set('api_key', opts.apiKey);
    if (opts?.q) qs.set('q', opts.q);
    return request<import('@/types').EmbeddingModelsResponse>(
      `/api/settings/embedding-models?${qs}`,
    );
  },
};

// Background reembed job (vector store rebuild on embedding change)
export const reembedApi = {
  status: () => request<import('@/types').ReembedStatus>('/api/admin/reembed/status'),
};

// Daily LLM budget status (banner + pause/resume state)
export const budgetApi = {
  status: () => request<import('@/types').BudgetStatus>('/api/budget/status'),
};

// Personal access tokens (machine clients / MCP)
export const tokensApi = {
  list: () => request<import('@/types').AccessToken[]>('/api/settings/tokens/'),
  create: (name: string, scope: 'read' | 'full') =>
    request<import('@/types').AccessTokenCreated>('/api/settings/tokens/', {
      method: 'POST',
      body: JSON.stringify({ name, scope }),
    }),
  remove: (id: string) =>
    request<{ ok: boolean }>(`/api/settings/tokens/${id}`, { method: 'DELETE' }),
};

// First-run setup wizard
export const setupApi = {
  status: () => request<{ configured: boolean; auth_required: boolean }>('/api/setup/status'),
  submit: (payload: import('@/types').SetupPayload) =>
    request<{ configured: boolean }>('/api/setup/', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  test: (payload: {
    llm_provider: string;
    ollama_base_url?: string;
    ollama_api_key?: string;
    openrouter_base_url?: string;
    openrouter_api_key?: string;
  }) =>
    request<{ ok: boolean; detail: string }>('/api/setup/test', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
};

// Auth — login/register use OAuth2 form encoding (not JSON).
export const authApi = {
  login: async (username: string, password: string) => {
    const form = new URLSearchParams({ username, password });
    const res = await fetch(`${API_BASE}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form.toString(),
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`API ${res.status}: ${body}`);
    }
    const data = (await res.json()) as { access_token: string; token_type: string };
    setToken(data.access_token);
    return data;
  },
  register: async (username: string, password: string) => {
    const form = new URLSearchParams({ username, password });
    const res = await fetch(`${API_BASE}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form.toString(),
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`API ${res.status}: ${body}`);
    }
    const data = (await res.json()) as { access_token: string; token_type: string };
    setToken(data.access_token);
    return data;
  },
  me: () => request<{ username: string; email: string }>('/api/auth/me'),
  logout: () => clearToken(),
};

// Stats
export const statsApi = {
  get: () => request<import('@/types').StatsResponse>('/api/stats/'),
};

// Entities (flat path)
export const entitiesApi = {
  list: (params?: { type?: string; search?: string; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params?.type) qs.set('type', params.type);
    if (params?.search) qs.set('search', params.search);
    if (params?.limit !== undefined) qs.set('limit', String(params.limit));
    const suffix = qs.toString() ? `?${qs}` : '';
    return request<import('@/types').Entity[]>(`/api/entities/${suffix}`);
  },
  get: (id: string) => request<import('@/types').Entity>(`/api/entities/${id}`),
  subgraph: (id: string, depth = 2) => {
    const qs = new URLSearchParams({ depth: String(depth) });
    return request<import('@/types').SubgraphResponse>(`/api/entities/${id}/subgraph?${qs}`);
  },
};

// Export
export const exportApi = {
  json: () => request<import('@/types').ExportJsonResponse>('/api/export/json'),
};

// Retrieval
export const retrievalApi = {
  hybrid: (body: import('@/types').HybridSearchRequest) =>
    request<import('@/types').HybridSearchResult[]>('/api/retrieval/hybrid', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
};

// Buddy
export const buddyApi = {
  getProfile: () => request<import('@/types').BuddyProfile>('/api/buddy/profile'),
  updateProfile: (body: import('@/types').BuddyProfile) =>
    request<import('@/types').BuddyProfile>('/api/buddy/profile', {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  getGreeting: () => request<import('@/types').BuddyGreeting>('/api/buddy/greeting'),
  getCards: () => request<import('@/types').BuddyCards>('/api/buddy/cards'),
};
