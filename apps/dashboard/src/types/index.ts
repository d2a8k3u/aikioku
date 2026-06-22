// TypeScript types matching backend models

export interface Note {
  id: string;
  title: string;
  content: string;
  frontmatter: Record<string, unknown>;
  links: string[];
  path: string;
  created: string;
  modified: string;
}

export type MemoryTier = 'hot' | 'warm' | 'cold';

export interface Memory {
  id: string;
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
  source: string;
  created: string;
  modified: string;
  vitality_score: number;
  tier: MemoryTier;
}

export type CardType = 'cloze' | 'qa' | 'connection';
export type CardStatus = 'new' | 'learning' | 'review' | 'suspended';

export interface Card {
  id: string;
  note_id: string;
  type: CardType;
  front: string;
  back: string;
  ease_factor: number;
  interval: number;
  repetitions: number;
  next_review: string;
  status: CardStatus;
}

export type EntityType =
  | 'Person'
  | 'Place'
  | 'Concept'
  | 'Project'
  | 'Event'
  | 'Organization'
  | 'Document'
  | 'Task';

export interface Entity {
  id: string;
  name: string;
  type: EntityType;
  aliases: string[];
  properties: Record<string, unknown>;
  confidence: number;
  source_note_ids: string[];
}

export type RelationType =
  | 'works_at'
  | 'created'
  | 'depends_on'
  | 'related_to'
  | 'part_of'
  | 'located_in'
  | 'mentions'
  | 'follows';

export interface Relation {
  id: string;
  source_entity_id: string;
  target_entity_id: string;
  type: RelationType;
  confidence: number;
  properties: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
}

// A persisted chat message returned by /api/conversations/messages.
export interface ConversationMessage {
  id: string;
  user_id: string;
  role: 'user' | 'assistant';
  content: string;
  citations: { note_id: string; title?: string; snippet: string }[];
  sub_questions: string[];
  created: string;
  in_progress: boolean;
}

export interface ReviewRating {
  card_id: string;
  quality: number; // 0-5 SM-2 quality rating
}

export interface AppSettings {
  llm_provider: string;
  embedding_provider: string;
  embedding_model: string;
  auto_extract: boolean;
  auto_consolidation: boolean;
  // Extended (returned by the backend; optional for older callers).
  llm_model?: string;
  ollama_base_url?: string;
  ollama_embedding_base_url?: string;
  ollama_embedding_model?: string;
  openrouter_base_url?: string;
  embedding_dimension?: number;
  embedding_strict?: boolean;
  hf_embedding_model?: string;
  openai_base_url?: string;
  openai_embedding_model?: string;
  auth_required?: boolean;
  cors_origins?: string;
  secret_keys?: string[];
  llm_daily_budget_usd?: number;
}

// Personal access token (machine clients / MCP). The plaintext `token` is
// present only on the create response (AccessTokenCreated), never on list.
export interface AccessToken {
  id: string;
  name: string;
  scope: 'read' | 'full';
  prefix: string;
  created: string;
  last_used: string | null;
}

export interface AccessTokenCreated extends AccessToken {
  token: string;
}

export interface SetupPayload {
  llm_provider?: string;
  llm_model?: string;
  ollama_base_url?: string;
  ollama_embedding_base_url?: string;
  ollama_embedding_model?: string;
  openrouter_base_url?: string;
  embedding_provider?: string;
  embedding_model?: string;
  embedding_dimension?: number;
  embedding_strict?: boolean;
  hf_embedding_model?: string;
  openai_base_url?: string;
  openai_embedding_model?: string;
  auto_extract?: boolean;
  auto_consolidation?: boolean;
  llm_daily_budget_usd?: number;
  cors_origins?: string;
  ollama_api_key?: string;
  openrouter_api_key?: string;
  hf_api_key?: string;
  openai_api_key?: string;
  account: { username: string; password: string; email?: string };
}

// Background reembed job status (vector store rebuild on embedding change).
export interface ReembedStatus {
  state: 'idle' | 'running' | 'failed';
  target_fp: string | null;
  processed_notes: number;
  total_notes: number;
  processed_convs: number;
  total_convs: number;
  error: string | null;
}

// Daily LLM budget state. `paused` means the cap is reached and LLM-backed
// ingestion is deferred to a queue until the UTC-midnight reset; `warning` is
// the near-limit band before that.
export interface BudgetStatus {
  state: 'active' | 'warning' | 'paused';
  daily_budget: number;
  today_cost: number;
  remaining: number;
  fraction: number;
  pending_count: number;
  warning_fraction: number;
  reset_at: string;
}

export interface StatsResponse {
  notes: number;
  entities: number;
  relations: number;
  memories: number;
  cards: number;
  version: string;
}

export interface ChatRequest {
  query: string;
  mode: 'simple' | 'multi_hop';
}

export interface ChatResponse {
  response: string;
  citations: { note_id: string; title: string; snippet: string }[];
  mode: string;
  sub_questions?: string[];
}

export interface GraphStats {
  entities: number;
  relations: number;
  types: Record<string, number>;
}

export interface GraphPathsResponse {
  paths: Entity[][];
}

export interface MemoryStats {
  total: number;
  hot: number;
  warm: number;
  cold: number;
}

export interface ExtractRequest {
  note_id: string;
}

export interface ReviewStats {
  total: number;
  due: number;
  new: number;
  learning: number;
  review: number;
  suspended: number;
}

export interface HybridSearchRequest {
  query: string;
  limit: number;
}

export interface HybridSearchResult {
  note_id: string;
  score: number;
  source: string;
  snippet: string;
  metadata: Record<string, unknown>;
}

export interface SubgraphResponse {
  root_id: string;
  depth: number;
  nodes: Entity[];
  edges: Relation[];
}

export interface GraphEdge {
  source_entity_id: string;
  target_entity_id: string;
  type: RelationType;
  confidence: number;
}

export interface GraphFullResponse {
  nodes: Entity[];
  edges: GraphEdge[];
}

export interface ExportJsonResponse {
  version: string;
  notes: Note[];
  entities: Entity[];
  relations: Relation[];
  memories: Memory[];
  cards: Record<string, unknown>[];
}

// BOD 4: Provider model listing
export interface ProviderModel {
  id: string;
  name: string;
  type: 'chat' | 'embedding';
  family?: string;
  parameter_size?: string;
  context_length?: number;
  pricing_prompt?: number;
  pricing_completion?: number;
}

export interface ModelsResponse {
  models: ProviderModel[];
  error: string | null;
}

// BOD 5: Embedding model listing
export interface EmbeddingModel {
  id: string;
  name: string;
  provider: 'ollama' | 'huggingface' | 'openai';
  dimensions: number | null;
  dynamic?: boolean;
}

export interface EmbeddingModelsResponse {
  models: EmbeddingModel[];
  error: string | null;
}

// Backlinks for a note
export interface BacklinksResponse {
  backlinks: { id: string; title: string; snippet: string }[];
}

// Related entities, notes, and surprise connections for a note
export interface RelatedResponse {
  entities: Entity[];
  related_notes: { id: string; title: string; shared_entities: string[] }[];
  surprise: { path: string[]; entities: string[] };
  insight?: string;
}

// Buddy types
export interface BuddyProfile {
  name: string;
  tone: 'warm' | 'focused' | 'playful';
  lm_provider: 'local' | 'remote' | 'router';
}

export interface BuddyGreeting {
  greeting: string;
  buddy_line: string;
  about_you: string[];
}

export interface BuddyCard {
  kicker: string;
  title: string;
  body: string;
  action: string;
  rgb: string;
  go: string;
}

export interface BuddyCards {
  cards: BuddyCard[];
}
