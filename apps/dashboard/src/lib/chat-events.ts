// Pure helpers for normalizing chat SSE event payloads.
//
// The backend `citations` SSE event has two shapes:
//   - simple mode:    an ARRAY of Citation
//   - multi_hop mode: an OBJECT { citations: Citation[], sub_questions: string[] }
// This module normalizes both into a single shape so the UI never has to care.

export interface Citation {
  note_id: string;
  title: string;
  snippet: string;
}

export interface NormalizedCitationsEvent {
  citations: Citation[];
  sub_questions: string[];
}

function isCitation(value: unknown): value is Citation {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v.note_id === 'string' && typeof v.title === 'string';
}

/**
 * Normalize a parsed `citations` SSE payload into { citations, sub_questions }.
 *
 * Accepts:
 *   - an array of citation objects (simple mode)
 *   - an object { citations, sub_questions } (multi_hop mode)
 * Anything malformed degrades to empty arrays rather than throwing.
 */
export function normalizeCitationsEvent(payload: unknown): NormalizedCitationsEvent {
  if (Array.isArray(payload)) {
    return { citations: payload.filter(isCitation), sub_questions: [] };
  }

  if (typeof payload === 'object' && payload !== null) {
    const obj = payload as Record<string, unknown>;
    const citations = Array.isArray(obj.citations) ? obj.citations.filter(isCitation) : [];
    const sub_questions = Array.isArray(obj.sub_questions)
      ? obj.sub_questions.filter((q): q is string => typeof q === 'string')
      : [];
    return { citations, sub_questions };
  }

  return { citations: [], sub_questions: [] };
}
