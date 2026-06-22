'use client';

import { useState } from 'react';
import { useReembedStatus } from './ReembedProvider';

export default function ReembedBanner() {
  const status = useReembedStatus();
  // Track which error we dismissed so a NEW failure re-shows the banner.
  const [dismissedError, setDismissedError] = useState<string | null>(null);

  if (status.state === 'running') {
    const done = status.processed_notes + status.processed_convs;
    const total = status.total_notes + status.total_convs;
    return (
      <div className="flex items-center gap-2 border-b border-aiki-accent/30 bg-aiki-accent/10 px-4 py-2 text-sm text-aiki-accent">
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
          className="flex-shrink-0 "
        >
          <line x1="12" y1="2" x2="12" y2="6" />
          <line x1="12" y1="18" x2="12" y2="22" />
          <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
          <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
          <line x1="2" y1="12" x2="6" y2="12" />
          <line x1="18" y1="12" x2="22" y2="12" />
          <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
          <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
        </svg>
        <span>
          Knowledge is being processed — search results may be incomplete.
          {total > 0 && (
            <span className="ml-1 text-aiki-accent/80">
              ({done}/{total})
            </span>
          )}
        </span>
      </div>
    );
  }

  if (status.state === 'failed' && status.error !== dismissedError) {
    return (
      <div className="flex items-center gap-2 border-b border-aiki-danger/30 bg-aiki-danger/10 px-4 py-2 text-sm text-aiki-danger">
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
          className="flex-shrink-0"
        >
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
          <line x1="12" y1="9" x2="12" y2="13" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
        <span className="flex-1">
          Reembedding failed{status.error ? `: ${status.error}` : ''}. Your previous embeddings are
          unchanged.
        </span>
        <button
          onClick={() => setDismissedError(status.error)}
          aria-label="Dismiss"
          className="rounded p-1 hover:bg-aiki-danger/20 transition-colors"
        >
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
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
    );
  }

  return null;
}
