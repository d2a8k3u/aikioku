'use client';

import { memo } from 'react';
import Markdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { cn } from '@/lib/cn';

// Offset rendered headings so the note title (<h2>) stays the highest content
// heading and the document outline remains valid.
const components: Components = {
  h1: ({ node, ...props }) => <h3 {...props} />,
  h2: ({ node, ...props }) => <h4 {...props} />,
  h3: ({ node, ...props }) => <h5 {...props} />,
  h4: ({ node, ...props }) => <h6 {...props} />,
  h5: ({ node, ...props }) => <h6 {...props} />,
  h6: ({ node, ...props }) => <h6 {...props} />,
  a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
};

type MarkdownPreviewProps = {
  readonly content: string;
  readonly className?: string;
  readonly emptyLabel?: string;
};

function MarkdownPreviewImpl({
  content,
  className,
  emptyLabel = 'Nothing to preview — switch to Edit to start writing.',
}: MarkdownPreviewProps) {
  if (!content.trim()) {
    return <p className="font-mono text-sm text-aiki-text-muted">{emptyLabel}</p>;
  }

  return (
    <div className={cn('aiki-md', className)}>
      <Markdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </Markdown>
    </div>
  );
}

export const MarkdownPreview = memo(MarkdownPreviewImpl);
