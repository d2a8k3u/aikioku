'use client';

import { cn } from '@/lib/cn';

export function HudDivider({ className = '' }: { readonly className?: string }) {
  return <hr className={cn('border-t border-aiki-border-subtle', className)} />;
}
