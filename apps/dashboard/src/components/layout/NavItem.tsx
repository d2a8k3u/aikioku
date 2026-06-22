'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

import { cn } from '@/lib/cn';
import { type NavItemConfig } from './navConfig';

type NavItemProps = {
  readonly item: NavItemConfig;
  readonly variant?: 'rail' | 'expanded';
  readonly onNavigate?: () => void;
};

export function NavItem({ item, variant = 'rail', onNavigate }: NavItemProps) {
  const pathname = usePathname();
  const { href, icon: Icon, label } = item;
  const isActive = pathname === href || (href !== '/' && pathname.startsWith(href));
  const expanded = variant === 'expanded';

  return (
    <Link
      href={href}
      onClick={onNavigate}
      aria-current={isActive ? 'page' : undefined}
      className={cn(
        'group relative flex items-center rounded-xl py-2.5 transition-colors',
        isActive
          ? 'bg-aiki-accent-bg text-aiki-accent-hover'
          : 'text-aiki-text-tertiary hover:bg-white/[0.04]',
        expanded
          ? 'justify-start gap-3 px-3'
          : 'mx-2 justify-center px-0 lg:mx-0 lg:justify-start lg:gap-3 lg:px-3',
      )}
    >
      {isActive && (
        <span
          aria-hidden="true"
          className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded bg-aiki-accent shadow-[0_0_8px_#B87333]"
        />
      )}
      <Icon />
      <span
        className={cn(
          'font-mono font-medium uppercase tracking-[0.12em]',
          expanded ? 'text-[11px]' : 'sr-only lg:not-sr-only lg:text-[11px]',
        )}
      >
        {label}
      </span>
      {!expanded && (
        <span
          aria-hidden="true"
          className="pointer-events-none absolute left-full top-1/2 z-20 ml-2 -translate-y-1/2 whitespace-nowrap rounded-md border border-aiki-border-subtle bg-aiki-elevated px-2 py-1 font-mono text-[10px] uppercase tracking-[0.12em] text-aiki-text-secondary opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 lg:hidden"
        >
          {label}
        </span>
      )}
    </Link>
  );
}
