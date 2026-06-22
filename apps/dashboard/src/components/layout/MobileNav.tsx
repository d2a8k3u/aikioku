'use client';

import { useId } from 'react';

import { HudDrawer } from '@/components/hud/HudDrawer';
import { LockButton } from './LockButton';
import { NavItem } from './NavItem';
import { Logo, StatusDot } from './RailParts';
import { NAV_ITEMS, SETTINGS_ITEM } from './navConfig';

function MenuIcon() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
    >
      <line x1="4" y1="7" x2="20" y2="7" />
      <line x1="4" y1="12" x2="20" y2="12" />
      <line x1="4" y1="17" x2="20" y2="17" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
    >
      <line x1="6" y1="6" x2="18" y2="18" />
      <line x1="18" y1="6" x2="6" y2="18" />
    </svg>
  );
}

type MobileNavProps = {
  readonly open: boolean;
  readonly onOpen: () => void;
  readonly onClose: () => void;
};

export function MobileNav({ open, onOpen, onClose }: MobileNavProps) {
  const drawerId = useId();
  const wordmarkId = useId();

  return (
    <>
      <header className="fixed inset-x-0 top-0 z-40 flex h-12 items-center gap-3 border-b border-aiki-border-subtle bg-[rgba(7,10,15,0.6)] px-3 backdrop-blur-[14px] md:hidden">
        <button
          type="button"
          onClick={onOpen}
          aria-label="Open navigation menu"
          aria-expanded={open}
          aria-controls={drawerId}
          className="flex h-10 w-10 items-center justify-center rounded-lg text-aiki-text-tertiary transition-colors hover:bg-white/[0.04] hover:text-aiki-text-secondary"
        >
          <MenuIcon />
        </button>
        <Logo wordmark="always" />
        <div className="flex-1" />
        <StatusDot />
      </header>

      <HudDrawer
        open={open}
        onClose={onClose}
        labelledById={wordmarkId}
        className="border-r border-aiki-border-subtle bg-aiki-panel/95 px-3 py-5 backdrop-blur-[14px]"
      >
        <div id={drawerId} className="flex h-full flex-col">
          <div className="mb-7 flex items-center justify-between">
            <Logo wordmark="always" wordmarkId={wordmarkId} onNavigate={onClose} />
            <button
              type="button"
              onClick={onClose}
              aria-label="Close navigation menu"
              className="flex h-9 w-9 items-center justify-center rounded-lg text-aiki-text-tertiary transition-colors hover:bg-white/[0.04] hover:text-aiki-text-secondary"
            >
              <CloseIcon />
            </button>
          </div>

          <nav aria-label="Primary" className="flex flex-col gap-1.5">
            {NAV_ITEMS.map((item) => (
              <NavItem key={item.href} item={item} variant="expanded" onNavigate={onClose} />
            ))}
          </nav>

          <div className="flex-1" />

          <NavItem item={SETTINGS_ITEM} variant="expanded" onNavigate={onClose} />
          <LockButton variant="expanded" onAction={onClose} />
          <div className="mt-4 px-3">
            <StatusDot />
          </div>
        </div>
      </HudDrawer>
    </>
  );
}
