'use client';

import { type ReactNode, useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';

import { useMediaQuery } from '@/hooks/useMediaQuery';
import { LockButton } from './LockButton';
import { MobileNav } from './MobileNav';
import { NavItem } from './NavItem';
import { Logo, StatusDot } from './RailParts';
import { NAV_ITEMS, SETTINGS_ITEM } from './navConfig';

// ── Rail Nav (tablet + desktop) ─────────────────────────

function RailNav() {
  return (
    <nav
      aria-label="Primary"
      className="hidden shrink-0 flex-col border-r border-aiki-border-subtle bg-[rgba(7,10,15,0.4)] py-5 backdrop-blur-[14px] md:flex md:w-[72px] lg:w-60 lg:px-3"
    >
      <div className="mb-7 lg:px-3">
        <Logo wordmark="responsive" />
      </div>

      <div className="flex flex-col gap-1.5">
        {NAV_ITEMS.map((item) => (
          <NavItem key={item.href} item={item} variant="rail" />
        ))}
      </div>

      <div className="flex-1" />

      <NavItem item={SETTINGS_ITEM} variant="rail" />
      <LockButton variant="rail" />

      <div className="mt-4 flex justify-center lg:justify-start lg:px-3">
        <StatusDot label="responsive" />
      </div>
    </nav>
  );
}

// ── Layout ──────────────────────────────────────────────

type RailLayoutProps = {
  readonly children: ReactNode;
};

export function RailLayout({ children }: RailLayoutProps) {
  const pathname = usePathname();
  const [navOpen, setNavOpen] = useState(false);
  const isDesktop = useMediaQuery('(min-width: 768px)');

  useEffect(() => {
    if (isDesktop) setNavOpen(false);
  }, [isDesktop]);

  useEffect(() => {
    setNavOpen(false);
  }, [pathname]);

  return (
    <div
      id="app-shell"
      style={{
        position: 'fixed',
        inset: 0,
        background: 'radial-gradient(125% 95% at 50% -8%, #0c1119 0%, #080b10 46%, #05070b 100%)',
        overflow: 'hidden',
      }}
    >
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-2 focus:top-2 focus:z-[70] focus:rounded-md focus:bg-aiki-panel focus:px-3 focus:py-2 focus:font-mono focus:text-xs focus:text-aiki-text focus:outline focus:outline-1 focus:outline-aiki-accent"
      >
        Skip to content
      </a>

      {/* Neural background canvas placeholder */}
      <canvas id="neural-bg" style={{ position: 'absolute', inset: 0, opacity: 0.5 }} />

      {/* Ambient glow blobs */}
      <div
        style={{
          position: 'absolute',
          top: '-12%',
          left: '50%',
          transform: 'translateX(-50%)',
          width: 820,
          height: 560,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(184,115,51,0.13) 0%, rgba(184,115,51,0) 64%)',
          filter: 'blur(22px)',
          animation: 'breatheSlow 10s ease-in-out infinite',
          pointerEvents: 'none',
        }}
      />
      <div
        style={{
          position: 'absolute',
          bottom: '-14%',
          right: '6%',
          width: 560,
          height: 460,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(52,214,196,0.1) 0%, rgba(52,214,196,0) 64%)',
          filter: 'blur(24px)',
          animation: 'breatheSlow 13s ease-in-out infinite 2s',
          pointerEvents: 'none',
        }}
      />

      <MobileNav open={navOpen} onOpen={() => setNavOpen(true)} onClose={() => setNavOpen(false)} />

      {/* Content */}
      <div style={{ position: 'relative', height: '100vh', display: 'flex', zIndex: 1 }}>
        <RailNav />
        <main
          id="main-content"
          tabIndex={-1}
          className="flex min-w-0 flex-1 flex-col pt-12 md:pt-0"
        >
          <div
            style={{
              flex: 1,
              minHeight: 0,
              display: 'flex',
              flexDirection: 'column',
              animation: 'fadeSlideIn .55s cubic-bezier(.22,1,.36,1)',
            }}
          >
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
