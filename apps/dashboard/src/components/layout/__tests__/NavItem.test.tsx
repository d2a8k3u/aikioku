import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { NavItem } from '../NavItem';
import { NAV_ITEMS } from '../navConfig';

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
}));

const chat = NAV_ITEMS.find((i) => i.href === '/chat')!;
const home = NAV_ITEMS.find((i) => i.href === '/')!;

describe('NavItem', () => {
  it('marks the active route with aria-current="page"', () => {
    render(<NavItem item={chat} />);
    expect(screen.getByRole('link', { name: chat.label })).toHaveAttribute('aria-current', 'page');
  });

  it('does not set aria-current on inactive routes', () => {
    render(<NavItem item={home} />);
    expect(screen.getByRole('link', { name: home.label })).not.toHaveAttribute('aria-current');
  });

  it('keeps the label as the accessible name in the collapsed rail variant', () => {
    render(<NavItem item={home} variant="rail" />);
    expect(screen.getByRole('link', { name: home.label })).toBeInTheDocument();
  });
});
