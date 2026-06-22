import { useState } from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { MobileNav } from '../MobileNav';

vi.mock('next/navigation', () => ({
  usePathname: () => '/',
}));

function Harness() {
  const [open, setOpen] = useState(false);
  return <MobileNav open={open} onOpen={() => setOpen(true)} onClose={() => setOpen(false)} />;
}

describe('MobileNav', () => {
  it('opens the drawer via the hamburger and reflects aria-expanded', () => {
    render(<Harness />);
    const trigger = screen.getByRole('button', { name: 'Open navigation menu' });

    expect(trigger).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('dialog')).toBeNull();

    fireEvent.click(trigger);

    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it('closes the drawer on Escape', () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Open navigation menu' }));
    expect(screen.getByRole('dialog')).toBeInTheDocument();

    fireEvent.keyDown(window, { key: 'Escape' });

    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('locks body scroll while open and restores it on close', () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Open navigation menu' }));
    expect(document.body.style.overflow).toBe('hidden');

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(document.body.style.overflow).toBe('');
  });
});
