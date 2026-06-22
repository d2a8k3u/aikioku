import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';

function HelloWorld() {
  return <div data-testid="hello">Hello, Aikioku!</div>;
}

describe('smoke test', () => {
  it('renders a simple component', () => {
    render(<HelloWorld />);
    expect(screen.getByTestId('hello')).toBeInTheDocument();
    expect(screen.getByText('Hello, Aikioku!')).toBeInTheDocument();
  });
});
