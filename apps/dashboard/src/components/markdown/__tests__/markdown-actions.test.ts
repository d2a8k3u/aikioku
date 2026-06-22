import { describe, expect, it } from 'vitest';

import { applyMarkdown } from '../markdown-actions';

describe('applyMarkdown — wrap actions', () => {
  it('wraps a selection in bold markers and keeps the inner text selected', () => {
    const r = applyMarkdown('hello world', 6, 11, 'bold');
    expect(r.value).toBe('hello **world**');
    expect(r.value.slice(r.selStart, r.selEnd)).toBe('world');
  });

  it('inserts empty bold markers with the caret between them', () => {
    const r = applyMarkdown('', 0, 0, 'bold');
    expect(r.value).toBe('****');
    expect(r.selStart).toBe(2);
    expect(r.selEnd).toBe(2);
  });

  it('wraps a selection in single asterisks for italic', () => {
    const r = applyMarkdown('hi there', 3, 8, 'italic');
    expect(r.value).toBe('hi *there*');
    expect(r.value.slice(r.selStart, r.selEnd)).toBe('there');
  });

  it('wraps a selection in backticks for inline code', () => {
    const r = applyMarkdown('run npm', 4, 7, 'inlineCode');
    expect(r.value).toBe('run `npm`');
  });
});

describe('applyMarkdown — line-prefix actions', () => {
  it('prefixes a single line with an H1 marker', () => {
    const r = applyMarkdown('title', 0, 0, 'h1');
    expect(r.value).toBe('# title');
    expect(r.selStart).toBe(2);
  });

  it('prefixes a single line with an H2 marker', () => {
    const r = applyMarkdown('title', 0, 5, 'h2');
    expect(r.value).toBe('## title');
  });

  it('prefixes only the current line, not the whole document', () => {
    const value = 'first\nsecond\nthird';
    const r = applyMarkdown(value, 6, 6, 'quote'); // caret on "second"
    expect(r.value).toBe('first\n> second\nthird');
  });

  it('prefixes every line in a multi-line selection with a bullet', () => {
    const value = 'a\nb\nc';
    const r = applyMarkdown(value, 0, 5, 'bulletList');
    expect(r.value).toBe('- a\n- b\n- c');
  });

  it('numbers each line incrementally for an ordered list', () => {
    const value = 'a\nb\nc';
    const r = applyMarkdown(value, 0, 5, 'orderedList');
    expect(r.value).toBe('1. a\n2. b\n3. c');
  });

  it('does not prefix the trailing line when the selection ends on a boundary', () => {
    const value = 'a\nb\nc';
    const r = applyMarkdown(value, 0, 2, 'bulletList'); // selects "a\n"
    expect(r.value).toBe('- a\nb\nc');
  });
});

describe('applyMarkdown — link', () => {
  it('uses the selection as link text and selects the url placeholder', () => {
    const r = applyMarkdown('see docs', 4, 8, 'link');
    expect(r.value).toBe('see [docs](url)');
    expect(r.value.slice(r.selStart, r.selEnd)).toBe('url');
  });

  it('inserts a text placeholder and selects it when nothing is selected', () => {
    const r = applyMarkdown('', 0, 0, 'link');
    expect(r.value).toBe('[text](url)');
    expect(r.value.slice(r.selStart, r.selEnd)).toBe('text');
  });
});

describe('applyMarkdown — code block', () => {
  it('fences an empty selection on its own lines with the caret inside', () => {
    const r = applyMarkdown('', 0, 0, 'codeBlock');
    expect(r.value).toBe('```\n\n```');
    expect(r.selStart).toBe(4); // after "```\n"
    expect(r.selEnd).toBe(4);
  });

  it('adds a leading newline when not at a line start', () => {
    const r = applyMarkdown('text', 4, 4, 'codeBlock');
    expect(r.value).toBe('text\n```\n\n```');
  });

  it('keeps the selected text inside the fence and selected', () => {
    const r = applyMarkdown('code', 0, 4, 'codeBlock');
    expect(r.value).toBe('```\ncode\n```');
    expect(r.value.slice(r.selStart, r.selEnd)).toBe('code');
  });
});
