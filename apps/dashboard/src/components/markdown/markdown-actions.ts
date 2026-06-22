export type MdAction =
  | 'h1'
  | 'h2'
  | 'bold'
  | 'italic'
  | 'inlineCode'
  | 'bulletList'
  | 'orderedList'
  | 'quote'
  | 'codeBlock'
  | 'link';

export interface MdResult {
  value: string;
  selStart: number;
  selEnd: number;
}

function applyWrap(value: string, selStart: number, selEnd: number, marker: string): MdResult {
  const selected = value.slice(selStart, selEnd);
  const newValue = value.slice(0, selStart) + marker + selected + marker + value.slice(selEnd);
  if (selStart === selEnd) {
    const caret = selStart + marker.length;
    return { value: newValue, selStart: caret, selEnd: caret };
  }
  return { value: newValue, selStart: selStart + marker.length, selEnd: selEnd + marker.length };
}

function applyLinePrefix(
  value: string,
  selStart: number,
  selEnd: number,
  makePrefix: (lineIndex: number) => string,
): MdResult {
  const blockStart = value.lastIndexOf('\n', selStart - 1) + 1;
  // When the selection ends exactly on a line boundary, don't prefix the next line.
  const refEnd = selEnd > selStart && value[selEnd - 1] === '\n' ? selEnd - 1 : selEnd;
  let blockEnd = value.indexOf('\n', refEnd);
  if (blockEnd === -1) blockEnd = value.length;

  const block = value.slice(blockStart, blockEnd);
  const prefixed = block.split('\n').map((line, i) => makePrefix(i) + line);
  const newBlock = prefixed.join('\n');
  const newValue = value.slice(0, blockStart) + newBlock + value.slice(blockEnd);

  const firstPrefixLen = makePrefix(0).length;
  const added = newBlock.length - block.length;
  return {
    value: newValue,
    selStart: selStart + firstPrefixLen,
    selEnd: selEnd + added,
  };
}

function applyLink(value: string, selStart: number, selEnd: number): MdResult {
  const selected = value.slice(selStart, selEnd);
  const text = selected || 'text';
  const url = 'url';
  const inserted = `[${text}](${url})`;
  const newValue = value.slice(0, selStart) + inserted + value.slice(selEnd);
  if (selected) {
    const urlStart = selStart + 1 + text.length + 2; // past "[text]("
    return { value: newValue, selStart: urlStart, selEnd: urlStart + url.length };
  }
  const textStart = selStart + 1;
  return { value: newValue, selStart: textStart, selEnd: textStart + text.length };
}

function applyCodeBlock(value: string, selStart: number, selEnd: number): MdResult {
  const selected = value.slice(selStart, selEnd);
  const lead = selStart > 0 && value[selStart - 1] !== '\n' ? '\n' : '';
  const trail = selEnd < value.length && value[selEnd] !== '\n' ? '\n' : '';
  const inserted = `${lead}\`\`\`\n${selected}\n\`\`\`${trail}`;
  const newValue = value.slice(0, selStart) + inserted + value.slice(selEnd);
  const contentStart = selStart + lead.length + 4; // past lead + "```\n"
  return {
    value: newValue,
    selStart: contentStart,
    selEnd: contentStart + selected.length,
  };
}

export function applyMarkdown(
  value: string,
  selStart: number,
  selEnd: number,
  action: MdAction,
): MdResult {
  switch (action) {
    case 'bold':
      return applyWrap(value, selStart, selEnd, '**');
    case 'italic':
      return applyWrap(value, selStart, selEnd, '*');
    case 'inlineCode':
      return applyWrap(value, selStart, selEnd, '`');
    case 'h1':
      return applyLinePrefix(value, selStart, selEnd, () => '# ');
    case 'h2':
      return applyLinePrefix(value, selStart, selEnd, () => '## ');
    case 'quote':
      return applyLinePrefix(value, selStart, selEnd, () => '> ');
    case 'bulletList':
      return applyLinePrefix(value, selStart, selEnd, () => '- ');
    case 'orderedList':
      return applyLinePrefix(value, selStart, selEnd, (i) => `${i + 1}. `);
    case 'codeBlock':
      return applyCodeBlock(value, selStart, selEnd);
    case 'link':
      return applyLink(value, selStart, selEnd);
  }
}
