import { type ComponentType } from 'react';

export type NavItemConfig = {
  readonly href: string;
  readonly icon: ComponentType;
  readonly label: string;
};

function HomeIcon() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      width="19"
      height="19"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
    >
      <path d="M4 11.5 12 4l8 7.5" />
      <path d="M6 10v9h12v-9" />
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      width="19"
      height="19"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
    >
      <path d="M4 5h16v11H9l-5 4z" />
    </svg>
  );
}

function NotesIcon() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      width="19"
      height="19"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
    >
      <path d="M7 3h7l5 5v13H7z" />
      <path d="M13 3v6h6" />
      <path d="M10 13h6M10 17h5" />
    </svg>
  );
}

function RecallIcon() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      width="19"
      height="19"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
    >
      <rect x="4" y="6" width="13" height="14" rx="2" />
      <path d="M8 6V4h12v14h-2" />
    </svg>
  );
}

function GraphIcon() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      width="19"
      height="19"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
    >
      <circle cx="6" cy="7" r="2.3" />
      <circle cx="18" cy="6" r="2.3" />
      <circle cx="17" cy="18" r="2.3" />
      <circle cx="7" cy="17" r="2.3" />
      <path d="M8 8l8-1M8 16l7 1M7 9v6M16 8v8" />
    </svg>
  );
}

function SettingsIcon() {
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
    >
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2" />
    </svg>
  );
}

export const NAV_ITEMS: readonly NavItemConfig[] = [
  { href: '/', icon: HomeIcon, label: 'HOME' },
  { href: '/chat', icon: ChatIcon, label: 'TALK' },
  { href: '/notes', icon: NotesIcon, label: 'THOUGHTS' },
  { href: '/review', icon: RecallIcon, label: 'REVIEW' },
  { href: '/graph', icon: GraphIcon, label: 'GRAPH' },
];

export const SETTINGS_ITEM: NavItemConfig = {
  href: '/settings',
  icon: SettingsIcon,
  label: 'SETTINGS',
};
