import type { Metadata } from 'next';
import './globals.css';
import SetupGate from '@/components/SetupGate';

export const metadata: Metadata = {
  title: 'Aikioku — AI-Augmented PKM',
  description:
    'AI-augmented personal knowledge management with knowledge graphs, RAG, and spaced repetition.',
  icons: {
    icon: '/favicon.ico',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        {/* Loaded via <link> rather than next/font: the knowledge-graph canvas needs the literal
            "Spectral" family name for ctx.font, which next/font's hashed family names cannot provide. */}
        {/* eslint-disable-next-line @next/next/no-page-custom-font */}
        <link
          href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-dark-950 text-dark-50 antialiased">
        <SetupGate>{children}</SetupGate>
      </body>
    </html>
  );
}
