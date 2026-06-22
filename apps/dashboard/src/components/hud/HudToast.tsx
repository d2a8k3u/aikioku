'use client';

import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
} from 'react';
import { createPortal } from 'react-dom';

import { cn } from '@/lib/cn';

// ── Types ──────────────────────────────────────────────

type ToastVariant = 'success' | 'error' | 'warning' | 'info';

type Toast = {
  readonly id: string;
  readonly message: string;
  readonly variant: ToastVariant;
  readonly durationMs: number;
  readonly createdAt: number;
};

type ToastContextValue = {
  readonly addToast: (message: string, variant?: ToastVariant, durationMs?: number) => void;
};

// ── Context ────────────────────────────────────────────

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return ctx;
}

// ── Provider ───────────────────────────────────────────

type ToastProviderProps = {
  readonly children: ReactNode;
  readonly maxToasts?: number;
};

export function ToastProvider({ children, maxToasts = 5 }: ToastProviderProps) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    const timer = timersRef.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timersRef.current.delete(id);
    }
  }, []);

  const addToast = useCallback(
    (message: string, variant: ToastVariant = 'info', durationMs = 4000) => {
      const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const toast: Toast = { id, message, variant, durationMs, createdAt: Date.now() };

      setToasts((prev) => {
        const next = [...prev, toast];
        // Enforce max
        while (next.length > maxToasts) {
          const oldest = next.shift();
          if (oldest) {
            const timer = timersRef.current.get(oldest.id);
            if (timer) {
              clearTimeout(timer);
              timersRef.current.delete(oldest.id);
            }
          }
        }
        return next;
      });

      const timer = setTimeout(() => removeToast(id), durationMs);
      timersRef.current.set(id, timer);
    },
    [maxToasts, removeToast],
  );

  // Cleanup all timers on unmount
  useEffect(() => {
    return () => {
      for (const timer of timersRef.current.values()) {
        clearTimeout(timer);
      }
      timersRef.current.clear();
    };
  }, []);

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      <ToastContainer toasts={toasts} onDismiss={removeToast} />
    </ToastContext.Provider>
  );
}

// ── Container (portal) ─────────────────────────────────

const variantStyles: Record<ToastVariant, string> = {
  success: 'border-aiki-success/30 bg-aiki-success/10 text-aiki-success',
  error: 'border-aiki-danger/30 bg-aiki-danger/10 text-aiki-danger',
  warning: 'border-aiki-warning/30 bg-aiki-warning/10 text-aiki-warning',
  info: 'border-aiki-accent/30 bg-aiki-accent/10 text-aiki-accent',
};

const variantIcons: Record<ToastVariant, ReactNode> = {
  success: (
    <svg
      aria-hidden="true"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  ),
  error: (
    <svg
      aria-hidden="true"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="15" y1="9" x2="9" y2="15" />
      <line x1="9" y1="9" x2="15" y2="15" />
    </svg>
  ),
  warning: (
    <svg
      aria-hidden="true"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  ),
  info: (
    <svg
      aria-hidden="true"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  ),
};

function ToastContainer({
  toasts,
  onDismiss,
}: {
  readonly toasts: readonly Toast[];
  readonly onDismiss: (id: string) => void;
}) {
  if (toasts.length === 0) return null;

  return createPortal(
    <div
      aria-live="polite"
      aria-label="Notifications"
      className="fixed bottom-4 right-4 z-[70] flex flex-col-reverse gap-2 pointer-events-none"
    >
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </div>,
    document.body,
  );
}

// ── Individual toast ───────────────────────────────────

function ToastItem({
  toast,
  onDismiss,
}: {
  readonly toast: Toast;
  readonly onDismiss: (id: string) => void;
}) {
  const [visible, setVisible] = useState(false);
  const [exiting, setExiting] = useState(false);
  const dismissTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const labelId = useId();

  // Slide in on mount
  useEffect(() => {
    const raf = requestAnimationFrame(() => setVisible(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  // Auto-dismiss
  useEffect(() => {
    dismissTimerRef.current = setTimeout(() => {
      setExiting(true);
      setTimeout(() => onDismiss(toast.id), 300); // wait for exit animation
    }, toast.durationMs);
    return () => clearTimeout(dismissTimerRef.current);
  }, [toast.durationMs, toast.id, onDismiss]);

  const handleDismiss = () => {
    clearTimeout(dismissTimerRef.current);
    setExiting(true);
    setTimeout(() => onDismiss(toast.id), 300);
  };

  return (
    <div
      role="status"
      aria-labelledby={labelId}
      className={cn(
        'pointer-events-auto',
        'flex items-center gap-2 px-3 py-2',
        'backdrop-blur-sm rounded-lg border',
        'text-xs font-mono',
        'transition-all duration-300 ease-out',
        variantStyles[toast.variant],
        visible && !exiting ? 'translate-x-0 opacity-100' : 'translate-x-4 opacity-0',
      )}
    >
      {variantIcons[toast.variant]}
      <span id={labelId} className="flex-1">
        {toast.message}
      </span>
      <button
        type="button"
        onClick={handleDismiss}
        aria-label="Dismiss notification"
        className="flex-shrink-0 opacity-50 hover:opacity-100 transition-opacity"
      >
        <svg
          aria-hidden="true"
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>
    </div>
  );
}

// Re-export HudToast for convenience (the provider)
export { ToastProvider as HudToast };
