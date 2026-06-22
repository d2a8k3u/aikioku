import { create } from 'zustand';

type AuthState = {
  unlocked: boolean;
  hasPassword: boolean | null;
  hydrated: boolean;
  lastActivity: number;
  unlock: () => void;
  lock: () => void;
  setHasPassword: (has: boolean) => void;
  setHydrated: () => void;
  touchActivity: () => void;
};

const SESSION_KEY = 'aikioku_dashboard_session';

export const useAuthStore = create<AuthState>((set) => ({
  unlocked: false,
  hasPassword: null,
  hydrated: false,
  lastActivity: Date.now(),

  unlock: () => {
    if (typeof window !== 'undefined') sessionStorage.setItem(SESSION_KEY, 'true');
    set({ unlocked: true, lastActivity: Date.now() });
  },

  lock: () => {
    if (typeof window !== 'undefined') sessionStorage.removeItem(SESSION_KEY);
    set({ unlocked: false });
  },

  setHasPassword: (has) => set({ hasPassword: has }),
  setHydrated: () => set({ hydrated: true }),
  touchActivity: () => set({ lastActivity: Date.now() }),
}));
