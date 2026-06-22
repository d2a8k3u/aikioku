import { create } from 'zustand';

export type SystemStatus = {
  uptime: number;
  version: string;
  goalsActiveCount: number;
};

export type SystemHealth = {
  memory: {
    heapUsed: number;
    heapTotal: number;
  };
};

type SystemState = {
  status: SystemStatus | null;
  health: SystemHealth | null;
  setStatus: (status: SystemStatus) => void;
  setHealth: (health: SystemHealth) => void;
  clearAll: () => void;
};

const DEFAULT_STATUS: SystemStatus = {
  uptime: 0,
  version: '—',
  goalsActiveCount: 0,
};

export const useSystemStore = create<SystemState>((set) => ({
  status: null,
  health: null,
  setStatus: (status) => set({ status }),
  setHealth: (health) => set({ health }),
  clearAll: () => set({ status: DEFAULT_STATUS, health: null }),
}));
