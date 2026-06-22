import { create } from 'zustand';

export type ConnectionStatus = 'connected' | 'connecting' | 'reconnecting' | 'disconnected';

type ConnectionState = {
  status: ConnectionStatus;
  setStatus: (status: ConnectionStatus) => void;
};

export const useConnectionStore = create<ConnectionState>((set) => ({
  status: 'disconnected',
  setStatus: (status) => set({ status }),
}));
