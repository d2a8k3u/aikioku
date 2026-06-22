import { create } from 'zustand';

export type NeuralEvent = 'idle' | 'processing' | 'success' | 'error' | 'restart';

type NeuralState = {
  activityLevel: number;
  currentEvent: NeuralEvent;
  setActivityLevel: (level: number) => void;
  fireEvent: (event: NeuralEvent) => void;
};

export const useNeuralStore = create<NeuralState>((set) => ({
  activityLevel: 0,
  currentEvent: 'idle',
  setActivityLevel: (activityLevel) => set({ activityLevel }),
  fireEvent: (currentEvent) => set({ currentEvent }),
}));
