import { create } from 'zustand';

interface PairState {
  selectedPair: string | null;
  setSelectedPair: (pair: string) => void;
}

export const usePairStore = create<PairState>((set) => ({
  selectedPair: null,
  setSelectedPair: (pair) => set({ selectedPair: pair }),
}));