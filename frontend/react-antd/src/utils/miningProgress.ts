export type FitnessHistory = {
  best: number[];
  average: number[];
};

export type ExtendedFitnessHistory = FitnessHistory & {
  running_best?: number[];
};

type ProgressLikeStatus = {
  fitness_history?: FitnessHistory | null;
  current_generation?: number | null;
  best_fitness?: number | null;
  avg_fitness?: number | null;
};

export const EMPTY_FITNESS_HISTORY: FitnessHistory = {
  best: [],
  average: [],
};

export const normalizeFitnessHistory = (history?: FitnessHistory | null): FitnessHistory => ({
  best: Array.isArray(history?.best) ? history.best.map((value) => Number(value || 0)) : [],
  average: Array.isArray(history?.average) ? history.average.map((value) => Number(value || 0)) : [],
});

export const hasFitnessHistory = (history?: FitnessHistory | null): boolean => {
  const normalized = normalizeFitnessHistory(history);
  return normalized.best.length > 0 || normalized.average.length > 0;
};

export const buildProgressFallbackHistory = (
  status?: Pick<ProgressLikeStatus, "current_generation" | "best_fitness" | "avg_fitness"> | null,
): FitnessHistory => {
  const generationCount = Number(status?.current_generation || 0);
  if (generationCount <= 0) {
    return EMPTY_FITNESS_HISTORY;
  }
  return {
    best: Array(generationCount).fill(Number(status?.best_fitness || 0)),
    average: Array(generationCount).fill(Number(status?.avg_fitness || 0)),
  };
};

export const buildProgressHistory = (status?: ProgressLikeStatus | null): FitnessHistory => {
  const normalized = normalizeFitnessHistory(status?.fitness_history);
  if (hasFitnessHistory(normalized)) {
    return normalized;
  }
  return buildProgressFallbackHistory(status);
};

export const buildRunningBestHistory = (history?: FitnessHistory | null): ExtendedFitnessHistory => {
  const normalized = normalizeFitnessHistory(history);
  let bestSoFar = Number.NEGATIVE_INFINITY;
  const runningBest = normalized.best.map((value) => {
    bestSoFar = Math.max(bestSoFar, Number(value || 0));
    return bestSoFar;
  });
  return {
    ...normalized,
    running_best: runningBest,
  };
};
