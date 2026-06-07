import request from "./api";

export interface GeneticMiningRequest {
  stock_code: string;
  base_factors: string[];
  start_date: string;
  end_date: string;
  population_size: number;
  n_generations: number;
  cx_prob: number;
  mut_prob: number;
  elite_size: number;
  fitness_objective: string;
  ic_threshold: number;
}

export const autoMiningApi = {
  startTask(data: GeneticMiningRequest) {
    return request.post("/mining/genetic", data, { timeout: 300000 });
  },

  getTaskStatus(taskId: string) {
    return request.get(`/mining/status/${taskId}`, { timeout: 300000 });
  },

  getTaskResult(taskId: string) {
    return request.get(`/mining/results/${taskId}`, { timeout: 300000 });
  },
};
