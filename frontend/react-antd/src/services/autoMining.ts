import request from "./api";

export type AutoMiningDirection =
  | "score"
  | "ls_sharpe"
  | "ls_return"
  | "wq_rating"
  | "wq_fitness"
  | "wq_return"
  | "report_sharpe";

export interface AutoMiningFactorSelectionRequest {
  prompt: string;
  direction?: string;
  start_date: string;
  end_date: string;
  universe: string;
  benchmark: string;
  max_factor_count?: number;
  candidate_limit?: number;
  selection_mode?: "auto" | "manual_genetic";
}

export interface AutoMiningRequest {
  prompt: string;
  base_factors: string[];
  start_date: string;
  end_date: string;
  universe: string;
  benchmark: string;
  n_groups: number;
  holding_period: number;
  n_candidates: number;
  direction?: AutoMiningDirection | string;
  neutralize_industry?: boolean;
  neutralize_cap?: boolean;
}

export interface AutoMiningCampaignRequest {
  prompt: string;
  base_factors: string[];
  start_date: string;
  end_date: string;
  universe: string;
  benchmark: string;
  n_groups: number;
  holding_period: number;
  exploration_rounds: number;
  n_candidates_per_round: number;
  additional_factor_count_per_round: number;
  factor_update_mode?: "append" | "reselect";
  parent_selection_strategy?: "best_score_so_far" | "latest_round";
  direction?: AutoMiningDirection | string;
  neutralize_industry: boolean;
  neutralize_cap: boolean;
  retention_filter: {
    match_mode: "all" | "any";
    score_min?: number;
    wq_ratings?: string[];
    ls_sharpe_min?: number;
    ls_return_min?: number;
    wq_return_min?: number;
  };
}

export interface ContinueAutoMiningRequest {
  parent_task_id: string;
  prompt?: string;
  direction?: AutoMiningDirection | string;
  factor_update_mode?: "append" | "reselect";
  additional_base_factors?: string[];
  n_candidates?: number;
  n_groups?: number;
  holding_period?: number;
  neutralize_industry?: boolean;
  neutralize_cap?: boolean;
}

export interface ContinueAutoMiningFactorSelectionRequest {
  parent_task_id: string;
  prompt?: string;
  direction?: AutoMiningDirection | string;
  factor_update_mode?: "append" | "reselect";
  max_factor_count?: number;
  candidate_limit?: number;
}

const AUTO_MINING_TIMEOUT = 300000;
const AUTO_MINING_SELECTION_TIMEOUT = 120000;

export const autoMiningApi = {
  selectFactors(data: AutoMiningFactorSelectionRequest) {
    return request.post("/mining/auto/select-factors", data, { timeout: AUTO_MINING_SELECTION_TIMEOUT });
  },

  startTask(data: AutoMiningRequest) {
    return request.post("/mining/auto", data, { timeout: AUTO_MINING_TIMEOUT });
  },

  startCampaign(data: AutoMiningCampaignRequest) {
    return request.post("/mining/auto/campaign", data, { timeout: AUTO_MINING_TIMEOUT });
  },

  continueTask(data: ContinueAutoMiningRequest) {
    return request.post("/mining/auto/continue", data, { timeout: AUTO_MINING_TIMEOUT });
  },

  selectContinueFactors(data: ContinueAutoMiningFactorSelectionRequest) {
    return request.post("/mining/auto/continue/select-factors", data, {
      timeout: AUTO_MINING_SELECTION_TIMEOUT,
    });
  },

  getTaskStatus(taskId: string) {
    return request.get(`/mining/auto/status/${taskId}`, { timeout: AUTO_MINING_TIMEOUT });
  },

  getTaskResult(taskId: string) {
    return request.get(`/mining/auto/results/${taskId}`, { timeout: AUTO_MINING_TIMEOUT });
  },

  getCampaignStatus(taskId: string) {
    return request.get(`/mining/auto/campaign/status/${taskId}`, { timeout: AUTO_MINING_TIMEOUT });
  },

  getCampaignResult(taskId: string) {
    return request.get(`/mining/auto/campaign/results/${taskId}`, { timeout: AUTO_MINING_TIMEOUT });
  },
};
