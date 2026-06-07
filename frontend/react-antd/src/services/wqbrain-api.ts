import request from './api'

type JsonObject = Record<string, unknown>

export interface WQBrainStatusResponse {
  success?: boolean
  connected?: boolean
  account?: string
  status?: {
    configured?: boolean
    [key: string]: unknown
  }
}

export interface WQBrainConfigResponse {
  success?: boolean
  default_account?: string
  primary_email?: string
  alt_email?: string
  has_primary_password?: boolean
  has_alt_password?: boolean
  message?: string
}

export interface WQBrainAlphaItem {
  alpha_id?: string
  id?: string
  expression?: string
  status?: string
  sharpe?: number | string
  fitness?: number | string
  returns?: number | string
  [key: string]: unknown
}

export interface WQBrainCandidateItem {
  factor_id: number
  name?: string
  category?: string
  origin_type?: string
  expression?: string
  score?: number | null
  grade?: string | null
  wq_rating?: string | null
  ls_sharpe?: number | null
  ls_return?: number | null
  wq_return?: number | null
  report_sharpe?: number | null
  alpha_id?: string | null
  submission_status?: string | null
  platform_status?: string | null
  sc_result?: string | null
  source_task_id?: string | null
  report_url?: string | null
  updated_at?: string | null
  [key: string]: unknown
}

export interface WQBrainAlphasResponse {
  success?: boolean
  account?: string
  configured?: boolean
  alphas?: WQBrainAlphaItem[]
  message?: string
  raw?: unknown
}

export interface WQBrainCandidatesResponse {
  success?: boolean
  total?: number
  candidates?: WQBrainCandidateItem[]
}

export const wqbrainApi = {
  getConfig() {
    return request.get<unknown, WQBrainConfigResponse>('/wqbrain/config')
  },

  saveConfig(data: {
    default_account: string
    primary_email?: string
    primary_password?: string | null
    alt_email?: string
    alt_password?: string | null
  }) {
    return request.post<unknown, WQBrainConfigResponse>('/wqbrain/config', data)
  },

  getWQStatus(account = 'primary') {
    return request.get<unknown, WQBrainStatusResponse>('/wqbrain/status', { params: { account } })
  },

  getWQUserInfo(account = 'primary') {
    return request.get('/wqbrain/user-info', { params: { account } })
  },

  getPlatformAlphas(account = 'primary', limit = 50) {
    return request.get<unknown, WQBrainAlphasResponse>('/wqbrain/platform-alphas', { params: { account, limit } })
  },

  getCandidates(limit = 100) {
    return request.get<unknown, WQBrainCandidatesResponse>('/wqbrain/candidates', { params: { limit } })
  },

  submitAlpha(data: JsonObject) {
    return request.post('/wqbrain/submit', data)
  },

  batchSubmit(data: JsonObject) {
    return request.post('/wqbrain/batch-submit', data)
  },

  checkAlphas(data: JsonObject) {
    return request.post('/wqbrain/check-alphas', data)
  },

  finalizeSubmissions(data: JsonObject) {
    return request.post('/wqbrain/finalize', data)
  },

  syncCandidates(data: JsonObject) {
    return request.post('/wqbrain/sync-candidates', data)
  }
}

export default wqbrainApi
