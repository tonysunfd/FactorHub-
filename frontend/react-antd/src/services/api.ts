import axios from 'axios'
export { resolveApiUrl } from './url'

type QueryParams = Record<string, string | number | boolean | null | undefined>
type JsonObject = Record<string, unknown>

type RDAgentMiningRequest = {
  objective: string
  candidate_universe: string[]
  base_factors?: string[]
  start_date: string
  end_date: string
  universe: string
  benchmark: string
  max_iterations: number
  candidates_per_iteration: number
  n_groups?: number
  holding_period?: number
  direction?: string
  neutralize_industry?: boolean
  neutralize_cap?: boolean
  sota_library_id?: string
  continuation_of?: string
  previous_feedback_id?: string
  previous_expressions?: string[]
  acceptance_policy: {
    max_correlation_with_sota: number
    min_rank_ic: number
    min_annualized_return_delta: number
    max_drawdown_regression: number
    min_valid_coverage: number
  }
}

const backendPort = String(import.meta.env.VITE_BACKEND_PORT || '8001')

const resolveApiBaseUrl = () => {
  if (typeof window === 'undefined') {
    return `http://127.0.0.1:${backendPort}/api`
  }
  const host = window.location.hostname || '127.0.0.1'
  return `${window.location.protocol}//${host}:${backendPort}/api`
}

const request = axios.create({
  baseURL: resolveApiBaseUrl(),
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

request.interceptors.request.use(
  config => {
    return config
  },
  error => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

request.interceptors.response.use(
  response => {
    return response.data
  },
  error => {
    console.error('响应错误:', error)

    let message = '请求失败'

    if (error.response) {
      const { status, data } = error.response

      switch (status) {
        case 400:
          message = data.detail || data.message || '请求参数错误'
          break
        case 401:
          message = '未授权，请重新登录'
          break
        case 403:
          message = '拒绝访问'
          break
        case 404:
          message = '请求的资源不存在'
          break
        case 500:
          message = data.detail || data.message || '服务器错误'
          break
        default:
          message = data.detail || data.message || `请求失败 (${status})`
      }
    } else if (error.request) {
      message = '后端服务未连接，请先确认 FactorHub 后端已启动'
    } else {
      message = error.message || '请求失败'
    }

    return Promise.reject(new Error(message))
  }
)

export const api = {
  getFactorStats() {
    return request.get('/factors/stats')
  },

  getFactors(params?: QueryParams) {
    return request.get('/factors/', { params })
  },

  createFactor(data: JsonObject) {
    return request.post('/factors/', data)
  },

  updateFactor(id: number, data: JsonObject) {
    return request.put(`/factors/${id}`, data)
  },

  deleteFactor(id: number) {
    return request.delete(`/factors/${id}`)
  },

  getFactorDetail(id: number) {
    return request.get(`/factors/${id}`)
  },

  getFactorSnapshots(id: number) {
    return request.get(`/factors/${id}/snapshots`)
  },

  getFactorSnapshot(id: number, snapshotId: number) {
    return request.get(`/factors/${id}/snapshots/${snapshotId}`)
  },

  calculateIC(data: {
    factor_name: string
    stock_codes: string[]
    start_date: string
    end_date: string
  }) {
    return request.post('/analysis/ic', data)
  },

  calculateFactor(data: {
    factor_name: string
    stock_codes: string[]
    start_date: string
    end_date: string
  }) {
    return request.post('/analysis/calculate', data)
  },

  getStockData(code: string, startDate: string, endDate: string) {
    return request.get(`/data/stock/${code}`, {
      params: { start_date: startDate, end_date: endDate }
    })
  },

  analyzePortfolio(data: JsonObject) {
    return request.post('/portfolio/analyze', data)
  },

  runBacktest(data: JsonObject) {
    return request.post('/backtesting/run', data)
  },

  getBacktestResult(taskId: string) {
    return request.get(`/backtesting/results/${taskId}`)
  },

  validateFactor(data: JsonObject) {
    return request.post('/factors/validate/', data)
  },

  batchGenerateFactors(data: JsonObject) {
    return request.post('/factors/batch-generate/', data)
  },

  copyFactor(id: number) {
    return request.post(`/factors/${id}/copy`)
  },

  getPaperFactorStorage() {
    return request.get('/paper-factors/storage')
  },

  getPaperFactorRuntimeStatus() {
    return request.get('/paper-factors/runtime-status')
  },

  getPaperFactorFiles() {
    return request.get('/paper-factors/files')
  },

  uploadPaperFactorFile(file: File, sourceType: 'upload' | 'third_party_download' = 'upload') {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('source_type', sourceType)
    return request.post('/paper-factors/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000,
    })
  },

  downloadPaperFactorFile(data: {
    url: string
    filename?: string
  }) {
    return request.post('/paper-factors/download', data, { timeout: 120000 })
  },

  extractPaperFactors(data: {
    filenames: string[]
    category?: string
  }) {
    return request.post('/paper-factors/extract', data, { timeout: 300000 })
  },

  getPaperFactorLibrary() {
    return request.get('/paper-factors/library')
  },

  convertPaperFactors(data: {
    entry_ids: string[]
    category?: string
  }) {
    return request.post('/paper-factors/convert', data, { timeout: 300000 })
  },

  searchPaperSources(data: {
    source: string
    query: string
    limit?: number
  }) {
    return request.post('/paper-factors/search', data, { timeout: 120000 })
  },

  refreshPaperSources(data: {
    query: string
    sources?: string[]
    limit_per_source?: number
    auto_extract?: boolean
  }) {
    return request.post('/paper-factors/refresh', data, { timeout: 300000 })
  },

  importPaperSearchResults(data: {
    items: Record<string, any>[]
    category?: string
    auto_extract?: boolean
  }) {
    return request.post('/paper-factors/import-search-results', data, { timeout: 300000 })
  },

  analyzeExposure(data: {
    factor_name: string
    stock_codes: string[]
    start_date: string
    end_date: string
  }) {
    return request.post('/analysis/exposure', data)
  },

  analyzeEffectiveness(data: {
    factor_name: string
    stock_codes: string[]
    start_date: string
    end_date: string
  }) {
    return request.post('/analysis/effectiveness', data)
  },

  analyzeAttribution(data: {
    factor_name: string
    stock_codes: string[]
    start_date: string
    end_date: string
  }) {
    return request.post('/analysis/attribution', data)
  },

  analyzeMonitoring(data: {
    factor_name: string
    stock_codes: string[]
    start_date: string
    end_date: string
  }) {
    return request.post('/analysis/monitoring', data)
  },

  recomputeTaskDetails(data: {
    factor_name: string
    stock_codes: string[]
    start_date: string
    end_date: string
  }) {
    return request.post('/analysis/recompute-task-details', data, { timeout: 120000 })
  },

  startGeneticMining(data: {
    stock_code: string
    base_factors: string[]
    start_date: string
    end_date: string
    population_size: number
    n_generations: number
    cx_prob: number
    mut_prob: number
    elite_size: number
    fitness_objective: string
    ic_threshold: number
  }) {
    return request.post('/mining/genetic', data, { timeout: 300000 })
  },

  getMiningStatus(taskId: string) {
    return request.get(`/mining/status/${taskId}`, { timeout: 300000 })
  },

  getMiningResults(taskId: string) {
    return request.get(`/mining/results/${taskId}`, { timeout: 300000 })
  },

  selectAutoMiningFactors(data: {
    prompt: string
    direction?: string
    start_date: string
    end_date: string
    universe: string
    benchmark: string
    max_factor_count?: number
    candidate_limit?: number
    selection_mode?: 'auto' | 'manual_genetic'
  }) {
    return request.post('/mining/auto/select-factors', data, { timeout: 120000 })
  },

  selectRDAgentBootstrap(data: {
    objective: string
    direction?: string
    start_date: string
    end_date: string
    universe: string
    benchmark: string
    max_factor_count?: number
    max_candidate_field_count?: number
    candidate_limit?: number
  }) {
    return request.post('/mining/rdagent/select-bootstrap', data, { timeout: 120000 })
  },

  startAutoMining(data: {
    prompt: string
    base_factors: string[]
    start_date: string
    end_date: string
    universe: string
    benchmark: string
    n_groups: number
    holding_period: number
    n_candidates: number
    direction?: string
    neutralize_industry?: boolean
    neutralize_cap?: boolean
  }) {
    return request.post('/mining/auto', data, { timeout: 300000 })
  },

  startAutoMiningCampaign(data: {
    prompt: string
    base_factors: string[]
    start_date: string
    end_date: string
    universe: string
    benchmark: string
    n_groups: number
    holding_period: number
    exploration_rounds: number
    n_candidates_per_round: number
    additional_factor_count_per_round: number
    factor_update_mode?: 'append' | 'reselect'
    parent_selection_strategy?: 'best_score_so_far' | 'latest_round'
    direction?: string
    neutralize_industry: boolean
    neutralize_cap: boolean
    retention_filter: {
      match_mode: 'all' | 'any'
      score_min?: number
      wq_ratings?: string[]
      ls_sharpe_min?: number
      ls_return_min?: number
      wq_return_min?: number
    }
  }) {
    return request.post('/mining/auto/campaign', data, { timeout: 300000 })
  },

  continueAutoMining(data: {
    parent_task_id: string
    prompt?: string
    direction?: string
    factor_update_mode?: 'append' | 'reselect'
    additional_base_factors?: string[]
    n_candidates?: number
    n_groups?: number
    holding_period?: number
    neutralize_industry?: boolean
    neutralize_cap?: boolean
  }) {
    return request.post('/mining/auto/continue', data, { timeout: 300000 })
  },

  selectContinueAutoMiningFactors(data: {
    parent_task_id: string
    prompt?: string
    direction?: string
    factor_update_mode?: 'append' | 'reselect'
    max_factor_count?: number
    candidate_limit?: number
  }) {
    return request.post('/mining/auto/continue/select-factors', data, { timeout: 120000 })
  },

  getAutoMiningStatus(taskId: string) {
    return request.get(`/mining/auto/status/${taskId}`, { timeout: 300000 })
  },

  getAutoMiningResults(taskId: string) {
    return request.get(`/mining/auto/results/${taskId}`, { timeout: 300000 })
  },

  getAutoMiningCampaignStatus(taskId: string) {
    return request.get(`/mining/auto/campaign/status/${taskId}`, { timeout: 300000 })
  },

  getAutoMiningCampaignResults(taskId: string) {
    return request.get(`/mining/auto/campaign/results/${taskId}`, { timeout: 300000 })
  },

  listMiningTasks(kind?: string, limit = 20) {
    const params = new URLSearchParams()
    if (kind) params.set('kind', kind)
    params.set('limit', String(limit))
    return request.get(`/mining/tasks?${params.toString()}`, { timeout: 120000 })
  },

  startRDAgentMining(data: RDAgentMiningRequest) {
    return request.post('/mining/rdagent', data, { timeout: 300000 })
  },

  getRDAgentMiningStatus(taskId: string) {
    return request.get(`/mining/rdagent/status/${taskId}`, { timeout: 300000 })
  },

  getRDAgentMiningResults(taskId: string) {
    return request.get(`/mining/rdagent/results/${taskId}`, { timeout: 300000 })
  },

  cancelRDAgentMining(taskId: string) {
    return request.post(`/mining/rdagent/${taskId}/cancel`, {}, { timeout: 120000 })
  },

  getLLMConfig() {
    return request.get('/llm/config')
  },

  saveLLMConfig(data: {
    api_key?: string | null
    base_url: string
    model: string
  }) {
    return request.post('/llm/config', data)
  },

  restartLLMService() {
    return request.post('/llm/restart')
  },

  migrateTaskSnapshots(dryRun = false) {
    return request.post(`/factors/migrate-task-snapshots?dry_run=${dryRun}`)
  }
}

export default request
