import request from './api'

export interface PaperStrategySummary {
  id: number
  name: string
  status: 'active' | 'paused' | 'stopped'
  current_value: number
  initial_capital: number
  total_return: number
  last_rebalance_date?: string | null
  next_rebalance_date?: string | null
  backtest_id: number
  created_at?: string | null
}

export interface PaperOrder {
  id: number
  date: string
  stock_code: string
  direction: 'buy' | 'sell'
  shares: number
  price: number
  amount: number
  commission: number
  slippage: number
}

export const paperApi = {
  createStrategy(data: { backtest_id: number; name?: string }) {
    return request.post('/paper/strategies', data)
  },
  listStrategies(includeStopped = false) {
    return request.get('/paper/strategies', { params: { include_stopped: includeStopped } })
  },
  getStrategy(id: number) {
    return request.get(`/paper/strategies/${id}`)
  },
  getOrders(id: number, limit = 50) {
    return request.get(`/paper/strategies/${id}/orders`, { params: { limit } })
  },
  updateStrategy(id: number, status: 'active' | 'paused' | 'stopped') {
    return request.patch(`/paper/strategies/${id}`, { status })
  },
  settleStrategy(id: number, force = false) {
    return request.post(`/paper/strategies/${id}/settle`, null, { params: { force } })
  },
  settleAll() {
    return request.post('/paper/settle')
  }
}
