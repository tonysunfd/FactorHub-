import React from 'react'
import { Card, Col, Empty, Row, Space, Tag } from 'antd'
import { ExperimentOutlined } from '@ant-design/icons'
import { resolveApiUrl } from '@/services/url'

export interface FactorTaskDetails {
  expression?: string
  report_url?: string
  metrics?: Record<string, any>
  report_metrics?: Record<string, any>
  backtest_summary?: Record<string, any>
  wq_brain?: Record<string, any>
  anti_overfit?: Record<string, any>
  scoring?: {
    score?: number
    grade?: string
    component_scores?: Record<string, any>
    wq_fitness?: number
    wq_pass_count?: number
    wq_total_tests?: number
    capped?: boolean
    cap_reason?: string
    [key: string]: any
  }
  interpretation?: Record<string, any>
  params?: Record<string, any>
  llm?: Record<string, any>
}

interface FactorTaskDetailsPanelProps {
  taskId?: string
  source?: string
  details?: FactorTaskDetails | null
}

const fmtNum = (v: any, digits = 2) => {
  const n = Number(v)
  if (!Number.isFinite(n)) return undefined
  return n.toFixed(digits)
}

const fmtPct = (v: any, digits = 1) => {
  const n = Number(v)
  if (!Number.isFinite(n)) return undefined
  return `${(n * 100).toFixed(digits)}%`
}

const fmtText = (v: any) => {
  if (v === null || v === undefined || v === '') return '—'
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : '—'
  if (typeof v === 'boolean') return v ? '是' : '否'
  return String(v)
}

const MetricGrid: React.FC<{ items: Array<{ label: string; value?: string }> }> = ({ items }) => {
  const visible = items.filter(item => item.value !== undefined)
  if (!visible.length) return null
  return (
    <Row gutter={[10, 10]}>
      {visible.map((item) => (
        <Col xs={12} md={6} key={item.label}>
          <Card variant="borderless" bodyStyle={{ padding: 12, background: 'rgba(248, 250, 252, 0.95)', borderRadius: 12 }}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6, wordBreak: 'break-word' }}>{item.label}</div>
            <div style={{ fontSize: 13, color: '#0f172a', lineHeight: 1.6, wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>{item.value}</div>
          </Card>
        </Col>
      ))}
    </Row>
  )
}

const SectionTitle: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div style={{ fontSize: 14, fontWeight: 600, color: '#0f172a', marginBottom: 10 }}>{children}</div>
)

const FactorTaskDetailsPanel: React.FC<FactorTaskDetailsPanelProps> = ({ taskId, source, details }) => {
  if (!details) {
    return (
      <Card variant="borderless">
        <div style={{ textAlign: 'center', padding: '60px 20px', color: '#64748b' }}>
          <ExperimentOutlined style={{ fontSize: 48, marginBottom: 16 }} />
          <p>No task details</p>
        </div>
      </Card>
    )
  }

  const prompt = details.params?.prompt || details.llm?.prompt
  const expression = details.expression || details.params?.expression || details.llm?.generated_expression
  const tag = details.params?.tag
  const rating = details.interpretation?.rating || details.wq_brain?.wq_rating || details.scoring?.grade
  const backtest = details.backtest_summary || {}
  const wq = details.wq_brain || {}
  const reportMetrics = details.report_metrics || details.metrics || {}
  const reportLink = details.report_url ? resolveApiUrl(details.report_url) : undefined

  const keyMetrics = [
    { label: 'L/S Sharpe', value: fmtNum(backtest.long_short_sharpe, 2) },
    { label: 'L/S Annual', value: fmtPct(backtest.long_short_annual, 1) },
    { label: 'Rank IC', value: fmtNum(backtest.rank_ic_mean, 4) },
    { label: 'IC IR', value: fmtNum(backtest.ic_ir, 2) },
    { label: 'Turnover', value: fmtNum(backtest.turnover, 3) },
    { label: 'Fitness', value: fmtNum(backtest.wq_fitness ?? details.scoring?.wq_fitness, 3) },
    { label: 'Monotonicity', value: fmtNum(backtest.monotonicity_score, 2) },
    { label: 'Spread', value: fmtNum(backtest.spread, 2) },
  ]

  const wqMetrics = [
    { label: 'WQ Sharpe', value: fmtNum(wq.wq_sharpe, 2) },
    { label: 'WQ Fitness', value: fmtNum(wq.wq_fitness, 3) },
    { label: 'WQ Returns', value: fmtPct(wq.wq_returns, 1) },
    { label: 'WQ Rating', value: wq.wq_rating ? String(wq.wq_rating) : undefined },
  ]

  const reportSummary = [
    { label: 'Score', value: fmtNum(details.scoring?.score, 2) },
    { label: 'Grade', value: details.scoring?.grade ? String(details.scoring.grade) : undefined },
    { label: 'Sharpe', value: fmtNum(reportMetrics.sharpe ?? reportMetrics.report_sharpe, 2) },
    { label: 'Max Drawdown', value: fmtPct(reportMetrics.max_drawdown ?? reportMetrics.report_max_drawdown, 1) },
  ]

  return (
    <Card variant="borderless">
      <Space direction="vertical" size={20} style={{ width: '100%' }}>
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 12 }}>
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#0f172a', marginBottom: 4 }}>Task Detail</div>
              {taskId ? <div style={{ fontSize: 12, color: '#64748b' }}>{taskId}</div> : null}
            </div>
            <Space wrap>
              {source ? <Tag>{source}</Tag> : null}
              {rating ? <Tag color="blue">{String(rating)}</Tag> : null}
            </Space>
          </div>

          {prompt ? (
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>Prompt</div>
              <div style={{ fontSize: 13, color: '#0f172a', lineHeight: 1.6, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{fmtText(prompt)}</div>
            </div>
          ) : null}

          {expression ? (
            <div style={{ marginBottom: tag ? 14 : 0 }}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>Expression</div>
              <pre style={{ background: 'rgba(248, 250, 252, 0.95)', border: '1px solid rgba(59, 130, 246, 0.12)', borderRadius: 12, padding: 16, fontFamily: "'SF Mono', 'Monaco', 'Consolas', monospace", fontSize: 12, lineHeight: 1.7, color: '#334155', overflowX: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0 }}>{expression}</pre>
            </div>
          ) : null}

          {tag ? (
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>Tag</div>
              <Tag color="purple">{fmtText(tag)}</Tag>
            </div>
          ) : null}
        </div>

        {Object.keys(backtest).length ? (
          <div>
            <SectionTitle>Key Metrics</SectionTitle>
            <MetricGrid items={keyMetrics} />
          </div>
        ) : null}

        {Object.keys(wq).length ? (
          <div>
            <SectionTitle>WQ Brain</SectionTitle>
            <MetricGrid items={wqMetrics} />
          </div>
        ) : null}

        {Object.keys(reportMetrics).length || details.scoring ? (
          <div>
            <SectionTitle>Report Summary</SectionTitle>
            <MetricGrid items={reportSummary} />
          </div>
        ) : null}

        {details.interpretation && Object.keys(details.interpretation).length ? (
          <div>
            <SectionTitle>AI Analysis</SectionTitle>
            <Card variant="borderless" bodyStyle={{ padding: 14, background: 'rgba(248, 250, 252, 0.95)', borderRadius: 12 }}>
              {details.interpretation?.conclusion ? <p style={{ fontSize: 13, color: '#0f172a', lineHeight: 1.6 }}><strong>Conclusion: </strong>{fmtText(details.interpretation.conclusion)}</p> : null}
              {details.interpretation?.logic ? <p style={{ fontSize: 13, color: '#0f172a', lineHeight: 1.6 }}><strong>Logic: </strong>{fmtText(details.interpretation.logic)}</p> : null}
              {details.interpretation?.guidance ? <p style={{ fontSize: 13, color: '#0f172a', lineHeight: 1.6, marginBottom: 0 }}><strong>Guidance: </strong>{fmtText(details.interpretation.guidance)}</p> : null}
            </Card>
          </div>
        ) : null}

        {reportLink ? (
          <div>
            <SectionTitle>Backtest Report</SectionTitle>
            <Card variant="borderless" bodyStyle={{ padding: 0, overflow: 'hidden' }}>
              <iframe
                src={reportLink}
                title="Backtest Report"
                style={{ width: '100%', height: 640, border: 0, display: 'block', background: '#fff' }}
              />
            </Card>
          </div>
        ) : null}

        {!prompt && !expression && !Object.keys(backtest).length && !Object.keys(wq).length && !Object.keys(reportMetrics).length && !reportLink ? (
          <Empty description="暂无可展示的任务详情" />
        ) : null}
      </Space>
    </Card>
  )
}

export default FactorTaskDetailsPanel
