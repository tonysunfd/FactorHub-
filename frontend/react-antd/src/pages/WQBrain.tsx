import { useEffect, useState } from 'react'
import { Alert, Button, Card, Col, Empty, Form, Input, Row, Space, Table, Tag, Typography, message } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import wqbrainApi, {
  type WQBrainAlphaItem,
  type WQBrainCandidateItem,
  type WQBrainConfigResponse,
  type WQBrainStatusResponse,
} from '@/services/wqbrain-api'

const { Title, Text, Link } = Typography

const platformColumns: ColumnsType<WQBrainAlphaItem> = [
  { title: 'Alpha ID', dataIndex: 'alpha_id', key: 'alpha_id', width: 180 },
  { title: 'Expression', dataIndex: 'expression', key: 'expression', ellipsis: true },
  {
    title: 'Status',
    dataIndex: 'status',
    key: 'status',
    width: 120,
    render: (value: string) => <Tag color={value === 'ACTIVE' ? 'green' : 'blue'}>{value || '--'}</Tag>,
  },
  { title: 'Sharpe', dataIndex: 'sharpe', key: 'sharpe', width: 120 },
  { title: 'Fitness', dataIndex: 'fitness', key: 'fitness', width: 120 },
  { title: 'Returns', dataIndex: 'returns', key: 'returns', width: 120 },
]

const statusColorMap: Record<string, string> = {
  ACTIVE: 'green',
  SC_PENDING: 'gold',
  SC_FAIL: 'red',
  SUBMITTED: 'blue',
  SIMULATED: 'cyan',
  UNSUBMITTED: 'default',
  ERROR: 'red',
}

const formatMetric = (value?: number | string | null) => {
  if (value === null || value === undefined || value === '') {
    return '--'
  }
  const numeric = Number(value)
  if (Number.isFinite(numeric)) {
    return numeric.toFixed(3)
  }
  return String(value)
}

const WQBrain: React.FC = () => {
  const [form] = Form.useForm()
  const [config, setConfig] = useState<WQBrainConfigResponse | null>(null)
  const [status, setStatus] = useState<WQBrainStatusResponse | null>(null)
  const [alphas, setAlphas] = useState<WQBrainAlphaItem[]>([])
  const [candidates, setCandidates] = useState<WQBrainCandidateItem[]>([])
  const [configLoading, setConfigLoading] = useState(false)
  const [configSaving, setConfigSaving] = useState(false)
  const [statusLoading, setStatusLoading] = useState(false)
  const [alphasLoading, setAlphasLoading] = useState(false)
  const [candidatesLoading, setCandidatesLoading] = useState(false)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [alphasError, setAlphasError] = useState<string | null>(null)
  const [candidatesError, setCandidatesError] = useState<string | null>(null)
  const [alphasMessage, setAlphasMessage] = useState<string | null>(null)
  const [submittingId, setSubmittingId] = useState<number | null>(null)
  const [syncingId, setSyncingId] = useState<number | null>(null)
  const [syncingAll, setSyncingAll] = useState(false)

  const currentAccount = form.getFieldValue('default_account') || 'primary'

  const loadConfig = async () => {
    setConfigLoading(true)
    try {
      const res = await wqbrainApi.getConfig()
      setConfig(res)
      form.setFieldsValue({
        default_account: res.default_account || 'primary',
        primary_email: res.primary_email || '',
        primary_password: '',
        alt_email: res.alt_email || '',
        alt_password: '',
      })
    } catch (error: unknown) {
      message.error(error instanceof Error ? error.message : '加载配置失败')
    } finally {
      setConfigLoading(false)
    }
  }

  const loadStatus = async (account?: string) => {
    setStatusLoading(true)
    setStatusError(null)
    try {
      const statusRes = await wqbrainApi.getWQStatus(account || currentAccount)
      setStatus(statusRes)
    } catch (error: unknown) {
      setStatusError(error instanceof Error ? error.message : '加载状态失败')
    } finally {
      setStatusLoading(false)
    }
  }

  const loadAlphas = async (account?: string) => {
    setAlphasLoading(true)
    setAlphasError(null)
    try {
      const res = await wqbrainApi.getPlatformAlphas(account || currentAccount)
      setAlphas(res.alphas || [])
      setAlphasMessage(res.message || null)
    } catch (error: unknown) {
      setAlphasError(error instanceof Error ? error.message : '加载平台 Alpha 失败')
    } finally {
      setAlphasLoading(false)
    }
  }

  const loadCandidates = async () => {
    setCandidatesLoading(true)
    setCandidatesError(null)
    try {
      const res = await wqbrainApi.getCandidates(200)
      setCandidates(res.candidates || [])
    } catch (error: unknown) {
      setCandidatesError(error instanceof Error ? error.message : '加载候选因子失败')
    } finally {
      setCandidatesLoading(false)
    }
  }

  const loadAll = async () => {
    await loadConfig()
    await loadStatus()
    await loadAlphas()
    await loadCandidates()
  }

  useEffect(() => {
    void loadAll()
  }, [])

  const handleSave = async () => {
    const values = await form.validateFields()
    setConfigSaving(true)
    try {
      const payload = {
        default_account: values.default_account,
        primary_email: values.primary_email,
        primary_password: values.primary_password ? values.primary_password : null,
        alt_email: values.alt_email,
        alt_password: values.alt_password ? values.alt_password : null,
      }
      const res = await wqbrainApi.saveConfig(payload)
      setConfig(res)
      form.setFieldsValue({ primary_password: '', alt_password: '' })
      message.success(res.message || '配置已保存')
      await loadStatus(values.default_account)
      await loadAlphas(values.default_account)
    } catch (error: unknown) {
      message.error(error instanceof Error ? error.message : '保存配置失败')
    } finally {
      setConfigSaving(false)
    }
  }

  const refreshAfterMutation = async () => {
    await loadCandidates()
    await loadAlphas()
    await loadStatus()
  }

  const handleSubmitCandidate = async (candidate: WQBrainCandidateItem) => {
    setSubmittingId(candidate.factor_id)
    try {
      const res = await wqbrainApi.submitAlpha({
        factor_id: candidate.factor_id,
        account: currentAccount,
        auto_submit: true,
      })
      const result = res as { success?: boolean; message?: string; alpha_id?: string; submitted?: boolean }
      if (!result.success) {
        throw new Error(result.message || '提交失败')
      }
      message.success(result.alpha_id ? `已提交到 WQ Brain：${result.alpha_id}` : '提交成功')
      await refreshAfterMutation()
    } catch (error: unknown) {
      message.error(error instanceof Error ? error.message : '提交失败')
    } finally {
      setSubmittingId(null)
    }
  }

  const handleSyncCandidates = async (factorIds: number[], factorIdForLoading?: number) => {
    if (factorIdForLoading) {
      setSyncingId(factorIdForLoading)
    } else {
      setSyncingAll(true)
    }
    try {
      const res = await wqbrainApi.syncCandidates({ factor_ids: factorIds, account: currentAccount })
      const summary = (res as { summary?: { active?: number } }).summary
      message.success(summary ? `同步完成，当前 ACTIVE ${summary.active || 0} 个` : '同步完成')
      await refreshAfterMutation()
    } catch (error: unknown) {
      message.error(error instanceof Error ? error.message : '同步失败')
    } finally {
      setSyncingId(null)
      setSyncingAll(false)
    }
  }

  const syncableFactorIds = candidates.filter((item) => item.alpha_id).map((item) => item.factor_id)

  const candidateColumns: ColumnsType<WQBrainCandidateItem> = [
    { title: '因子', dataIndex: 'name', key: 'name', width: 160 },
    { title: '来源', dataIndex: 'origin_type', key: 'origin_type', width: 120 },
    { title: 'Expression', dataIndex: 'expression', key: 'expression', ellipsis: true },
    { title: 'Score', dataIndex: 'score', key: 'score', width: 110, render: (value) => formatMetric(value) },
    { title: 'WQ Rating', dataIndex: 'wq_rating', key: 'wq_rating', width: 120, render: (value) => value || '--' },
    { title: 'L/S Sharpe', dataIndex: 'ls_sharpe', key: 'ls_sharpe', width: 120, render: (value) => formatMetric(value) },
    { title: 'L/S Return', dataIndex: 'ls_return', key: 'ls_return', width: 120, render: (value) => formatMetric(value) },
    { title: 'WQ Return', dataIndex: 'wq_return', key: 'wq_return', width: 120, render: (value) => formatMetric(value) },
    {
      title: '提交状态',
      dataIndex: 'submission_status',
      key: 'submission_status',
      width: 140,
      render: (value: string | null | undefined) => {
        const label = value || 'UNSUBMITTED'
        return <Tag color={statusColorMap[label] || 'default'}>{label}</Tag>
      },
    },
    {
      title: 'Alpha ID',
      dataIndex: 'alpha_id',
      key: 'alpha_id',
      width: 180,
      render: (value) => value || '--',
    },
    {
      title: '报告',
      dataIndex: 'report_url',
      key: 'report_url',
      width: 100,
      render: (value: string | null | undefined) => (
        value ? <Link href={value} target="_blank">查看</Link> : '--'
      ),
    },
    {
      title: '操作',
      key: 'actions',
      fixed: 'right',
      width: 220,
      render: (_, record) => (
        <Space>
          <Button
            type="primary"
            size="small"
            loading={submittingId === record.factor_id}
            onClick={() => void handleSubmitCandidate(record)}
          >
            提交
          </Button>
          <Button
            size="small"
            disabled={!record.alpha_id}
            loading={syncingId === record.factor_id}
            onClick={() => void handleSyncCandidates([record.factor_id], record.factor_id)}
          >
            同步状态
          </Button>
        </Space>
      ),
    },
  ]

  return (
    <div style={{ padding: '24px 0' }}>
      <Title level={2} style={{ marginBottom: 24 }}>WQ BRAIN</Title>

      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="连接配置" loading={configLoading}>
            <Form form={form} layout="vertical">
              <Row gutter={[16, 16]}>
                <Col xs={24} md={12}>
                  <Form.Item label="默认账户" name="default_account" rules={[{ required: true, message: '请输入账户名' }]}>
                    <Input placeholder="primary" />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item label="主账号邮箱" name="primary_email">
                    <Input placeholder="WQ_BRAIN_EMAIL" />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item label="主账号密码" name="primary_password">
                    <Input.Password placeholder={config?.has_primary_password ? '留空则保持当前密码不变' : '填写主账号密码'} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item label="备用账号邮箱" name="alt_email">
                    <Input placeholder="WQ_BRAIN_ALT_EMAIL" />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item label="备用账号密码" name="alt_password">
                    <Input.Password placeholder={config?.has_alt_password ? '留空则保持当前密码不变' : '可选：填写备用账号密码'} />
                  </Form.Item>
                </Col>
                <Col span={24}>
                  <Space direction="vertical" size={4}>
                    <Text type="secondary">当前模式：直接连接 WQ BRAIN</Text>
                    <Text type="secondary">自动挖掘结果和因子库中的候选因子会先进入候选池，再由这里统一提交和同步平台状态。</Text>
                  </Space>
                </Col>
                <Col span={24}>
                  <Space>
                    <Button type="primary" onClick={() => void handleSave()} loading={configSaving}>保存配置</Button>
                    <Button onClick={() => void loadStatus()}>测试连接</Button>
                    <Button onClick={() => void loadAll()}>刷新</Button>
                  </Space>
                </Col>
              </Row>
            </Form>
          </Card>
        </Col>

        <Col span={24}>
          <Card title="WQ BRAIN 状态" loading={statusLoading}>
            {statusError ? (
              <Alert type="error" message={statusError} showIcon />
            ) : (
              <Row gutter={[16, 16]}>
                <Col xs={24} md={8}>
                  <div><strong>账户：</strong>{status?.account || '--'}</div>
                </Col>
                <Col xs={24} md={8}>
                  <div>
                    <strong>连接状态：</strong>
                    <Tag color={status?.connected ? 'green' : 'red'} style={{ marginLeft: 8 }}>
                      {status?.connected ? '已连接' : '未连接'}
                    </Tag>
                  </div>
                </Col>
                <Col xs={24} md={8}>
                  <div><strong>配置状态：</strong>{status?.status?.configured === false ? '未配置' : '已配置'}</div>
                </Col>
              </Row>
            )}
          </Card>
        </Col>

        <Col span={24}>
          <Card
            title="WQ Brain 候选因子"
            extra={(
              <Space>
                <Button onClick={() => void loadCandidates()} loading={candidatesLoading}>刷新候选池</Button>
                <Button
                  onClick={() => void handleSyncCandidates(syncableFactorIds)}
                  disabled={syncableFactorIds.length === 0}
                  loading={syncingAll}
                >
                  批量同步状态
                </Button>
              </Space>
            )}
            loading={candidatesLoading}
          >
            {candidatesError ? (
              <Alert type="error" message={candidatesError} showIcon />
            ) : candidates.length > 0 ? (
              <Table
                rowKey={(record) => String(record.factor_id)}
                columns={candidateColumns}
                dataSource={candidates}
                pagination={{ pageSize: 10 }}
                scroll={{ x: 1800 }}
              />
            ) : (
              <Empty description="暂无可提交的候选因子" />
            )}
          </Card>
        </Col>

        <Col span={24}>
          <Card title="平台 Alpha" loading={alphasLoading}>
            {alphasError ? (
              <Alert type="error" message={alphasError} showIcon />
            ) : alphas.length > 0 ? (
              <Table
                rowKey={(record) => record.alpha_id || record.id || JSON.stringify(record)}
                columns={platformColumns}
                dataSource={alphas}
                pagination={{ pageSize: 10 }}
                scroll={{ x: 1000 }}
              />
            ) : (
              <>
                {alphasMessage ? <Alert type="warning" message={alphasMessage} showIcon style={{ marginBottom: 16 }} /> : null}
                <Empty description="暂无平台 Alpha 数据" />
              </>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default WQBrain
