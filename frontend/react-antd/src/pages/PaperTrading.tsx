import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Button, Card, Empty, List, message, Space, Spin, Statistic, Switch, Table, Tag } from 'antd'
import { LineChartOutlined, PauseCircleOutlined, PlayCircleOutlined, ReloadOutlined, StopOutlined } from '@ant-design/icons'

import { paperApi, type PaperOrder, type PaperStrategySummary } from '../services/paper-api'

const pct = (n: number) => `${(n * 100).toFixed(2)}%`
const fmt = (n: number) => n.toLocaleString('zh-CN', { maximumFractionDigits: 0 })

const statusColor: Record<string, string> = {
  active: 'green',
  paused: 'orange',
  stopped: 'default'
}

const statusLabel: Record<string, string> = {
  active: '运行中',
  paused: '已暂停',
  stopped: '已停止'
}

const PaperTrading: React.FC = () => {
  const [loading, setLoading] = useState(true)
  const [strategies, setStrategies] = useState<PaperStrategySummary[]>([])
  const [selected, setSelected] = useState<any | null>(null)
  const [orders, setOrders] = useState<PaperOrder[]>([])
  const [detailLoading, setDetailLoading] = useState(false)
  const [settling, setSettling] = useState(false)
  const [showStopped, setShowStopped] = useState(false)

  const load = useCallback(async () => {
    try {
      setLoading(true)
      const resp = await paperApi.listStrategies(showStopped)
      const nextStrategies = resp.data || []
      setStrategies(nextStrategies)
      if (selected?.id && !nextStrategies.some((item: PaperStrategySummary) => item.id === selected.id)) {
        setSelected(null)
        setOrders([])
      }
    } catch (e) {
      message.error(e instanceof Error ? e.message : '加载模拟盘失败')
    } finally {
      setLoading(false)
    }
  }, [showStopped, selected?.id])

  useEffect(() => {
    load()
  }, [load])

  const loadDetail = useCallback(async (id: number) => {
    try {
      setDetailLoading(true)
      const [detailResp, ordersResp] = await Promise.all([
        paperApi.getStrategy(id),
        paperApi.getOrders(id)
      ])
      setSelected(detailResp.data)
      setOrders(ordersResp.data || [])
    } catch (e) {
      message.error(e instanceof Error ? e.message : '加载详情失败')
    } finally {
      setDetailLoading(false)
    }
  }, [])

  const updateStatus = useCallback(async (id: number, status: 'active' | 'paused' | 'stopped') => {
    try {
      await paperApi.updateStrategy(id, status)
      message.success('状态已更新')
      await load()
      if (selected?.id === id) {
        await loadDetail(id)
      }
    } catch (e) {
      message.error(e instanceof Error ? e.message : '更新状态失败')
    }
  }, [load, loadDetail, selected?.id])

  const settleOne = useCallback(async (id: number, force = false) => {
    try {
      setSettling(true)
      await paperApi.settleStrategy(id, force)
      message.success(force ? '策略已强制结算' : '策略已结算')
      await load()
      await loadDetail(id)
    } catch (e) {
      message.error(e instanceof Error ? e.message : '策略结算失败')
    } finally {
      setSettling(false)
    }
  }, [load, loadDetail])

  const settleAll = useCallback(async () => {
    try {
      setSettling(true)
      await paperApi.settleAll()
      message.success('全部运行中策略已结算')
      await load()
      if (selected?.id) {
        await loadDetail(selected.id)
      }
    } catch (e) {
      message.error(e instanceof Error ? e.message : '全部结算失败')
    } finally {
      setSettling(false)
    }
  }, [load, loadDetail, selected?.id])

  const emptyDescription = useMemo(() => (
    showStopped ? '暂无模拟盘策略' : '暂无活跃/暂停中的模拟盘策略，请先在策略回测中保存并上模拟盘'
  ), [showStopped])

  return (
    <div style={{ padding: 24 }}>
      <Card
        title={<span><LineChartOutlined style={{ marginRight: 8 }} />模拟盘</span>}
        extra={
          <Space>
            <Space size="small">
              <span style={{ color: '#666' }}>显示已停止</span>
              <Switch checked={showStopped} onChange={setShowStopped} />
            </Space>
            <Button loading={settling} onClick={settleAll}>全部结算</Button>
            <Button icon={<ReloadOutlined />} onClick={load}>刷新</Button>
          </Space>
        }
      >
        {loading ? <div style={{ textAlign: 'center', padding: 48 }}><Spin /></div> : strategies.length === 0 ? (
          <Empty description={emptyDescription} />
        ) : (
          <List
            dataSource={strategies}
            renderItem={(item) => (
              <List.Item
                onClick={() => loadDetail(item.id)}
                style={{ cursor: 'pointer' }}
                actions={[
                  <Button key="settle" size="small" loading={settling && selected?.id === item.id} onClick={(e) => { e.stopPropagation(); settleOne(item.id, true) }}>立即结算</Button>,
                  item.status === 'active' ? (
                    <Button key="pause" size="small" icon={<PauseCircleOutlined />} onClick={(e) => { e.stopPropagation(); updateStatus(item.id, 'paused') }}>暂停</Button>
                  ) : item.status === 'paused' ? (
                    <Button key="resume" size="small" icon={<PlayCircleOutlined />} onClick={(e) => { e.stopPropagation(); updateStatus(item.id, 'active') }}>恢复</Button>
                  ) : null,
                  item.status !== 'stopped' ? (
                    <Button key="stop" danger size="small" icon={<StopOutlined />} onClick={(e) => { e.stopPropagation(); updateStatus(item.id, 'stopped') }}>移除</Button>
                  ) : null
                ].filter(Boolean as any)}
              >
                <List.Item.Meta
                  title={<Space><span>{item.name}</span><Tag color={statusColor[item.status] || 'default'}>{statusLabel[item.status] || item.status}</Tag></Space>}
                  description={
                    <Space size="large">
                      <span>回测ID: {item.backtest_id}</span>
                      <span>净值: ¥{fmt(item.current_value)}</span>
                      <span>收益: {pct(item.total_return)}</span>
                      {item.last_rebalance_date ? <span>上次换仓: {item.last_rebalance_date}</span> : null}
                      {item.next_rebalance_date ? <span>下次换仓: {item.next_rebalance_date}</span> : null}
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        )}
      </Card>

      {selected && (
        <Card
          title={`策略详情 - ${selected.name}`}
          style={{ marginTop: 24 }}
          loading={detailLoading}
          extra={
            <Space>
              <Button loading={settling} onClick={() => settleOne(selected.id, true)}>立即结算</Button>
              <Button onClick={() => loadDetail(selected.id)}>刷新详情</Button>
            </Space>
          }
        >
          <Space size="large" wrap style={{ marginBottom: 24 }}>
            <Statistic title="初始资金" value={selected.initial_capital} formatter={(v) => `¥${fmt(Number(v))}`} />
            <Statistic title="当前净值" value={selected.current_value} formatter={(v) => `¥${fmt(Number(v))}`} />
            <Statistic title="累计收益" value={selected.total_return * 100} precision={2} suffix="%" />
          </Space>

          <Table
            rowKey="id"
            dataSource={orders}
            pagination={false}
            size="small"
            columns={[
              { title: '日期', dataIndex: 'date' },
              { title: '股票', dataIndex: 'stock_code' },
              { title: '方向', dataIndex: 'direction' },
              { title: '股数', dataIndex: 'shares' },
              { title: '价格', dataIndex: 'price' },
              { title: '金额', dataIndex: 'amount' },
            ]}
          />
        </Card>
      )}
    </div>
  )
}

export default PaperTrading
