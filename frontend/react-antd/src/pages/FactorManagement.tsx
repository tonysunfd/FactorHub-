import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Table,
  Button,
  Input,
  Select,
  Space,
  Modal,
  Form,
  message,
  Tag,
  Card,
  Tabs,
  Tooltip
} from 'antd'
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table'
import {
  PlusOutlined,
  ReloadOutlined,
  SearchOutlined,
  DeleteOutlined,
  EyeOutlined,
  CopyOutlined,
  QuestionCircleOutlined,
  WarningOutlined,
  DatabaseOutlined
} from '@ant-design/icons'
import { api } from '@/services/api'
import './FactorManagement.css'

const { TextArea } = Input
const { Option } = Select

interface Factor {
  id: number
  name: string
  code: string
  category: string
  source: 'preset' | 'user'
  description?: string
  formula_type?: string
}


const FactorManagement: React.FC = () => {
  const navigate = useNavigate()
  const [factors, setFactors] = useState<Factor[]>([])
  const [filteredFactors, setFilteredFactors] = useState<Factor[]>([])
  const [loading, setLoading] = useState(false)
  const [categories, setCategories] = useState<string[]>([])

  // Tab状态
  const [activeTab, setActiveTab] = useState<'user' | 'preset'>('user')

  // 筛选状态
  const [categoryFilter, setCategoryFilter] = useState<string>('')
  const [sourceFilter, setSourceFilter] = useState<string>('')
  const [searchText, setSearchText] = useState<string>('')

  // 分页状态
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: 10,
    total: 0
  })

  // 弹窗状态
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [form] = Form.useForm()
  const [selectedFormulaType, setSelectedFormulaType] = useState<string>('expression')

  // 公式类型帮助内容（用于Tooltip）
  const getFormulaHelpContent = (formulaType: string) => {
    if (formulaType === 'expression') {
      return (
        <div style={{ maxWidth: '500px', fontSize: '12px', color: '#fff' }}>
          <div style={{ marginBottom: '12px' }}>
            <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>表达式类型因子</div>
            <p style={{ margin: 0, color: '#ccc', lineHeight: '1.6' }}>使用 pandas 链式语法编写因子表达式</p>
          </div>
          <div style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: '1px solid #444' }}>
            <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>可用字段</div>
            <code style={{ background: 'rgba(255, 255, 255, 0.1)', color: '#4dabf7', padding: '2px 6px', borderRadius: '4px', fontFamily: 'monospace', fontSize: '12px' }}>close, open, high, low, volume, amount</code>
          </div>
          <div style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: '1px solid #444' }}>
            <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>常用函数</div>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>rolling(n).mean()</code> - n日移动平均</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>rolling(n).std()</code> - n日标准差</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>rolling(n).max()</code> / <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>rolling(n).min()</code> - n日最大/最小值</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>shift(n)</code> - 向前移n行</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>diff()</code> - 一阶差分</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>pct_change()</code> - 百分比变化</li>
            </ul>
          </div>
          <div>
            <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>示例</div>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>close.rolling(20).mean()</code> - 20日均线</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>close / close.rolling(20).mean()</code> - 相对20日均线</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>close.pct_change(5)</code> - 5日收益率</li>
            </ul>
          </div>
        </div>
      )
    } else {
      return (
        <div style={{ maxWidth: '600px', fontSize: '12px', color: '#fff' }}>
          <div style={{ marginBottom: '12px' }}>
            <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>函数类型因子</div>
            <p style={{ margin: 0, color: '#ccc', lineHeight: '1.6' }}>支持预定义函数和自定义def函数两种写法</p>
          </div>
          <div style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: '1px solid #444' }}>
            <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>方式一：预定义函数</div>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>RSI(close, 14)</code> - 14日RSI</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>MACD(close, 12, 26, 9)[0]</code> - MACD快线</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>EMA(close, 20)</code> - 20日指数移动平均</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>SMA(close, 60)</code> / <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>MA(close, 60)</code> - 简单移动平均</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>BOLL(close, 20, 2)</code> - 布林带上轨</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>KDJ(high, low, close, 9, 3, 3)[0]</code> - KDJ的K值</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>ATR(high, low, close, 14)</code> - 14日ATR</li>
            </ul>
          </div>
          <div style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: '1px solid #444' }}>
            <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>方式二：自定义def函数</div>
            <p style={{ margin: '0 0 8px 0', color: '#ccc', lineHeight: '1.6', fontSize: '12px' }}>使用Python def语法编写复杂逻辑（<WarningOutlined style={{ color: '#f59e0b' }} /> 函数名必须为 calculate_factor）</p>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <strong style={{ color: '#f59e0b' }}>函数名必须固定为：</strong><code style={{ color: '#f59e0b', background: 'rgba(245, 158, 11, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>def calculate_factor(df):</code></li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• 参数 <code style={{ color: '#4dabf7', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>df</code> 是包含 open/high/low/close/volume 的 DataFrame</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• 必须返回 Series 或可转换为 Series 的数组</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• 支持多行代码、条件判断、循环等复杂逻辑</li>
              <li style={{ padding: '2px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>• <strong style={{ color: '#10b981' }}>✓ 完全兼容麦语言函数：</strong><code style={{ color: '#10b981', background: 'rgba(16, 185, 129, 0.1)', padding: '2px 4px', borderRadius: '3px' }}>REF, HHV, LLV, CROSS, IF, MA, SUM, STD</code> 等</li>
            </ul>
          </div>
          <div>
            <div style={{ fontWeight: 600, marginBottom: '6px', fontSize: '13px', color: '#fff' }}>def函数示例</div>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              <li style={{ padding: '4px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>
                <div style={{ marginBottom: '4px', color: '#ccc' }}>• 条件组合因子：</div>
                <code style={{ display: 'block', color: '#4dabf7', background: 'rgba(0, 0, 0, 0.3)', padding: '6px 8px', borderRadius: '4px', fontSize: '11px', fontFamily: 'monospace', whiteSpace: 'pre-wrap', marginTop: '4px' }}>def calculate_factor(df):
    ma20 = df['close'].rolling(20).mean()
    ma60 = df['close'].rolling(60).mean()
    return (ma20 &gt; ma60).astype(int)</code>
              </li>
              <li style={{ padding: '4px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>
                <div style={{ marginBottom: '4px', color: '#ccc' }}>• 使用麦语言函数：</div>
                <code style={{ display: 'block', color: '#4dabf7', background: 'rgba(0, 0, 0, 0.3)', padding: '6px 8px', borderRadius: '4px', fontSize: '11px', fontFamily: 'monospace', whiteSpace: 'pre-wrap', marginTop: '4px' }}>def calculate_factor(df):
    ma5 = MA(df['close'], 5)
    ma10 = MA(df['close'], 10)
    return CROSS(ma5, ma10).astype(int)</code>
              </li>
              <li style={{ padding: '4px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>
                <div style={{ marginBottom: '4px', color: '#ccc' }}>• 带条件判断的因子：</div>
                <code style={{ display: 'block', color: '#4dabf7', background: 'rgba(0, 0, 0, 0.3)', padding: '6px 8px', borderRadius: '4px', fontSize: '11px', fontFamily: 'monospace', whiteSpace: 'pre-wrap', marginTop: '4px' }}>def calculate_factor(df):
    rsi = RSI(df['close'], 14)
    return np.where(rsi &gt; 70, -1, np.where(rsi &lt; 30, 1, 0))</code>
              </li>
              <li style={{ padding: '4px 0', color: '#fff', fontSize: '12px', lineHeight: '1.6' }}>
                <div style={{ marginBottom: '4px', color: '#ccc' }}>• 波动率加权因子：</div>
                <code style={{ display: 'block', color: '#4dabf7', background: 'rgba(0, 0, 0, 0.3)', padding: '6px 8px', borderRadius: '4px', fontSize: '11px', fontFamily: 'monospace', whiteSpace: 'pre-wrap', marginTop: '4px' }}>def calculate_factor(df):
    ret = df['close'].pct_change()
    vol = ret.rolling(20).std()
    signal = (df['close'] &gt; df['close'].shift(1)).astype(int)
    return signal * vol</code>
              </li>
            </ul>
          </div>
        </div>
      )
    }
  }

  // 加载因子列表
  const loadFactors = async () => {
    setLoading(true)
    try {
      const response = await api.getFactors() as any
      if (response.success) {
        setFactors(response.data)
        // 提取分类列表
        const cats = [...new Set((response.data as Factor[]).map((f: Factor) => f.category).filter(Boolean))] as string[]
        setCategories(cats)
      }
    } catch (error) {
      message.error('加载因子列表失败')
    } finally {
      setLoading(false)
    }
  }

  // Tab切换时自动设置sourceFilter
  useEffect(() => {
    setSourceFilter(activeTab)
    setCategoryFilter('')  // 重置分类筛选
  }, [activeTab])

  // 筛选因子
  useEffect(() => {
    let filtered = [...factors]

    if (categoryFilter) {
      filtered = filtered.filter(f => f.category === categoryFilter)
    }

    if (sourceFilter) {
      filtered = filtered.filter(f => f.source === sourceFilter)
    }

    if (searchText) {
      const searchLower = searchText.toLowerCase()
      filtered = filtered.filter(f =>
        f.name.toLowerCase().includes(searchLower) ||
        (f.description && f.description.toLowerCase().includes(searchLower)) ||
        f.code.toLowerCase().includes(searchLower)
      )
    }

    setFilteredFactors(filtered)
    setPagination(prev => ({ ...prev, total: filtered.length, current: 1 }))
  }, [factors, categoryFilter, sourceFilter, searchText])

  // 创建因子
  const handleCreateFactor = async (values: any) => {
    try {
      const response = await api.createFactor(values) as any
      if (response.success) {
        message.success('因子创建成功')
        setCreateModalVisible(false)
        form.resetFields()
        loadFactors()
      } else {
        message.error(response.message || '创建失败')
      }
    } catch (error) {
      message.error('创建因子失败')
    }
  }

  // 验证公式
  const handleValidateFormula = async () => {
    const code = form.getFieldValue('code')
    const formulaType = form.getFieldValue('formula_type')

    if (!code) {
      message.warning('请先输入因子代码')
      return
    }

    try {
      const response = await api.validateFactor({
        code,
        formula_type: formulaType
      } as any) as any
      if (response.success) {
        message.success('公式验证通过')
      } else {
        message.error(response.message || '公式验证失败')
      }
    } catch (error) {
      message.error('验证失败')
    }
  }

  // 删除因子
  const handleDeleteFactor = async (id: number, name: string) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定要删除因子 "${name}" 吗？`,
      onOk: async () => {
        try {
          const response = await api.deleteFactor(id) as any
          if (response.success) {
            message.success('删除成功')
            loadFactors()
          } else {
            message.error(response.message || '删除失败')
          }
        } catch (error) {
          message.error('删除失败')
        }
      }
    })
  }

  // 复制因子
  const handleCopyFactor = async (id: number) => {
    try {
      const response = await api.copyFactor(id) as any
      if (response.success) {
        message.success(`因子已复制为 "${response.data.name}"`)
        loadFactors()
      } else {
        message.error(response.message || '复制失败')
      }
    } catch (error) {
      message.error('复制失败')
    }
  }

  // 表格列定义
  const columns: ColumnsType<Factor> = [
    {
      title: '因子名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      render: (text: string, record: Factor) => (
        <div>
          <div className="factor-name">{text}</div>
          <div className="factor-code">{record.code}</div>
        </div>
      )
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      width: 120,
      render: (text: string) => <Tag color="blue">{text || '-'}</Tag>
    },
    {
      title: '来源',
      dataIndex: 'source',
      key: 'source',
      width: 100,
      render: (text: string) => (
        <Tag color={text === 'preset' ? 'success' : 'warning'}>
          {text === 'preset' ? '预置' : '自定义'}
        </Tag>
      )
    },
    {
      title: '说明',
      dataIndex: 'description',
      key: 'description',
      width: 300,
      ellipsis: true,
      render: (text: string) => text || '-'
    },
    {
      title: '操作',
      key: 'action',
      width: 200,
      fixed: 'right',
      render: (_: any, record: Factor) => (
        <Space size="small">
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => navigate(`/factor-detail?id=${record.id}`)}
          >
            查看
          </Button>
          <Button
            type="link"
            size="small"
            icon={<CopyOutlined />}
            onClick={() => handleCopyFactor(record.id)}
          >
            复制
          </Button>
          {record.source === 'user' && (
            <Button
              type="link"
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteFactor(record.id, record.name)}
            >
              删除
            </Button>
          )}
        </Space>
      )
    }
  ]

  // 分页配置
  const handleTableChange = (newPagination: TablePaginationConfig) => {
    setPagination({
      current: newPagination.current || 1,
      pageSize: newPagination.pageSize || 20,
      total: filteredFactors.length
    })
  }

  useEffect(() => {
    loadFactors()
  }, [])

  return (
    <div className="factor-management-container">
      {/* 背景装饰 */}
      <div className="bg-gradient"></div>
      <div className="bg-grid"></div>

      <div className="factor-management-content">
        {/* 页面标题 */}
        <div className="page-header">
          <div className="header-content">
            <DatabaseOutlined className="header-icon" />
            <div>
              <h1 className="page-title">因子管理</h1>
              <p className="page-subtitle">创建、管理和分析量化因子</p>
            </div>
          </div>
        </div>

        {/* Tab分类 */}
        <Card className="tab-card" variant="borderless" style={{ marginBottom: '16px' }}>
        <Tabs
          activeKey={activeTab}
          onChange={(key) => setActiveTab(key as 'user' | 'preset')}
          items={[
            {
              key: 'user',
              label: `自定义因子 (${factors.filter(f => f.source === 'user').length})`
            },
            {
              key: 'preset',
              label: `系统预置因子 (${factors.filter(f => f.source === 'preset').length})`
            }
          ]}
        />
      </Card>

      {/* 工具栏 */}
      <Card className="toolbar-card" variant="borderless">
        <div className="toolbar">
          <div className="filters">
            <Space size="middle" wrap>
              <div>
                <div className="filter-label">分类筛选</div>
                <Select
                  placeholder="全部"
                  allowClear
                  style={{ width: 150 }}
                  value={categoryFilter || undefined}
                  onChange={setCategoryFilter}
                >
                  {categories.map(cat => (
                    <Option key={cat} value={cat}>{cat}</Option>
                  ))}
                </Select>
              </div>
              {/* 来源筛选已由Tab替代 */}
              {/* <div>
                <div className="filter-label">来源筛选</div>
                <Select
                  placeholder="全部"
                  allowClear
                  style={{ width: 120 }}
                  value={sourceFilter || undefined}
                  onChange={setSourceFilter}
                >
                  <Option value="preset">预置</Option>
                  <Option value="user">自定义</Option>
                </Select>
              </div> */}
              <div>
                <div className="filter-label">搜索</div>
                <Input
                  placeholder="搜索因子名称..."
                  prefix={<SearchOutlined />}
                  style={{ width: 200 }}
                  value={searchText}
                  onChange={e => setSearchText(e.target.value)}
                  allowClear
                />
              </div>
            </Space>
          </div>
          <div className="actions">
            <Space size="small">
              <Button
                icon={<ReloadOutlined />}
                onClick={loadFactors}
                loading={loading}
              >
                刷新
              </Button>
              {activeTab === 'user' && (
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => {
                    setCreateModalVisible(true)
                  }}
                >
                  新增因子
                </Button>
              )}
            </Space>
          </div>
        </div>
      </Card>

      {/* 因子列表表格 */}
      <Card className="table-card" variant="borderless">
        <Table
          columns={columns}
          dataSource={filteredFactors}
          rowKey="id"
          loading={loading}
          pagination={{
            current: pagination.current,
            pageSize: pagination.pageSize,
            total: pagination.total,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `显示第 ${range[0]} 到 ${range[1]} 条，共 ${total} 条`,
            pageSizeOptions: ['10', '20', '50', '100']
          }}
          onChange={handleTableChange}
          scroll={{ x: 1000 }}
        />
      </Card>

      {/* 新增因子弹窗 */}
      <Modal
        title="创建新因子"
        open={createModalVisible}
        onCancel={() => {
          setCreateModalVisible(false)
          form.resetFields()
        }}
        footer={null}
        width={600}
        destroyOnHidden
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateFactor}
        >
          <Form.Item
            label="因子名称"
            name="name"
            rules={[{ required: true, message: '请输入因子名称' }]}
          >
            <Input placeholder="例如：RSI指标" />
          </Form.Item>

          <Form.Item
            label="分类"
            name="category"
            rules={[{ required: true, message: '请选择分类' }]}
          >
            <Select placeholder="请选择">
              <Option value="技术指标">技术指标</Option>
              <Option value="价格动量">价格动量</Option>
              <Option value="成交量">成交量</Option>
              <Option value="波动率">波动率</Option>
              <Option value="自定义">自定义</Option>
            </Select>
          </Form.Item>

          <Form.Item
            label="说明"
            name="description"
          >
            <TextArea rows={3} placeholder="简要描述因子的含义和用途" />
          </Form.Item>

          <Form.Item
            label="公式类型"
            name="formula_type"
            initialValue="expression"
          >
            <Select onChange={(value) => setSelectedFormulaType(value)}>
              <Option value="expression">表达式</Option>
              <Option value="function">函数</Option>
            </Select>
          </Form.Item>

          <Form.Item
            label={
              <Space>
                <span>因子代码</span>
                <Tooltip title={getFormulaHelpContent(selectedFormulaType)} placement="right" overlayStyle={{ maxWidth: '600px' }}>
                  <QuestionCircleOutlined style={{ color: '#1890ff', cursor: 'help' }} />
                </Tooltip>
              </Space>
            }
            name="code"
            rules={[{ required: true, message: '请输入因子代码' }]}
          >
            <TextArea
              rows={6}
              placeholder={selectedFormulaType === 'expression' ? '例如：close.rolling(20).mean()' : '例如：def calculate_factor(df):\n    return df["close"].rolling(20).mean()'}
              className="font-mono"
              style={{
                backgroundColor: '#f6f8fa',
                fontSize: '14px',
                fontFamily: 'Consolas, Monaco, monospace',
                borderRadius: '6px',
                padding: '12px',
                minHeight: '150px',
                maxHeight: '300px',
                overflowY: 'auto'
              }}
            />
          </Form.Item>

          <Form.Item>
            <Space style={{ width: '100%', justifyContent: 'space-between' }}>
              <Button type="primary" htmlType="submit" style={{ flex: 1 }}>
                创建因子
              </Button>
              <Button onClick={handleValidateFormula}>
                验证公式
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
      </div>
    </div>
  )
}

export default FactorManagement
