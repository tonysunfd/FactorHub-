import { useEffect, useMemo, useState } from 'react'
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
  Tooltip,
  Upload,
  List,
  Typography,
  Checkbox,
  Switch,
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
import type { UploadProps } from 'antd'
import { api } from '@/services/api'
import './FactorManagement.css'

const { TextArea } = Input
const { Option } = Select
const { Text } = Typography

interface FactorTaskSnapshot {
  task_id?: string | number
  source?: string
  payload?: Record<string, any>
}

interface Factor {
  id: number
  name: string
  code: string
  category: string
  source: 'preset' | 'user'
  description?: string
  formula_type?: string
  task_metadata?: Record<string, any>
  latest_task_snapshot?: FactorTaskSnapshot
  task_snapshots?: FactorTaskSnapshot[]
  target_stock_code?: string
  target_universe?: string
  scope_type?: 'base' | 'stock' | 'universe'
  origin_type?: string
}

type FactorTabKey = 'base' | 'stock' | 'universe' | 'auto_mined' | 'paper' | 'preset' | 'all'
type FactorScopeType = 'base' | 'stock' | 'universe'
type FactorOriginType = 'preset' | 'manual' | 'genetic_mining' | 'auto_mining' | 'paper_factor' | 'copied' | 'unknown'

interface PaperFactorFileItem {
  name: string
  path: string
  size: number
  modified_at: number
  source_type?: string
  source_url?: string
}

interface PaperFactorLibraryEntry {
  id: string
  name: string
  display_name?: string
  description?: string
  category?: string
  paper_title?: string
  paper_file?: string
  formulation?: string
  variables?: Record<string, any>
  status?: string
  linked_factor_id?: number
  linked_factor_name?: string
}

interface PaperSearchResultItem {
  source: string
  id?: string
  title?: string
  abstract?: string | null
  authors?: string[]
  published_at?: string | null
  pdf_url?: string | null
  landing_url?: string | null
  external_id?: string | null
  can_download_pdf?: boolean
}

interface PaperRuntimeStatus {
  available: boolean
  active_path?: string | null
  checked_paths?: Array<{ path: string; exists: boolean }>
}

const normalizeText = (value?: string | null) => (value || '').trim()

const extractParams = (factor: Factor): Record<string, any> => {
  const snapshotPayload = factor.latest_task_snapshot?.payload || {}
  const snapshotParams = snapshotPayload?.params
  if (snapshotParams && typeof snapshotParams === 'object') return snapshotParams

  const taskDetails = factor.task_metadata?.task_details
  if (taskDetails?.params && typeof taskDetails.params === 'object') return taskDetails.params

  const metadataParams = factor.task_metadata?.params
  if (metadataParams && typeof metadataParams === 'object') return metadataParams

  return {}
}

const extractTargetStockCode = (factor: Factor): string => {
  const params = extractParams(factor)
  const candidates = [
    params.stock_code,
    params.symbol,
    factor.target_stock_code,
    factor.task_metadata?.stock_code,
    factor.task_metadata?.symbol,
  ]
  const matched = candidates.find(v => typeof v === 'string' && normalizeText(v))
  if (matched) return normalizeText(matched as string)

  const nameMatch = factor.name.match(/(?:^|_)(\d{6}(?:\.(?:SH|SZ))?)(?:_|$)/i)
  return normalizeText(nameMatch?.[1] || '')
}

const extractTargetUniverse = (factor: Factor): string => {
  const params = extractParams(factor)
  const candidates = [
    params.universe,
    params.pool,
    factor.target_universe,
    factor.task_metadata?.universe,
    factor.task_metadata?.pool,
  ]
  const matched = candidates.find(v => typeof v === 'string' && normalizeText(v))
  if (matched) return normalizeText(matched as string)

  const nameMatch = factor.name.match(/(?:^|_)(hs300|zz500|zz1000|all)(?:_|$)/i)
  return normalizeText(nameMatch?.[1] || '')
}

const getFactorScopeType = (factor: Factor): FactorScopeType => {
  if (factor.source === 'preset') return 'base'
  if (extractTargetStockCode(factor)) return 'stock'
  if (extractTargetUniverse(factor)) return 'universe'
  return 'base'
}

const isAutoMinedFactor = (factor: Factor) => factor.source === 'user' && factor.category === '自动挖掘'

const getFactorOriginType = (factor: Factor): FactorOriginType => {
  if (factor.source === 'preset') return 'preset'

  const snapshotSource = normalizeText(factor.latest_task_snapshot?.source).toLowerCase()
  const metadataSource = normalizeText((factor.task_metadata?.source as string) || '').toLowerCase()
  const category = normalizeText(factor.category)
  const name = normalizeText(factor.name)

  if (snapshotSource.includes('auto') || metadataSource.includes('auto') || category === '自动挖掘') {
    return 'auto_mining'
  }
  if (snapshotSource.includes('paper') || metadataSource.includes('paper') || category === '论文因子') {
    return 'paper_factor'
  }
  if (snapshotSource.includes('copy') || metadataSource.includes('copy') || /_\d+$/.test(name)) {
    return 'copied'
  }
  if (category === '遗传挖掘') {
    return 'genetic_mining'
  }
  if (factor.source === 'user') {
    return 'manual'
  }
  return 'unknown'
}

const getScopeLabel = (factor: Factor) => {
  const scopeType = getFactorScopeType(factor)
  const stockCode = extractTargetStockCode(factor)
  const universe = extractTargetUniverse(factor)

  if (scopeType === 'stock') return `个股: ${stockCode}`
  if (scopeType === 'universe') return `股票池: ${universe || '未标注'}`
  return '基础'
}

const getOriginLabel = (originType: FactorOriginType) => {
  const mapping: Record<FactorOriginType, string> = {
    preset: '系统预置',
    manual: '手工创建',
    genetic_mining: '遗传挖掘',
    auto_mining: '自动挖掘',
    paper_factor: '论文因子',
    copied: '复制因子',
    unknown: '未知来源',
  }
  return mapping[originType]
}

const getOriginTagColor = (originType: FactorOriginType) => {
  const mapping: Record<FactorOriginType, string> = {
    preset: 'success',
    manual: 'blue',
    genetic_mining: 'purple',
    auto_mining: 'gold',
    paper_factor: 'volcano',
    copied: 'cyan',
    unknown: 'default',
  }
  return mapping[originType]
}

const getScopeTagColor = (scopeType: FactorScopeType) => {
  const mapping: Record<FactorScopeType, string> = {
    base: 'default',
    stock: 'magenta',
    universe: 'geekblue',
  }
  return mapping[scopeType]
}

const FactorManagement: React.FC = () => {
  const navigate = useNavigate()
  const [factors, setFactors] = useState<Factor[]>([])
  const [filteredFactors, setFilteredFactors] = useState<Factor[]>([])
  const [loading, setLoading] = useState(false)
  const [categories, setCategories] = useState<string[]>([])

  // Tab状态
  const [activeTab, setActiveTab] = useState<FactorTabKey>('base')

  // 筛选状态
  const [categoryFilter, setCategoryFilter] = useState<string>('')
  const [scopeFilter, setScopeFilter] = useState<string>('')
  const [originFilter, setOriginFilter] = useState<string>('')
  const [stockCodeFilter, setStockCodeFilter] = useState<string>('')
  const [universeFilter, setUniverseFilter] = useState<string>('')
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
  const [paperFiles, setPaperFiles] = useState<PaperFactorFileItem[]>([])
  const [paperRuntimeStatus, setPaperRuntimeStatus] = useState<PaperRuntimeStatus | null>(null)
  const [paperLoading, setPaperLoading] = useState(false)
  const [paperDownloadUrl, setPaperDownloadUrl] = useState('')
  const [paperDownloadFilename, setPaperDownloadFilename] = useState('')
  const [selectedPaperFiles, setSelectedPaperFiles] = useState<string[]>([])
  const [extractingPaperFactors, setExtractingPaperFactors] = useState(false)
  const [paperFactorLibrary, setPaperFactorLibrary] = useState<PaperFactorLibraryEntry[]>([])
  const [selectedPaperFactorEntries, setSelectedPaperFactorEntries] = useState<string[]>([])
  const [convertingPaperFactors, setConvertingPaperFactors] = useState(false)
  const [paperSearchQuery, setPaperSearchQuery] = useState('')
  const [paperSearchResults, setPaperSearchResults] = useState<PaperSearchResultItem[]>([])
  const [selectedPaperSearchResults, setSelectedPaperSearchResults] = useState<string[]>([])
  const [importingPaperSearchResults, setImportingPaperSearchResults] = useState(false)
  const [paperSearchOnlyDownloadable, setPaperSearchOnlyDownloadable] = useState(true)
  const [paperSearchPage, setPaperSearchPage] = useState(1)
  const paperSearchPageSize = 5
  const [paperAutoSources, setPaperAutoSources] = useState<string[]>(['openalex', 'arxiv'])
  const [refreshingPaperSources, setRefreshingPaperSources] = useState(false)
  const [paperAutoExtract, setPaperAutoExtract] = useState(true)

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

  const loadPaperStorage = async () => {
    try {
      const response = await api.getPaperFactorStorage() as any
      if (!response.success) {
        message.error('加载论文因子存储目录失败')
      }
    } catch (error) {
      message.error('加载论文因子存储目录失败')
    }
  }

  const loadPaperRuntimeStatus = async () => {
    try {
      const response = await api.getPaperFactorRuntimeStatus() as any
      if (response.success) {
        setPaperRuntimeStatus(response.data || null)
      }
    } catch (error) {
      message.error('加载论文抽取运行时状态失败')
    }
  }

  const loadPaperFiles = async () => {
    setPaperLoading(true)
    try {
      const response = await api.getPaperFactorFiles() as any
      if (response.success) {
        setPaperFiles(response.data || [])
        setSelectedPaperFiles(prev => prev.filter(name => (response.data || []).some((item: PaperFactorFileItem) => item.name === name)))
      }
    } catch (error) {
      message.error('加载论文 PDF 列表失败')
    } finally {
      setPaperLoading(false)
    }
  }

  const loadPaperFactorLibrary = async () => {
    try {
      const response = await api.getPaperFactorLibrary() as any
      if (response.success) {
        setPaperFactorLibrary(response.data || [])
        setSelectedPaperFactorEntries(prev =>
          prev.filter(id => (response.data || []).some((item: PaperFactorLibraryEntry) => item.id === id))
        )
      }
    } catch (error) {
      message.error('加载论文因子库失败')
    }
  }

  const factorTabCounts = useMemo(() => ({
    base: factors.filter(f => getFactorScopeType(f) === 'base').length,
    stock: factors.filter(f => getFactorScopeType(f) === 'stock').length,
    universe: factors.filter(f => getFactorScopeType(f) === 'universe').length,
    auto_mined: factors.filter(isAutoMinedFactor).length,
    paper: factors.filter(f => getFactorOriginType(f) === 'paper_factor' || f.category === '论文因子').length,
    preset: factors.filter(f => f.source === 'preset').length,
    all: factors.length,
  }), [factors])

  const universeOptions = useMemo(() => {
    return [...new Set(factors.map(extractTargetUniverse).filter(Boolean))]
  }, [factors])

  const tabDefaults = useMemo(() => {
    switch (activeTab) {
      case 'base':
        return { scopeType: 'base', originType: '' }
      case 'stock':
        return { scopeType: 'stock', originType: '' }
      case 'universe':
        return { scopeType: 'universe', originType: '' }
      case 'preset':
        return { scopeType: 'base', originType: 'preset' }
      case 'paper':
        return { scopeType: '', originType: 'paper_factor' }
      case 'auto_mined':
        return { scopeType: '', originType: 'auto_mining' }
      default:
        return { scopeType: '', originType: '' }
    }
  }, [activeTab])

  const effectiveScopeFilter = scopeFilter || tabDefaults.scopeType
  const effectiveOriginFilter = originFilter || tabDefaults.originType

  const activeFilterSummary = useMemo(() => {
    const summary: string[] = []
    if (activeTab !== 'all') summary.push(`视图: ${activeTab}`)
    if (categoryFilter) summary.push(`分类: ${categoryFilter}`)
    if (effectiveScopeFilter) summary.push(`范围: ${effectiveScopeFilter}`)
    if (effectiveOriginFilter) summary.push(`来源: ${effectiveOriginFilter}`)
    if (stockCodeFilter) summary.push(`股票: ${stockCodeFilter}`)
    if (universeFilter) summary.push(`股票池: ${universeFilter}`)
    if (searchText) summary.push(`搜索: ${searchText}`)
    return summary
  }, [activeTab, categoryFilter, effectiveScopeFilter, effectiveOriginFilter, stockCodeFilter, universeFilter, searchText])

  // 筛选因子
  useEffect(() => {
    let filtered = [...factors]

    if (activeTab === 'base') {
      filtered = filtered.filter(f => getFactorScopeType(f) === 'base')
    } else if (activeTab === 'stock') {
      filtered = filtered.filter(f => getFactorScopeType(f) === 'stock')
    } else if (activeTab === 'universe') {
      filtered = filtered.filter(f => getFactorScopeType(f) === 'universe')
    } else if (activeTab === 'auto_mined') {
      filtered = filtered.filter(isAutoMinedFactor)
    } else if (activeTab === 'paper') {
      filtered = filtered.filter(f => getFactorOriginType(f) === 'paper_factor' || f.category === '论文因子')
    } else if (activeTab === 'preset') {
      filtered = filtered.filter(f => f.source === 'preset')
    }

    if (categoryFilter) {
      filtered = filtered.filter(f => f.category === categoryFilter)
    }

    if (effectiveScopeFilter) {
      filtered = filtered.filter(f => getFactorScopeType(f) === effectiveScopeFilter)
    }

    if (effectiveOriginFilter) {
      filtered = filtered.filter(f => getFactorOriginType(f) === effectiveOriginFilter)
    }

    if (stockCodeFilter) {
      const keyword = stockCodeFilter.toLowerCase()
      filtered = filtered.filter(f => extractTargetStockCode(f).toLowerCase().includes(keyword))
    }

    if (universeFilter) {
      const keyword = universeFilter.toLowerCase()
      filtered = filtered.filter(f => extractTargetUniverse(f).toLowerCase().includes(keyword))
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
  }, [factors, categoryFilter, effectiveScopeFilter, effectiveOriginFilter, stockCodeFilter, universeFilter, searchText, activeTab])

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
      width: 260,
      render: (text: string, record: Factor) => {
        const scopeType = getFactorScopeType(record)
        const originType = getFactorOriginType(record)
        const codePreview = String(record.code || '').replace(/\s+/g, ' ').trim().slice(0, 80)
        return (
          <div>
            <div className="factor-name">{text}</div>
            <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              <Tag color={getScopeTagColor(scopeType)}>{getScopeLabel(record)}</Tag>
              <Tag color={getOriginTagColor(originType)}>{getOriginLabel(originType)}</Tag>
            </div>
            {codePreview && (
              <div className="factor-code" title={record.code}>
                {codePreview}{record.code.length > 80 ? '…' : ''}
              </div>
            )}
          </div>
        )
      }
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
      width: 120,
      render: (_: string, record: Factor) => {
        const originType = getFactorOriginType(record)
        return <Tag color={getOriginTagColor(originType)}>{getOriginLabel(originType)}</Tag>
      }
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
            onClick={() => navigate(`/factor-detail?id=${record.id}${isAutoMinedFactor(record) ? '&tab=task-details' : ''}`)}
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
    loadPaperStorage()
    loadPaperRuntimeStatus()
    loadPaperFiles()
    loadPaperFactorLibrary()
  }, [])

  const uploadProps: UploadProps = {
    accept: '.pdf',
    showUploadList: false,
    customRequest: async ({ file, onSuccess, onError }) => {
      try {
        await api.uploadPaperFactorFile(file as File)
        message.success('PDF 已保存到 WebDAV 文献目录')
        await loadPaperFiles()
        onSuccess?.({}, new XMLHttpRequest())
      } catch (error: any) {
        const err = error instanceof Error ? error : new Error('上传失败')
        message.error(err.message || '上传 PDF 失败')
        onError?.(err)
      }
    }
  }

  const handleDownloadPaperToWebdav = async () => {
    if (!paperDownloadUrl.trim()) {
      message.warning('请输入第三方 PDF 下载地址')
      return
    }
    try {
      await api.downloadPaperFactorFile({
        url: paperDownloadUrl.trim(),
        filename: paperDownloadFilename.trim() || undefined,
      })
      message.success('第三方 PDF 已保存到 WebDAV 文献目录')
      setPaperDownloadUrl('')
      setPaperDownloadFilename('')
      await loadPaperFiles()
    } catch (error: any) {
      message.error(error?.message || '第三方 PDF 下载失败')
    }
  }

  const getPaperSearchRowKey = (item: PaperSearchResultItem) => {
    return item.external_id || item.id || item.landing_url || item.title || Math.random().toString()
  }

  const filteredPaperSearchResults = useMemo(() => {
    const deduped = new Map<string, PaperSearchResultItem>()
    for (const item of paperSearchResults) {
      const key = getPaperSearchRowKey(item)
      if (!deduped.has(key)) {
        if (!paperSearchOnlyDownloadable || item.can_download_pdf) {
          deduped.set(key, item)
        }
      }
    }
    return Array.from(deduped.values())
  }, [paperSearchResults, paperSearchOnlyDownloadable])

  const pagedPaperSearchResults = useMemo(() => {
    const start = (paperSearchPage - 1) * paperSearchPageSize
    return filteredPaperSearchResults.slice(start, start + paperSearchPageSize)
  }, [filteredPaperSearchResults, paperSearchPage])

  const handleImportPaperSearchResults = async () => {
    const selectedItems = paperSearchResults.filter(item => selectedPaperSearchResults.includes(getPaperSearchRowKey(item)))
    if (!selectedItems.length) {
      message.warning('请先选择至少一个论文搜索结果')
      return
    }
    setImportingPaperSearchResults(true)
    try {
      const response = await api.importPaperSearchResults({
        items: selectedItems,
        category: '论文因子',
        auto_extract: true,
      }) as any
      if (response.success) {
        message.success(response.message || '已导入论文并录入论文因子草稿')
        setSelectedPaperSearchResults([])
        await loadPaperFiles()
        await loadPaperFactorLibrary()
      } else {
        message.error(response.message || '论文导入失败')
      }
    } catch (error: any) {
      message.error(error?.message || '论文导入失败')
    } finally {
      setImportingPaperSearchResults(false)
    }
  }

  const handleRefreshPaperSources = async () => {
    if (!paperSearchQuery.trim()) {
      message.warning('请输入自动更新关键词')
      return
    }
    if (!paperAutoSources.length) {
      message.warning('请至少选择一个论文源')
      return
    }
    setRefreshingPaperSources(true)
    try {
      const response = await api.refreshPaperSources({
        query: paperSearchQuery.trim(),
        sources: paperAutoSources,
        limit_per_source: 10,
        auto_extract: paperAutoExtract,
      }) as any
      if (response.success) {
        setPaperSearchResults(response.data?.matched_items || [])
        setPaperSearchPage(1)
        message.success(response.message || '已自动更新相关文章')
        await loadPaperFiles()
        await loadPaperFactorLibrary()
      } else {
        message.error(response.message || '自动更新相关文章失败')
      }
    } catch (error: any) {
      message.error(error?.message || '自动更新相关文章失败')
    } finally {
      setRefreshingPaperSources(false)
    }
  }

  const handleExtractPaperFactors = async () => {
    if (!selectedPaperFiles.length) {
      message.warning('请先选择至少一个 PDF 文件')
      return
    }
    setExtractingPaperFactors(true)
    try {
      const response = await api.extractPaperFactors({
        filenames: selectedPaperFiles,
        category: '论文因子',
      }) as any
      if (response.success) {
        const count = response.data?.saved_entries?.length || 0
        message.success(`已录入 ${count} 个论文因子草稿，请二次确认后再转化`)
        setSelectedPaperFiles([])
        await loadPaperFactorLibrary()
      } else {
        message.error(response.message || '论文因子抽取失败')
      }
    } catch (error: any) {
      message.error(error?.message || '论文因子抽取失败')
    } finally {
      setExtractingPaperFactors(false)
    }
  }

  const handleConvertPaperFactors = async () => {
    if (!selectedPaperFactorEntries.length) {
      message.warning('请先选择至少一个论文因子草稿')
      return
    }
    setConvertingPaperFactors(true)
    try {
      const response = await api.convertPaperFactors({
        entry_ids: selectedPaperFactorEntries,
        category: '论文因子',
      }) as any
      if (response.success) {
        const count = response.data?.converted_factors?.length || 0
        const failedCount = response.data?.failed_entries?.length || 0
        if (failedCount > 0) {
          message.warning(`已转换 ${count} 个论文因子，${failedCount} 个草稿仍未成功转换`)
        } else {
          message.success(`已转换 ${count} 个论文因子`)
        }
        setSelectedPaperFactorEntries([])
        await loadPaperFactorLibrary()
        await loadFactors()
      } else {
        message.error(response.message || '论文因子转化失败')
      }
    } catch (error: any) {
      message.error(error?.message || '论文因子转化失败')
    } finally {
      setConvertingPaperFactors(false)
    }
  }

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
          onChange={(key) => setActiveTab(key as FactorTabKey)}
          items={[
            {
              key: 'base',
              label: `基础因子 (${factorTabCounts.base})`
            },
            {
              key: 'stock',
              label: `个股优化 (${factorTabCounts.stock})`
            },
            {
              key: 'universe',
              label: `股票池优化 (${factorTabCounts.universe})`
            },
            {
              key: 'auto_mined',
              label: `自动挖掘 (${factorTabCounts.auto_mined})`
            },
            {
              key: 'paper',
              label: `论文因子 (${factorTabCounts.paper})`
            },
            {
              key: 'preset',
              label: `系统预置 (${factorTabCounts.preset})`
            },
            {
              key: 'all',
              label: `全部 (${factorTabCounts.all})`
            }
          ]}
        />
      </Card>

      {/* 工具栏 */}
      <Card className="toolbar-card" variant="borderless">
        {activeFilterSummary.length > 0 && (
          <div style={{ marginBottom: 12, fontSize: 12, color: '#64748b' }}>
            当前生效筛选：{activeFilterSummary.join(' · ')}
          </div>
        )}
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
                  onChange={value => setCategoryFilter(value || '')}
                >
                  {categories.map(cat => (
                    <Option key={cat} value={cat}>{cat}</Option>
                  ))}
                </Select>
              </div>
              <div>
                <div className="filter-label">适用范围</div>
                <Select
                  placeholder="全部"
                  allowClear
                  style={{ width: 140 }}
                  value={scopeFilter || undefined}
                  onChange={value => setScopeFilter(value || '')}
                >
                  <Option value="base">基础</Option>
                  <Option value="stock">个股</Option>
                  <Option value="universe">股票池</Option>
                </Select>
              </div>
              <div>
                <div className="filter-label">来源类型</div>
                <Select
                  placeholder="全部"
                  allowClear
                  style={{ width: 150 }}
                  value={originFilter || undefined}
                  onChange={value => setOriginFilter(value || '')}
                >
                  <Option value="preset">系统预置</Option>
                  <Option value="manual">手工创建</Option>
                  <Option value="genetic_mining">遗传挖掘</Option>
                  <Option value="auto_mining">自动挖掘</Option>
                  <Option value="paper_factor">论文因子</Option>
                  <Option value="copied">复制因子</Option>
                </Select>
              </div>
              <div>
                <div className="filter-label">股票代码</div>
                <Input
                  placeholder="如 600519.SH"
                  style={{ width: 150 }}
                  value={stockCodeFilter}
                  onChange={e => setStockCodeFilter(e.target.value)}
                  allowClear
                />
              </div>
              <div>
                <div className="filter-label">股票池</div>
                <Select
                  placeholder="全部"
                  allowClear
                  showSearch
                  style={{ width: 150 }}
                  value={universeFilter || undefined}
                  onChange={value => setUniverseFilter(value || '')}
                  optionFilterProp="children"
                >
                  {universeOptions.map(universe => (
                    <Option key={universe} value={universe}>{universe}</Option>
                  ))}
                </Select>
              </div>
              <div>
                <div className="filter-label">搜索</div>
                <Input
                  placeholder="搜索名称 / 说明 / 代码..."
                  prefix={<SearchOutlined />}
                  style={{ width: 220 }}
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
                onClick={() => {
                  setCategoryFilter('')
                  setScopeFilter('')
                  setOriginFilter('')
                  setStockCodeFilter('')
                  setUniverseFilter('')
                  setSearchText('')
                }}
              >
                清空筛选
              </Button>
              <Button
                icon={<ReloadOutlined />}
                onClick={() => {
                  loadFactors()
                  if (activeTab === 'paper') {
                    loadPaperStorage()
                    loadPaperRuntimeStatus()
                    loadPaperFiles()
                    loadPaperFactorLibrary()
                  }
                }}
                loading={loading}
              >
                刷新
              </Button>
              {activeTab !== 'preset' && (
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

      {activeTab === 'paper' && (
        <Card className="toolbar-card" variant="borderless" style={{ marginTop: 16 }}>
          <div style={{ display: 'grid', gap: 16 }}>
            <div>
              <div className="filter-label" style={{ marginBottom: 8 }}>第三方论文源</div>
              <Space wrap size="middle" align="start">
                <Input
                  placeholder="输入自动更新关键词，例如 factor investing"
                  style={{ width: 360 }}
                  value={paperSearchQuery}
                  onChange={(e) => setPaperSearchQuery(e.target.value)}
                />
                <Select
                  mode="multiple"
                  style={{ width: 240 }}
                  value={paperAutoSources}
                  onChange={(value) => setPaperAutoSources(value)}
                  options={[
                    { label: 'OpenAlex', value: 'openalex' },
                    { label: 'arXiv', value: 'arxiv' },
                  ]}
                />
                <Button type="primary" onClick={handleRefreshPaperSources} loading={refreshingPaperSources}>
                  自动更新相关文章
                </Button>
                <Checkbox
                  checked={paperSearchOnlyDownloadable}
                  onChange={(e) => {
                    setPaperSearchOnlyDownloadable(e.target.checked)
                    setPaperSearchPage(1)
                  }}
                >
                  仅显示可下载 PDF
                </Checkbox>
                <Space>
                  <Text type="secondary">自动送入草稿库</Text>
                  <Switch checked={paperAutoExtract} onChange={setPaperAutoExtract} />
                </Space>
                <Button
                  onClick={handleImportPaperSearchResults}
                  loading={importingPaperSearchResults}
                  disabled={!selectedPaperSearchResults.length}
                >
                  手动导入选中结果
                </Button>
              </Space>
              <div style={{ marginTop: 8, fontSize: 12, color: '#64748b' }}>
                共 {paperSearchResults.length} 条结果，去重后 {filteredPaperSearchResults.length} 条。
              </div>
              <List
                style={{ marginTop: 12 }}
                bordered
                dataSource={pagedPaperSearchResults}
                locale={{ emptyText: '暂无论文搜索结果' }}
                renderItem={(item) => {
                  const rowKey = getPaperSearchRowKey(item)
                  return (
                    <List.Item>
                      <Space direction="vertical" size={4} style={{ width: '100%' }}>
                        <Space wrap>
                          <Checkbox
                            checked={selectedPaperSearchResults.includes(rowKey)}
                            disabled={!item.can_download_pdf}
                            onChange={(e) => {
                              setSelectedPaperSearchResults(prev =>
                                e.target.checked
                                  ? [...prev, rowKey]
                                  : prev.filter(id => id !== rowKey)
                              )
                            }}
                          />
                          <Text strong>{item.title || '-'}</Text>
                          <Tag color={item.source === 'openalex' ? 'cyan' : 'geekblue'}>
                            {item.source === 'openalex' ? 'OpenAlex' : 'arXiv'}
                          </Tag>
                          <Tag color={item.can_download_pdf ? 'green' : 'default'}>
                            {item.can_download_pdf ? '可下载 PDF' : '无 PDF'}
                          </Tag>
                        </Space>
                        <Text type="secondary">
                          作者：{(item.authors || []).join('、') || '-'} · 日期：{item.published_at || '-'}
                        </Text>
                        <Text type="secondary">{item.landing_url || item.pdf_url || '-'}</Text>
                      </Space>
                    </List.Item>
                  )
                }}
              />
              {filteredPaperSearchResults.length > paperSearchPageSize && (
                <div style={{ marginTop: 12, display: 'flex', justifyContent: 'flex-end' }}>
                  <Space>
                    <Button
                      size="small"
                      disabled={paperSearchPage <= 1}
                      onClick={() => setPaperSearchPage(prev => Math.max(prev - 1, 1))}
                    >
                      上一页
                    </Button>
                    <Text type="secondary">第 {paperSearchPage} / {Math.ceil(filteredPaperSearchResults.length / paperSearchPageSize)} 页</Text>
                    <Button
                      size="small"
                      disabled={paperSearchPage >= Math.ceil(filteredPaperSearchResults.length / paperSearchPageSize)}
                      onClick={() =>
                        setPaperSearchPage(prev => Math.min(prev + 1, Math.ceil(filteredPaperSearchResults.length / paperSearchPageSize)))
                      }
                    >
                      下一页
                    </Button>
                  </Space>
                </div>
              )}
            </div>

            <div>
              <div className="filter-label">论文抽取运行时</div>
              {paperRuntimeStatus?.available ? (
                <Space direction="vertical" size={4}>
                  <Tag color="green">已就绪</Tag>
                  <Text copyable>{paperRuntimeStatus.active_path || '-'}</Text>
                </Space>
              ) : (
                <Space direction="vertical" size={4}>
                  <Tag color="red">未就绪</Tag>
                  <Text type="secondary">当前还没有在 FactorHub 内找到可用的 RD-Agent 运行时。</Text>
                </Space>
              )}
            </div>
            <Space wrap size="middle" align="start">
              <Upload {...uploadProps}>
                <Button icon={<PlusOutlined />}>上传 PDF 到 WebDAV</Button>
              </Upload>
              <Input
                placeholder="第三方 PDF 下载地址"
                style={{ width: 320 }}
                value={paperDownloadUrl}
                onChange={(e) => setPaperDownloadUrl(e.target.value)}
              />
              <Input
                placeholder="可选：保存文件名.pdf"
                style={{ width: 220 }}
                value={paperDownloadFilename}
                onChange={(e) => setPaperDownloadFilename(e.target.value)}
              />
              <Button type="primary" onClick={handleDownloadPaperToWebdav}>
                下载到 WebDAV
              </Button>
              <Button icon={<ReloadOutlined />} onClick={loadPaperFiles} loading={paperLoading}>
                刷新 PDF 列表
              </Button>
              <Button
                type="primary"
                onClick={handleExtractPaperFactors}
                loading={extractingPaperFactors}
                disabled={!selectedPaperFiles.length}
              >
                抽取到论文因子库
              </Button>
              <Button
                onClick={handleConvertPaperFactors}
                loading={convertingPaperFactors}
                disabled={!selectedPaperFactorEntries.length}
              >
                确认转化为 FactorHub 因子
              </Button>
            </Space>

            <List
              bordered
              loading={paperLoading}
              dataSource={paperFiles}
              locale={{ emptyText: 'WebDAV 文献目录中暂无 PDF' }}
              renderItem={(item) => (
                <List.Item>
                  <Space direction="vertical" size={2} style={{ width: '100%' }}>
                    <Space wrap>
                      <Checkbox
                        checked={selectedPaperFiles.includes(item.name)}
                        onChange={(e) => {
                          setSelectedPaperFiles(prev =>
                            e.target.checked
                              ? [...prev, item.name]
                              : prev.filter(name => name !== item.name)
                          )
                        }}
                      />
                      <Text strong>{item.name}</Text>
                      <Tag color="volcano">{item.source_type === 'third_party_download' ? '第三方下载' : '上传文件'}</Tag>
                    </Space>
                    <Text type="secondary">{item.path}</Text>
                    <Text type="secondary">
                      大小：{(item.size / 1024 / 1024).toFixed(2)} MB · 更新时间：
                      {new Date(item.modified_at * 1000).toLocaleString()}
                    </Text>
                  </Space>
                </List.Item>
              )}
            />

            <div>
              <div className="filter-label" style={{ marginBottom: 8 }}>论文因子库草稿</div>
              <List
                bordered
                dataSource={paperFactorLibrary}
                locale={{ emptyText: '暂未录入论文因子草稿' }}
                renderItem={(item) => (
                  <List.Item>
                    <Space direction="vertical" size={4} style={{ width: '100%' }}>
                      <Space wrap>
                        <Checkbox
                          checked={selectedPaperFactorEntries.includes(item.id)}
                          disabled={item.status === 'converted'}
                          onChange={(e) => {
                            setSelectedPaperFactorEntries(prev =>
                              e.target.checked
                                ? [...prev, item.id]
                                : prev.filter(id => id !== item.id)
                            )
                          }}
                        />
                        <Text strong>{item.display_name || item.name}</Text>
                        <Tag color={item.status === 'converted' ? 'green' : 'gold'}>
                          {item.status === 'converted' ? '已转化' : '待确认'}
                        </Tag>
                        {item.linked_factor_name && (
                          <Tag color="blue">已生成：{item.linked_factor_name}</Tag>
                        )}
                      </Space>
                      <Text type="secondary">{item.description || '-'}</Text>
                      <Text type="secondary">来源文件：{item.paper_file || '-'}</Text>
                      <Text type="secondary">论文公式：{item.formulation || '-'}</Text>
                    </Space>
                  </List.Item>
                )}
              />
            </div>
          </div>
        </Card>
      )}

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
