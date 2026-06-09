import { Suspense, useEffect, useState, createElement } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { Layout, Menu, Button, Drawer, Tooltip, message } from 'antd'
import {
  DashboardOutlined,
  FileTextOutlined,
  ExperimentOutlined,
  PieChartOutlined,
  SyncOutlined,
  MenuOutlined,
  ApiOutlined,
  PlayCircleOutlined
} from '@ant-design/icons'
import { APP_NAME, APP_VERSION } from './config/app'
import { getBackendStatus, requestBackendControl, resolveBackendBaseUrl, type BackendControlResponse, type BackendStatus } from './services/backendControl'
import routes from './utils/router'
import './styles/global.css'

const { Header, Content, Footer } = Layout

function App() {
  const location = useLocation()
  const navigate = useNavigate()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768)
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking')

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768)
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const checkBackendStatus = async () => {
    setBackendStatus(current => (current === 'starting' ? current : 'checking'))
    const status = await getBackendStatus()
    setBackendStatus(status)
  }

  useEffect(() => {
    checkBackendStatus()
  }, [])

  const handleStartBackend = async () => {
    setBackendStatus('starting')
    try {
      const data = await requestBackendControl<BackendControlResponse>('start')
      if (data.running) {
        setBackendStatus('online')
        message.success(data.alreadyRunning ? '后端已经在运行' : '后端已启动')
        return
      }

      message.info(data.message || '后端正在启动，请稍后刷新状态')
      setTimeout(checkBackendStatus, 2500)
    } catch (error) {
      setBackendStatus('offline')
      message.error(error instanceof Error ? error.message : '启动后端失败')
    }
  }

  const handleRestartBackend = async () => {
    setBackendStatus('starting')
    try {
      const data = await requestBackendControl<BackendControlResponse>('restart')
      if (data.running) {
        setBackendStatus('online')
        message.success(data.message || '后端已重启并加载最新代码')
        return
      }

      message.info(data.message || '后端正在重启，请稍后刷新状态')
      setTimeout(checkBackendStatus, 2500)
    } catch (error) {
      setBackendStatus('offline')
      message.error(error instanceof Error ? error.message : '重启后端失败')
    }
  }

  const backendStatusMeta = {
    checking: {
      text: '检查后端',
      color: '#64748b',
      background: 'rgba(100, 116, 139, 0.1)',
      border: 'rgba(100, 116, 139, 0.25)',
      icon: <SyncOutlined spin />
    },
    online: {
      text: '重启后端',
      color: '#059669',
      background: 'rgba(16, 185, 129, 0.12)',
      border: 'rgba(16, 185, 129, 0.32)',
      icon: <ApiOutlined />
    },
    offline: {
      text: '启动后端',
      color: '#b45309',
      background: 'rgba(245, 158, 11, 0.14)',
      border: 'rgba(245, 158, 11, 0.36)',
      icon: <PlayCircleOutlined />
    },
    starting: {
      text: '启动中',
      color: '#2563eb',
      background: 'rgba(59, 130, 246, 0.12)',
      border: 'rgba(59, 130, 246, 0.32)',
      icon: <SyncOutlined spin />
    }
  } satisfies Record<BackendStatus, { text: string; color: string; background: string; border: string; icon: React.ReactNode }>

  const backendMeta = backendStatusMeta[backendStatus]

  // 根据路径获取当前选中的菜单
  const getSelectedKey = () => {
    const route = routes.find(r => r.path === location.pathname)
    return route ? route.key : 'home'
  }

  // 菜单项配置
  const menuItems = routes
    .filter(route => !route.hideInMenu)
    .map(route => ({
      key: route.key,
      icon: route.icon ? createElement(createIcon(route.icon)) : undefined,
      label: route.label
    }))

  const handleMenuClick = ({ key }: { key: string }) => {
    const route = routes.find(r => r.key === key)
    if (route) {
      setMobileMenuOpen(false)
      navigate(route.path)
    }
  }

  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f9ff' }}>
      {/* 顶部导航栏 */}
      <Header
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 1000,
          background: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(59, 130, 246, 0.15)',
          boxShadow: '0 2px 12px rgba(59, 130, 246, 0.08)',
          padding: '0 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            cursor: 'pointer',
            transition: 'all 0.3s ease'
          }}
          onClick={() => navigate('/')}
        >
          <div
            style={{
              width: '40px',
              height: '40px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#3b82f6',
              background: 'rgba(59, 130, 246, 0.1)',
              border: '1px solid rgba(59, 130, 246, 0.3)',
              borderRadius: '8px'
            }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: '24px', height: '24px' }}>
              {/* 外圈六边形 */}
              <path d="M12 2L20 7V17L12 22L4 17V7L12 2Z" strokeOpacity="0.6" />
              {/* 内部数据点 */}
              <circle cx="12" cy="8" r="1.5" fill="currentColor" />
              <circle cx="8" cy="16" r="1.5" fill="currentColor" />
              <circle cx="16" cy="16" r="1.5" fill="currentColor" />
              {/* 连接线形成趋势图 */}
              <path d="M8 16L12 8L16 16" />
              {/* 中心核心点 */}
              <circle cx="12" cy="12" r="2" fill="currentColor" opacity="0.9" />
              <circle cx="12" cy="12" r="3.5" strokeOpacity="0.4" />
            </svg>
          </div>
          <div>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              <div
                style={{
                  fontFamily: '"SF Mono", "Monaco", "Inconsolata", monospace',
                  fontSize: '18px',
                  fontWeight: 800,
                  letterSpacing: '2px',
                  color: '#1f2937',
                  lineHeight: '1.2'
                }}
              >
                {APP_NAME}
              </div>
              <span
                style={{
                  fontFamily: '"SF Mono", "Monaco", monospace',
                  fontSize: '11px',
                  fontWeight: 700,
                  color: '#2563eb',
                  background: 'rgba(37, 99, 235, 0.1)',
                  border: '1px solid rgba(37, 99, 235, 0.2)',
                  borderRadius: '999px',
                  padding: '2px 8px',
                  lineHeight: 1.4
                }}
              >
                {APP_VERSION}
              </span>
            </div>
            <div
              style={{
                fontSize: '10px',
                letterSpacing: '2px',
                color: '#9ca3af',
                lineHeight: '1.2',
                fontWeight: 600
              }}
            >
              智能量化分析平台
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1, justifyContent: 'flex-end', minWidth: 0 }}>
          {/* 桌面端菜单 */}
          {!isMobile && (
            <Menu
              mode="horizontal"
              selectedKeys={[getSelectedKey()]}
              items={menuItems}
              onClick={handleMenuClick}
              style={{
                flex: 1,
                justifyContent: 'flex-end',
                background: 'transparent',
                border: 'none',
                fontFamily: '"SF Mono", "Monaco", monospace',
                fontSize: '13px',
                fontWeight: 700,
                letterSpacing: '0.5px',
                minWidth: 0,
                overflow: 'hidden'
              }}
            />
          )}

          <Tooltip title={backendStatus === 'online' ? '重启后端并加载当前最新代码' : backendStatus === 'offline' ? '本地开发模式下可启动 FactorHub 后端' : '检查后端状态'}>
            <Button
              size="small"
              icon={backendMeta.icon}
              loading={backendStatus === 'starting'}
              onClick={backendStatus === 'online' ? handleRestartBackend : backendStatus === 'offline' ? handleStartBackend : checkBackendStatus}
              style={{
                color: backendMeta.color,
                background: backendMeta.background,
                borderColor: backendMeta.border,
                fontFamily: '"SF Mono", "Monaco", monospace',
                fontSize: '12px',
                fontWeight: 700,
                flexShrink: 0
              }}
            >
              {!isMobile ? backendMeta.text : null}
            </Button>
          </Tooltip>
        </div>

        {/* 移动端菜单按钮 */}
        {isMobile && (
          <Button
            type="text"
            icon={<MenuOutlined style={{ fontSize: '24px', color: '#9ca3af' }} />}
            onClick={() => setMobileMenuOpen(true)}
          />
        )}
      </Header>

      {/* 移动端抽屉菜单 */}
      <Drawer
        title={
          <span style={{ fontFamily: '"SF Mono", "Monaco", monospace', fontSize: '14px', fontWeight: 700, letterSpacing: '1.5px', color: '#64748b' }}>
            导航菜单
          </span>
        }
        placement="right"
        open={mobileMenuOpen}
        onClose={() => setMobileMenuOpen(false)}
        styles={{
          body: { background: '#ffffff' },
          header: { background: '#f0f9ff', borderBottom: '1px solid #bae6fd' }
        }}
      >
        <Menu
          mode="vertical"
          selectedKeys={[getSelectedKey()]}
          items={menuItems}
          onClick={handleMenuClick}
          style={{ background: 'transparent', fontWeight: 700 }}
        />
      </Drawer>

      {/* 主内容区 */}
      <Content style={{ padding: 0, background: 'transparent' }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
          <Suspense fallback={<div style={{ textAlign: 'center', padding: '100px 0', color: '#64748b' }}>加载中...</div>}>
            <Routes>
              {routes.map(route => (
                <Route key={route.path} path={route.path} element={<route.component />} />
              ))}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </div>
      </Content>

      {/* 页脚 */}
      <Footer
        style={{
          background: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(20px)',
          padding: '24px',
          borderTop: '1px solid rgba(59, 130, 246, 0.15)'
        }}
      >
        <div style={{ maxWidth: '1400px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <p style={{ fontFamily: '"SF Mono", "Monaco", monospace', fontSize: '13px', fontWeight: 600, letterSpacing: '1px', color: '#64748b', margin: '0 0 4px 0' }}>
              © 2025 {APP_NAME}
            </p>
            <p style={{ fontSize: '11px', letterSpacing: '1px', color: '#4b5563', margin: 0 }}>
              智能量化因子分析平台
            </p>
          </div>
          <div style={{ display: 'flex', gap: '24px' }}>
            <a href={`${resolveBackendBaseUrl()}/docs`} target="_blank" rel="noopener" style={{ textDecoration: 'none', color: '#64748b', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '8px', transition: 'all 0.3s ease' }}>
              <span style={{ width: '36px', height: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)', borderRadius: '4px', fontFamily: '"SF Mono", monospace', fontSize: '10px', fontWeight: 600 }}>
                接口
              </span>
              <span>API 文档</span>
            </a>
            <a href="https://github.com" target="_blank" rel="noopener" style={{ textDecoration: 'none', color: '#64748b', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '8px', transition: 'all 0.3s ease' }}>
              <span style={{ width: '36px', height: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)', borderRadius: '4px', fontFamily: '"SF Mono", monospace', fontSize: '10px', fontWeight: 600 }}>
                仓库
              </span>
              <span>GitHub</span>
            </a>
          </div>
        </div>
      </Footer>
    </Layout>
  )
}

// 辅助函数：根据图标名称创建图标组件
function createIcon(iconName: string) {
  const icons: Record<string, React.ComponentType> = {
    DashboardOutlined,
    FileTextOutlined,
    ExperimentOutlined,
    PieChartOutlined,
    SyncOutlined
  }
  return icons[iconName] || DashboardOutlined
}

export default function AppWrapper() {
  return (
    <BrowserRouter>
      <App />
    </BrowserRouter>
  )
}
