import { lazy } from 'react'

// 懒加载页面组件
const Home = lazy(() => import('@/pages/Home'))
const FactorManagement = lazy(() => import('@/pages/FactorManagement'))
const FactorDetail = lazy(() => import('@/pages/FactorDetail'))
const FactorMining = lazy(() => import('@/pages/FactorMining'))
const PortfolioAnalysis = lazy(() => import('@/pages/PortfolioAnalysis'))
const Backtesting = lazy(() => import('@/pages/Backtesting'))
const PaperTrading = lazy(() => import('@/pages/PaperTrading'))
const WQBrain = lazy(() => import('@/pages/WQBrain'))

// 路由配置
export const routes = [
  {
    path: '/',
    key: 'home',
    label: '首页',
    icon: 'DashboardOutlined',
    component: Home
  },
  {
    path: '/factor-management',
    key: 'factor-management',
    label: '因子管理',
    icon: 'FileTextOutlined',
    component: FactorManagement
  },
  {
    path: '/factor-detail',
    key: 'factor-detail',
    label: '因子详情',
    component: FactorDetail,
    hideInMenu: true
  },
  {
    path: '/factor-mining',
    key: 'factor-mining',
    label: '因子挖掘',
    icon: 'ExperimentOutlined',
    component: FactorMining
  },
  {
    path: '/portfolio-analysis',
    key: 'portfolio-analysis',
    label: '组合分析',
    icon: 'PieChartOutlined',
    component: PortfolioAnalysis
  },
  {
    path: '/backtesting',
    key: 'backtesting',
    label: '策略回测',
    icon: 'SyncOutlined',
    component: Backtesting
  },
  {
    path: '/paper-trading',
    key: 'paper-trading',
    label: '模拟盘',
    icon: 'LineChartOutlined',
    component: PaperTrading
  },
  {
    path: '/wq-brain',
    key: 'wq-brain',
    label: 'WQ BRAIN',
    icon: 'DashboardOutlined',
    component: WQBrain
  }
]

export default routes
