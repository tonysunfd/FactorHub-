export type BackendStatus = 'checking' | 'online' | 'offline' | 'starting'

export type BackendControlResponse = {
  running?: boolean
  message?: string
  alreadyRunning?: boolean
  pid?: number
  stoppedPids?: number[]
  status?: number
  data?: Record<string, unknown> | null
  error?: string
}

const backendPort = String(import.meta.env.VITE_BACKEND_PORT || '8001')

const resolveBackendControlBase = () => {
  if (typeof window === 'undefined' || !import.meta.env.DEV) {
    return 'http://127.0.0.1:5173'
  }
  const host = window.location.hostname || '127.0.0.1'
  return `${window.location.protocol}//${host}:5173`
}

export const resolveBackendBaseUrl = () => {
  if (typeof window === 'undefined') {
    return `http://127.0.0.1:${backendPort}`
  }
  if (import.meta.env.DEV) {
    const host = window.location.hostname || '127.0.0.1'
    return `${window.location.protocol}//${host}:${backendPort}`
  }
  return window.location.origin
}

const resolveBackendHealthUrl = () => {
  return `${resolveBackendBaseUrl()}/health`
}

export async function requestBackendControl<T>(path: string, method: 'GET' | 'POST' = 'POST'): Promise<T> {
  if (!import.meta.env.DEV) {
    throw new Error('生产环境不支持通过前端控制后端进程')
  }
  const response = await fetch(`${resolveBackendControlBase()}/__factorhub_backend/${path}`, {
    method,
    cache: 'no-store'
  })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(data.message || `请求失败 (${response.status})`)
  }
  return data
}

export async function getBackendStatus(): Promise<BackendStatus> {
  try {
    const response = await fetch(resolveBackendHealthUrl(), {
      method: 'GET',
      cache: 'no-store'
    })
    if (response.ok) {
      return 'online'
    }
  } catch {
    // ignore and fall through to control-plane status
  }

  if (!import.meta.env.DEV) {
    return 'offline'
  }

  try {
    const data = await requestBackendControl<{ running?: boolean }>('status', 'GET')
    return data.running ? 'online' : 'offline'
  } catch {
    return 'offline'
  }
}
