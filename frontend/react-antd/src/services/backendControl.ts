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

const resolveBackendControlBase = () => {
  if (typeof window === 'undefined') {
    return 'http://127.0.0.1:5173'
  }
  const host = window.location.hostname || '127.0.0.1'
  return `${window.location.protocol}//${host}:5173`
}

const resolveBackendHealthUrl = () => {
  if (typeof window === 'undefined') {
    return 'http://127.0.0.1:8001/health'
  }
  const host = window.location.hostname || '127.0.0.1'
  return `${window.location.protocol}//${host}:8001/health`
}

export async function requestBackendControl<T>(path: string, method: 'GET' | 'POST' = 'POST'): Promise<T> {
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

  try {
    const data = await requestBackendControl<{ running?: boolean }>('status', 'GET')
    return data.running ? 'online' : 'offline'
  } catch {
    return 'offline'
  }
}
