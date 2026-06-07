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

export async function requestBackendControl<T>(path: string): Promise<T> {
  const response = await fetch(`/__factorhub_backend/${path}`)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(data.message || `请求失败 (${response.status})`)
  }
  return data
}

export async function getBackendStatus(): Promise<BackendStatus> {
  try {
    const data = await requestBackendControl<{ running?: boolean }>('status')
    return data.running ? 'online' : 'offline'
  } catch {
    return 'offline'
  }
}

