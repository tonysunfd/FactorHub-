import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'
import { spawn, spawnSync } from 'child_process'

// https://vite.dev/config/
const projectRoot = path.resolve(__dirname, '../..')
const backendLogDir = path.join(projectRoot, 'logs')
const backendStdoutLog = path.join(backendLogDir, 'start_api.log')
const backendStderrLog = path.join(backendLogDir, 'start_api.err.log')
const backendPidFile = path.join(backendLogDir, 'start_api.pid')
const projectPython = process.platform === 'win32'
  ? path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
  : path.join(projectRoot, '.venv', 'bin', 'python')
const backendHost = process.env.FACTORHUB_DEV_BACKEND_HOST || '127.0.0.1'
const backendPort = Number(process.env.VITE_BACKEND_PORT ?? '8001')
const backendBaseUrl = `http://${backendHost}:${backendPort}`

const sendJson = (res: any, statusCode: number, payload: Record<string, unknown>) => {
  res.statusCode = statusCode
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  res.end(JSON.stringify(payload))
}

const readTail = (filePath: string, maxBytes = 12000) => {
  if (!fs.existsSync(filePath)) return ''
  const stat = fs.statSync(filePath)
  const start = Math.max(0, stat.size - maxBytes)
  const fd = fs.openSync(filePath, 'r')
  try {
    const buffer = Buffer.alloc(stat.size - start)
    fs.readSync(fd, buffer, 0, buffer.length, start)
    return buffer.toString('utf8').replace(/\0/g, '').trim()
  } finally {
    fs.closeSync(fd)
  }
}

const checkBackendHealth = async () => {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 2000)
  try {
    const response = await fetch(`${backendBaseUrl}/health`, {
      signal: controller.signal
    })
    return {
      running: response.ok,
      status: response.status,
      data: response.ok ? await response.json().catch(() => null) : null
    }
  } catch (error) {
    return {
      running: false,
      error: error instanceof Error ? error.message : String(error)
    }
  } finally {
    clearTimeout(timeout)
  }
}

const startBackend = () => {
  fs.mkdirSync(backendLogDir, { recursive: true })

  const out = fs.openSync(backendStdoutLog, 'a')
  const err = fs.openSync(backendStderrLog, 'a')
  const command = fs.existsSync(projectPython) ? projectPython : 'python'
  const child = spawn(command, ['start_api.py'], {
    cwd: projectRoot,
    detached: true,
    stdio: ['ignore', out, err],
    env: {
      ...process.env,
      FACTORHUB_BACKEND_PORT: String(backendPort),
      FACTORHUB_RELOAD: process.env.FACTORHUB_RELOAD || '1'
    }
  })

  child.unref()
  fs.writeFileSync(backendPidFile, `${child.pid}\n`, 'utf8')
  return child.pid
}

const waitForBackendHealth = async (attempts = 8, delayMs = 800) => {
  for (let index = 0; index < attempts; index += 1) {
    const health = await checkBackendHealth()
    if (health.running) {
      return health
    }
    await new Promise(resolve => setTimeout(resolve, delayMs))
  }
  return checkBackendHealth()
}

const collectRunningBackendPids = () => {
  const pids = new Set<number>()

  const addPid = (value: string) => {
    const pid = Number.parseInt(value.trim(), 10)
    if (Number.isInteger(pid) && pid > 0 && pid !== process.pid) {
      pids.add(pid)
    }
  }

  if (fs.existsSync(backendPidFile)) {
    addPid(fs.readFileSync(backendPidFile, 'utf8'))
  }

  if (process.platform !== 'win32') {
    const lsofResult = spawnSync('lsof', ['-ti', `tcp:${backendPort}`], { encoding: 'utf8' })
    if (lsofResult.status === 0 && lsofResult.stdout) {
      lsofResult.stdout
        .split(/\r?\n/)
        .filter(Boolean)
        .forEach(addPid)
    }
  }

  return Array.from(pids)
}

const stopBackend = () => {
  const candidatePids = collectRunningBackendPids()
  const stoppedPids: number[] = []

  for (const pid of candidatePids) {
    try {
      process.kill(pid, 'SIGTERM')
      stoppedPids.push(pid)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      if (!message.includes('ESRCH')) {
        throw error
      }
    }
  }

  if (fs.existsSync(backendPidFile)) {
    fs.rmSync(backendPidFile, { force: true })
  }

  return stoppedPids
}

const ensureSingleBackend = async () => {
  const stoppedPids = stopBackend()
  if (stoppedPids.length > 0) {
    await new Promise(resolve => setTimeout(resolve, 1200))
  }
  return stoppedPids
}

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'factorhub-backend-control',
      configureServer(server) {
        server.middlewares.use('/__factorhub_backend/status', async (_req, res) => {
          const health = await checkBackendHealth()
          sendJson(res, 200, health)
        })

        server.middlewares.use('/__factorhub_backend/start', async (req, res) => {
          if (req.method !== 'POST' && req.method !== 'GET') {
            sendJson(res, 405, { running: false, message: 'Method Not Allowed' })
            return
          }
          const before = await checkBackendHealth()
          if (before.running) {
            sendJson(res, 200, { running: true, alreadyRunning: true })
            return
          }

          try {
            const stoppedPids = await ensureSingleBackend()
            const pid = startBackend()
            const after = await waitForBackendHealth()
            sendJson(res, after.running ? 200 : 202, {
              ...after,
              pid,
              stoppedPids,
              message: after.running ? '后端已启动' : '后端启动超时，请查看日志',
              stderrTail: after.running ? undefined : readTail(backendStderrLog),
              stdoutTail: after.running ? undefined : readTail(backendStdoutLog)
            })
          } catch (error) {
            sendJson(res, 500, {
              running: false,
              message: error instanceof Error ? error.message : String(error),
              stderrTail: readTail(backendStderrLog)
            })
          }
        })

        server.middlewares.use('/__factorhub_backend/restart', async (req, res) => {
          if (req.method !== 'POST' && req.method !== 'GET') {
            sendJson(res, 405, { running: false, message: 'Method Not Allowed' })
            return
          }
          try {
            const stoppedPids = await ensureSingleBackend()
            const pid = startBackend()
            const after = await waitForBackendHealth()
            sendJson(res, after.running ? 200 : 202, {
              ...after,
              pid,
              stoppedPids,
              message: after.running ? '后端已重启并加载最新代码' : '后端重启超时，请查看日志',
              stderrTail: after.running ? undefined : readTail(backendStderrLog),
              stdoutTail: after.running ? undefined : readTail(backendStdoutLog)
            })
          } catch (error) {
            sendJson(res, 500, {
              running: false,
              message: error instanceof Error ? error.message : String(error),
              stderrTail: readTail(backendStderrLog)
            })
          }
        })

        server.middlewares.use('/__factorhub_backend/logs', (_req, res) => {
          sendJson(res, 200, {
            stdoutTail: readTail(backendStdoutLog),
            stderrTail: readTail(backendStderrLog)
          })
        })
      }
    }
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined

          if (id.includes('/echarts/')) {
            return 'vendor-echarts'
          }
          if (id.includes('/highlight.js/') || id.includes('/react-simple-code-editor/')) {
            return 'vendor-editor'
          }
          return undefined
        }
      }
    }
  },
  server: {
    port: 5173,
    host: '0.0.0.0',
    strictPort: true,
    proxy: {
      '/api': {
        target: backendBaseUrl,
        changeOrigin: true
      }
    }
  }
})
