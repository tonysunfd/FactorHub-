export const resolveApiUrl = (path: string) => {
  if (!path) return ''
  if (/^https?:\/\//i.test(path)) return path
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  if (typeof window === 'undefined') return normalizedPath
  if (normalizedPath.startsWith('/api/')) {
    return new URL(normalizedPath, window.location.origin).toString()
  }
  return new URL(`/api${normalizedPath}`, window.location.origin).toString()
}

