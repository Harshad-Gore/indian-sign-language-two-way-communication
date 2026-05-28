import type { WSEvent } from '@/types'

type WSEventHandler = (event: WSEvent) => void

let socket: WebSocket | null = null
let handlers: WSEventHandler[] = []
let reconnectTimer: ReturnType<typeof setTimeout> | null = null
let shouldReconnect = true

const WS_URL = (() => {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = window.location.host
  return `${proto}://${host}/realtime/translate`
})()

function connect() {
  if (socket && socket.readyState === WebSocket.OPEN) return

  socket = new WebSocket(WS_URL)

  socket.onopen = () => {
    console.log('[WS] Connected')
    // Keep alive
    const ping = setInterval(() => {
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'ping' }))
      } else {
        clearInterval(ping)
      }
    }, 25000)
  }

  socket.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data) as WSEvent
      handlers.forEach(h => h(event))
    } catch {
      console.warn('[WS] Bad message:', e.data)
    }
  }

  socket.onclose = () => {
    console.log('[WS] Disconnected')
    if (shouldReconnect) {
      reconnectTimer = setTimeout(connect, 3000)
    }
  }

  socket.onerror = (err) => {
    console.error('[WS] Error', err)
    socket?.close()
  }
}

export const wsClient = {
  connect,

  disconnect() {
    shouldReconnect = false
    if (reconnectTimer) clearTimeout(reconnectTimer)
    socket?.close()
    socket = null
  },

  send(data: object) {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(data))
    } else {
      console.warn('[WS] Not connected — message dropped')
    }
  },

  translate(text: string, isl_grammar = true) {
    this.send({ type: 'translate', text, isl_grammar })
  },

  subscribe(handler: WSEventHandler): () => void {
    handlers.push(handler)
    return () => {
      handlers = handlers.filter(h => h !== handler)
    }
  },

  get isConnected() {
    return socket?.readyState === WebSocket.OPEN
  },
}
