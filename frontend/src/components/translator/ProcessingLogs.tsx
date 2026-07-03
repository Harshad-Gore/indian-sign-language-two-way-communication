import React from 'react'
import { motion } from 'framer-motion'
import { useTranslatorStore } from '@/stores/translatorStore'
import type { LogEntry } from '@/types'

interface ProcessingLogsProps {
  logs: LogEntry[]
}

const levelColors: Record<string, string> = {
  info:    '#818cf8',
  success: '#10b981',
  warning: '#f59e0b',
  error:   '#ef4444',
}

const levelIcons: Record<string, string> = {
  info:    'ℹ',
  success: '✓',
  warning: '⚠',
  error:   '✕',
}

export const ProcessingLogs: React.FC<ProcessingLogsProps> = ({ logs }) => {
  const { clearLogs } = useTranslatorStore()
  const bottomRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  return (
    <div className="glass" style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '10px 12px', borderBottom: '1px solid var(--border-subtle)',
      }}>
        <span className="panel-label" style={{ marginBottom: 0 }}>Processing Log</span>
        <button
          onClick={clearLogs}
          style={{
            background: 'none', border: 'none', color: 'var(--text-muted)',
            fontSize: '11px', cursor: 'pointer',
          }}
        >
          Clear
        </button>
      </div>

      <div
        className="scroll-area"
        style={{
          flex: 1, padding: '8px', minHeight: '120px', maxHeight: '220px',
          display: 'flex', flexDirection: 'column', gap: '3px',
        }}
      >
        {logs.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: '12px', padding: '8px', textAlign: 'center' }}>
            Waiting for input…
          </div>
        ) : (
          logs.slice(-60).map(log => (
            <motion.div
              key={log.id}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.15 }}
              style={{
                display: 'flex', alignItems: 'flex-start', gap: '6px',
                fontSize: '11px', fontFamily: 'var(--font-mono)',
                color: levelColors[log.level] ?? 'var(--text-secondary)',
              }}
            >
              <span style={{ flexShrink: 0, marginTop: '1px' }}>
                {levelIcons[log.level] ?? '·'}
              </span>
              <span style={{ wordBreak: 'break-word', lineHeight: '1.5' }}>{log.message}</span>
            </motion.div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
