import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { fetchHistory, clearHistory } from '@/api/translateApi'
import type { TranslationHistoryItem } from '@/types'
import { ClipboardList, Trash2, RefreshCw } from 'lucide-react'

export const HistoryPage: React.FC = () => {
  const [items, setItems] = useState<TranslationHistoryItem[]>([])
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    fetchHistory(50).then(data => { setItems(data); setLoading(false) }).catch(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const handleClear = async () => {
    await clearHistory()
    setItems([])
  }

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '32px 24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
        <div>
          <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '28px', fontWeight: 800 }}>
            Translation History
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginTop: '4px' }}>
            {items.length} translation{items.length !== 1 ? 's' : ''} stored
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="btn btn-secondary btn-sm" onClick={load} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <RefreshCw size={14} /> Refresh
          </button>
          <button className="btn btn-danger btn-sm" onClick={handleClear} disabled={items.length === 0} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Trash2 size={14} /> Clear All
          </button>
        </div>
      </div>

      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {[...Array(5)].map((_, i) => (
            <div key={i} className="glass" style={{ padding: '16px', height: '72px' }}>
              <div className="skeleton" style={{ height: '14px', width: '60%', marginBottom: '8px' }} />
              <div className="skeleton" style={{ height: '12px', width: '40%' }} />
            </div>
          ))}
        </div>
      ) : items.length === 0 ? (
        <div className="glass" style={{ padding: '60px', textAlign: 'center' }}>
          <div style={{
            width: '64px', height: '64px', borderRadius: '16px',
            background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 16px',
          }}>
            <ClipboardList size={28} color="var(--color-primary)" strokeWidth={1.5} />
          </div>
          <p style={{ color: 'var(--text-secondary)' }}>No translations yet. Go to the Translator to get started.</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {items.map((item, i) => (
            <motion.div
              key={item.id}
              className="glass glass-hover"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
              style={{ padding: '14px 18px', display: 'flex', alignItems: 'center', gap: '16px' }}
            >
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '4px' }}>
                  {item.original_text}
                </div>
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  {item.gloss_sequence.map((g, j) => (
                    <span key={j} className="chip chip-primary" style={{ fontSize: '10px' }}>{g}</span>
                  ))}
                </div>
              </div>
              <div style={{ textAlign: 'right', flexShrink: 0 }}>
                <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--color-success)' }}>
                  {Math.round(item.confidence * 100)}%
                </div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  {new Date(item.timestamp).toLocaleDateString()}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  )
}

export default HistoryPage
