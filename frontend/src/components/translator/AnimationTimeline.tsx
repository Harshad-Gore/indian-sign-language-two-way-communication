import React from 'react'
import { motion } from 'framer-motion'
import { useTranslatorStore } from '@/stores/translatorStore'

export const AnimationTimeline: React.FC = () => {
  const { animation, currentFrame } = useTranslatorStore()
  const timeline = animation?.gloss_timeline ?? []
  const total = animation?.total_frames ?? 1

  if (!timeline.length) return null

  return (
    <div style={{
      padding: '8px 16px',
      background: 'var(--bg-elevated)',
      borderTop: '1px solid var(--border-subtle)',
      overflowX: 'auto',
    }}>
      <div className="panel-label" style={{ marginBottom: '6px' }}>Sign Timeline</div>
      <div style={{ display: 'flex', height: '28px', gap: '2px', minWidth: 'max-content', position: 'relative' }}>
        {timeline.map((entry, i) => {
          const left  = (entry.start_frame / total) * 100
          const width = ((entry.end_frame - entry.start_frame) / total) * 100
          const isCurrent = currentFrame >= entry.start_frame && currentFrame <= entry.end_frame
          return (
            <motion.div
              key={i}
              animate={{ opacity: isCurrent ? 1 : 0.55, scale: isCurrent ? 1 : 0.97 }}
              style={{
                position: 'absolute',
                left: `${left}%`,
                width: `${Math.max(width, 3)}%`,
                height: '100%',
                background: isCurrent
                  ? 'linear-gradient(90deg, var(--color-primary), var(--color-secondary))'
                  : 'rgba(99,102,241,0.25)',
                borderRadius: '4px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '9px',
                fontWeight: 700,
                color: isCurrent ? '#fff' : 'var(--text-muted)',
                overflow: 'hidden',
                cursor: 'default',
                border: isCurrent ? '1px solid var(--color-secondary)' : '1px solid transparent',
                transition: 'all 0.15s',
              }}
              title={entry.gloss}
            >
              {width > 5 ? entry.gloss : ''}
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
