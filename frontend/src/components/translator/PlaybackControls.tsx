import React from 'react'
import { motion } from 'framer-motion'
import { useTranslatorStore } from '@/stores/translatorStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { Play, Pause, RotateCcw } from 'lucide-react'

export const PlaybackControls: React.FC = () => {
  const { isPlaying, setPlaying, currentFrame, animation, playbackSpeed, setPlaybackSpeed, reset } = useTranslatorStore()
  const totalFrames = animation?.total_frames ?? 0
  const progress = totalFrames > 0 ? (currentFrame / totalFrames) * 100 : 0

  const speedOptions = [0.25, 0.5, 1.0, 1.5, 2.0]

  return (
    <div style={{
      padding: '10px 16px',
      background: 'var(--bg-elevated)',
      borderTop: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      gap: '8px',
    }}>
      {/* Progress bar */}
      <div
        style={{
          width: '100%', height: '3px',
          background: 'rgba(99,102,241,0.2)',
          borderRadius: '2px', cursor: 'pointer',
        }}
        onClick={e => {
          const rect = e.currentTarget.getBoundingClientRect()
          const ratio = (e.clientX - rect.left) / rect.width
          // Seek not fully implemented, visual only
        }}
      >
        <motion.div
          style={{
            height: '100%',
            background: 'linear-gradient(90deg, var(--color-primary), var(--color-secondary))',
            borderRadius: '2px',
          }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.05 }}
        />
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {/* Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            className="btn btn-secondary btn-icon"
            style={{ width: '34px', height: '34px', fontSize: '14px' }}
            onClick={() => setPlaying(!isPlaying)}
            disabled={!animation}
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <button
            className="btn btn-secondary btn-sm"
            onClick={reset}
            disabled={!animation}
            style={{ fontSize: '12px', padding: '5px 10px' }}
          >
            <RotateCcw size={12} /> Reset
          </button>
        </div>

        {/* Frame counter */}
        <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          {currentFrame} / {totalFrames}
        </span>

        {/* Speed */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          {speedOptions.map(s => (
            <button
              key={s}
              onClick={() => setPlaybackSpeed(s)}
              style={{
                background: playbackSpeed === s ? 'var(--color-primary)' : 'rgba(99,102,241,0.12)',
                border: '1px solid',
                borderColor: playbackSpeed === s ? 'var(--color-primary)' : 'var(--border-subtle)',
                borderRadius: '4px',
                color: playbackSpeed === s ? '#fff' : 'var(--text-secondary)',
                fontSize: '10px',
                padding: '3px 6px',
                cursor: 'pointer',
                fontFamily: 'var(--font-mono)',
              }}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
