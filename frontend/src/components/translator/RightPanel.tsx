import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useTranslatorStore } from '@/stores/translatorStore'

export const RightPanel: React.FC = () => {
  const {
    originalText, simplifiedText, glossSequence,
    nlpBreakdown, animation, confidence, processingTimeMs,
    processingStage,
  } = useTranslatorStore()

  const posColors: Record<string, string> = {
    NN:'chip-cyan', NNS:'chip-cyan', NNP:'chip-cyan', NNPS:'chip-cyan',
    VB:'chip-primary', VBD:'chip-primary', VBG:'chip-primary', VBN:'chip-primary', VBP:'chip-primary', VBZ:'chip-primary',
    JJ:'chip-green', JJR:'chip-green', JJS:'chip-green',
    RB:'chip-orange', RBR:'chip-orange', RBS:'chip-orange',
    PRP:'chip-violet', 'PRP$':'chip-violet',
    WP:'chip-orange', WRB:'chip-orange', WDT:'chip-orange',
  }

  const confidencePct = Math.round(confidence * 100)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

      {/* Original sentence */}
      <Section label="Original Input">
        {originalText
          ? <p style={{ fontSize: '13px', color: 'var(--text-primary)', lineHeight: 1.6 }}>{originalText}</p>
          : <Placeholder />}
      </Section>

      {/* ISL Gloss */}
      <Section label="ISL Gloss Sequence">
        {glossSequence.length > 0 ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {glossSequence.map((g, i) => (
              <motion.span
                key={`${g}-${i}`}
                className="chip chip-primary"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.06 }}
                style={{ fontSize: '13px', padding: '4px 12px' }}
              >
                {g}
              </motion.span>
            ))}
          </div>
        ) : <Placeholder />}
      </Section>

      {/* Simplified ISL sentence */}
      <Section label="ISL Simplified">
        {simplifiedText
          ? <p style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--color-secondary)', letterSpacing: '0.08em' }}>{simplifiedText}</p>
          : <Placeholder />}
      </Section>

      {/* Confidence */}
      {confidence > 0 && (
        <Section label="Confidence Score">
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              flex: 1, height: '6px', borderRadius: '3px',
              background: 'rgba(99,102,241,0.2)', overflow: 'hidden',
            }}>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${confidencePct}%` }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
                style={{
                  height: '100%', borderRadius: '3px',
                  background: confidencePct > 70
                    ? 'linear-gradient(90deg, #10b981, #22d3ee)'
                    : confidencePct > 40
                    ? 'linear-gradient(90deg, #f59e0b, #10b981)'
                    : 'linear-gradient(90deg, #ef4444, #f59e0b)',
                }}
              />
            </div>
            <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', minWidth: '36px' }}>
              {confidencePct}%
            </span>
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
            {processingTimeMs > 0 && `Processed in ${processingTimeMs.toFixed(0)}ms`}
          </div>
        </Section>
      )}

      {/* NLP Token Breakdown */}
      {nlpBreakdown && (
        <Section label={`NLP Breakdown — ${nlpBreakdown.sentence_structure}`}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
            {nlpBreakdown.tokens.map((tok, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.03 }}
                style={{
                  display: 'flex', alignItems: 'center', gap: '6px',
                  padding: '4px 8px', borderRadius: '6px',
                  background: tok.kept_in_isl ? 'rgba(99,102,241,0.08)' : 'transparent',
                  opacity: tok.kept_in_isl ? 1 : 0.4,
                }}
              >
                <span style={{
                  fontSize: '12px',
                  color: tok.kept_in_isl ? 'var(--text-primary)' : 'var(--text-muted)',
                  fontWeight: tok.kept_in_isl ? 600 : 400,
                  minWidth: '80px',
                }}>
                  {tok.token}
                </span>
                <span className={`chip ${posColors[tok.pos] ?? 'chip-primary'}`} style={{ fontSize: '9px' }}>
                  {tok.pos}
                </span>
                {tok.kept_in_isl && (
                  <span style={{ fontSize: '10px', color: 'var(--color-success)', marginLeft: 'auto' }}>✓</span>
                )}
              </motion.div>
            ))}
          </div>
        </Section>
      )}

      {/* Animation info */}
      {animation && (
        <Section label="Animation Data">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            {[
              ['Frames', animation.total_frames],
              ['FPS', animation.fps],
              ['Duration', `${(animation.duration_ms / 1000).toFixed(1)}s`],
              ['Signs', animation.gloss_timeline.length],
            ].map(([label, val]) => (
              <div key={label as string} style={{ background: 'var(--bg-elevated)', borderRadius: '8px', padding: '8px' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{label}</div>
                <div style={{ fontSize: '16px', fontWeight: 700, color: 'var(--color-primary-light)', fontFamily: 'var(--font-mono)' }}>{val}</div>
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  )
}

const Section: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div className="glass" style={{ padding: '12px' }}>
    <div className="panel-label">{label}</div>
    {children}
  </div>
)

const Placeholder: React.FC = () => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
    <div className="skeleton" style={{ height: '12px', width: '80%' }} />
    <div className="skeleton" style={{ height: '12px', width: '60%' }} />
  </div>
)
