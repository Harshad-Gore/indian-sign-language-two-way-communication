import React from 'react'
import { motion } from 'framer-motion'
import { FlaskConical } from 'lucide-react'

const PIPELINE_STEPS = [
  { n: '01', title: 'Input Capture', desc: 'Text typed or voice recorded via Web Speech API / audio upload. Audio is sent as WAV/MP3 to the backend.', color: '#6366f1' },
  { n: '02', title: 'Speech Recognition', desc: 'OpenAI Whisper transcribes audio to text with word-level timestamps. Model: base (74M params).', color: '#818cf8' },
  { n: '03', title: 'NLP Preprocessing', desc: 'NLTK tokenisation → spaCy POS tagging → stopword/auxiliary verb removal → SOV reordering.', color: '#22d3ee' },
  { n: '04', title: 'Gloss Generation', desc: 'English tokens mapped to uppercase ISL gloss tokens. ISL grammar rules applied (Subject-Object-Verb order).', color: '#a78bfa' },
  { n: '05', title: 'Pose Lookup', desc: 'Each gloss token is looked up in the ISL pose library (71+ signs). Unknown words get procedural poses.', color: '#10b981' },
  { n: '06', title: 'Animation Synthesis', desc: 'Cubic Hermite spline interpolation between sign poses. Idle breathing + smoothstep easing.', color: '#f59e0b' },
  { n: '07', title: '3D Rendering', desc: 'React Three Fiber renders 75-point skeleton. Bloom + chromatic aberration post-processing.', color: '#ef4444' },
]

export const ResearchPage: React.FC = () => {
  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '32px 24px' }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div style={{ marginBottom: '40px' }}>
          <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '32px', fontWeight: 900, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <FlaskConical size={28} color="var(--color-primary)" strokeWidth={2} /> Research Methodology
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '15px', lineHeight: 1.8 }}>
            This platform implements a modular text-to-ISL translation pipeline based on computational linguistics
            and real-time skeletal animation.
          </p>
        </div>

        {/* Architecture diagram */}
        <div className="glass" style={{ padding: '24px', marginBottom: '32px' }}>
          <h2 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '16px' }}>System Architecture</h2>
          <pre style={{
            fontFamily: 'var(--font-mono)', fontSize: '12px',
            color: 'var(--color-secondary)', lineHeight: 1.8,
            overflow: 'auto',
          }}>{`
  ┌─────────────────────────────────────────┐
  │          React Frontend (Vite)          │
  │  Three.js ─ R3F ─ Bloom ─ Particles   │
  └──────────────┬──────────────────────────┘
                 │  REST + WebSocket
  ┌──────────────▼──────────────────────────┐
  │          FastAPI Backend                │
  │  ┌──────────┐  ┌────────────────────┐  │
  │  │  Whisper │  │    NLP Engine      │  │
  │  │  (STT)   │→ │ NLTK + spaCy + ISL │  │
  │  └──────────┘  └────────┬───────────┘  │
  │                         │              │
  │               ┌─────────▼──────────┐  │
  │               │  Sign Generator    │  │
  │               │  ISL Pose Library  │  │
  │               └─────────┬──────────┘  │
  │                         │              │
  │               ┌─────────▼──────────┐  │
  │               │  Animation Engine  │  │
  │               │  Cubic Hermite     │  │
  │               └────────────────────┘  │
  └─────────────────────────────────────────┘
`}</pre>
        </div>

        {/* Pipeline steps */}
        <h2 style={{ fontSize: '20px', fontWeight: 700, marginBottom: '20px' }}>Translation Pipeline</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '40px' }}>
          {PIPELINE_STEPS.map((step, i) => (
            <motion.div
              key={step.n}
              className="glass"
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.07 }}
              style={{ padding: '18px 20px', display: 'flex', gap: '20px', alignItems: 'flex-start' }}
            >
              <div style={{
                flexShrink: 0, width: '40px', height: '40px',
                borderRadius: '10px', background: `${step.color}22`,
                border: `1px solid ${step.color}55`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '11px', fontWeight: 800, fontFamily: 'var(--font-mono)',
                color: step.color,
              }}>
                {step.n}
              </div>
              <div>
                <h3 style={{ fontSize: '14px', fontWeight: 700, marginBottom: '4px' }}>{step.title}</h3>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>{step.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* ISL Grammar section */}
        <div className="glass" style={{ padding: '24px', marginBottom: '24px' }}>
          <h2 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '16px' }}>ISL Grammar Rules</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '13px', marginBottom: '14px', lineHeight: 1.7 }}>
            Indian Sign Language follows a Subject-Object-Verb (SOV) sentence structure, unlike English SVO.
            WH-questions are placed at the end of sentences in ISL.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            {[
              ['English (SVO)', '"What is your name?"', '#6366f1'],
              ['ISL Gloss (SOV)', '"YOUR NAME WHAT"', '#22d3ee'],
              ['English (SVO)', '"I am going to hospital"', '#6366f1'],
              ['ISL Gloss (SOV)', '"I HOSPITAL GO"', '#22d3ee'],
            ].map(([label, text, color], i) => (
              <div key={i} style={{ background: 'var(--bg-elevated)', borderRadius: '8px', padding: '12px' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '6px' }}>{label}</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', fontWeight: 600, color: color as string }}>{text}</div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default ResearchPage
