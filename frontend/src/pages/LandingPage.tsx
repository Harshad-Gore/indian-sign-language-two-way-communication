import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Mic,
  Brain,
  Bone,
  Sparkles,
  Zap,
  BarChart3,
  Hand,
  FlaskConical,
  ArrowRight,
  Camera,
} from 'lucide-react'

const FEATURES = [
  {
    icon: Mic,
    title: 'Voice Recognition',
    desc: 'Browser speech input plus audio upload transcription for text-to-sign workflows.',
    color: '#6366f1',
  },
  {
    icon: Brain,
    title: 'ISL Grammar Engine',
    desc: 'NLP cleanup converts English text into ISL-friendly gloss and SiGML lookup tokens.',
    color: '#22d3ee',
  },
  {
    icon: Bone,
    title: 'CWASA Avatar Playback',
    desc: 'SiGML assets are queued and played on a signing avatar with replay and queue controls.',
    color: '#a78bfa',
  },
  {
    icon: Sparkles,
    title: 'Sign to Text + Speech',
    desc: 'MediaPipe landmarks feed PyTorch sequence models, then browser TTS speaks results.',
    color: '#10b981',
  },
  {
    icon: Zap,
    title: 'Benchmark Training',
    desc: 'Hybrid, Transformer, TCN, GRU, and lite models can be compared with real metrics.',
    color: '#f59e0b',
  },
  {
    icon: BarChart3,
    title: 'Report Metrics',
    desc: 'Training history, top-k accuracy, macro-F1, confusion matrix, and benchmark results.',
    color: '#ef4444',
  },
]
const STATS = [
  { label: 'SiGML Signs', value: '848' },
  { label: 'Recognition Classes', value: '71' },
  { label: 'Landmark Samples', value: '1,120' },
  { label: 'Baseline Val Acc', value: '96.7%' },
]
const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } },
}
const item = {
  hidden: { opacity: 0, y: 24 },
  show:   { opacity: 1, y: 0 },
}

export const LandingPage: React.FC = () => {
  return (
    <div style={{ overflowX: 'hidden' }}>

      {/* â”€â”€ Hero â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      <section style={{ minHeight: 'calc(100vh - 60px)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', textAlign: 'center', padding: '60px 24px', position: 'relative' }}>

        {/* Animated background rings */}
        {[1.2, 1.8, 2.4].map((scale, i) => (
          <motion.div
            key={i}
            style={{
              position: 'absolute',
              width: '400px', height: '400px',
              border: '1px solid rgba(99,102,241,0.15)',
              borderRadius: '50%',
              top: '50%', left: '50%',
              transform: 'translate(-50%, -50%)',
            }}
            animate={{ scale: [scale, scale + 0.3, scale], opacity: [0.4, 0.1, 0.4] }}
            transition={{ duration: 4 + i, repeat: Infinity, delay: i * 0.8 }}
          />
        ))}

        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: '8px',
            background: 'rgba(99,102,241,0.12)',
            border: '1px solid rgba(99,102,241,0.3)',
            borderRadius: '999px',
            padding: '6px 16px',
            fontSize: '12px', fontWeight: 600,
            color: 'var(--color-primary-light)',
            marginBottom: '24px',
          }}>
            <span style={{ animation: 'pulse-dot 1.5s infinite', width: '6px', height: '6px', background: '#10b981', borderRadius: '50%', display: 'inline-block' }} />
            AI-Powered · Real-time · Open-source
          </div>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          style={{
            fontFamily: 'var(--font-display)',
            fontSize: 'clamp(36px, 6vw, 72px)',
            fontWeight: 900,
            lineHeight: 1.1,
            marginBottom: '20px',
            maxWidth: '900px',
          }}
        >
          <span className="gradient-text">Voice, Text</span>
          <br />
          & Camera ISL Communication
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          style={{
            fontSize: '18px', color: 'var(--text-secondary)',
            maxWidth: '600px', lineHeight: 1.7, marginBottom: '40px',
          }}
        >
          A research-grade accessibility platform for text or voice to ISL animation, plus camera-based sign recognition with measurable model training reports.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.3 }}
          style={{ display: 'flex', gap: '14px', flexWrap: 'wrap', justifyContent: 'center' }}
        >
          <Link to="/translate" style={{ textDecoration: 'none' }}>
            <motion.button
              className="btn btn-primary btn-lg"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.97 }}
              style={{ fontSize: '16px', padding: '14px 32px', display: 'flex', alignItems: 'center', gap: '8px' }}
            >
              <Zap size={18} /> Text to Sign
            </motion.button>
          </Link>
          <Link to="/recognize" style={{ textDecoration: 'none' }}>
            <motion.button
              className="btn btn-secondary btn-lg"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.97 }}
              style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
            >
              <Camera size={18} /> Sign to Text
            </motion.button>
          </Link>
          <Link to="/research" style={{ textDecoration: 'none' }}>
            <motion.button
              className="btn btn-secondary btn-lg"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.97 }}
              style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
            >
              <FlaskConical size={18} /> Research
            </motion.button>
          </Link>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          style={{
            display: 'flex', gap: '32px', flexWrap: 'wrap', justifyContent: 'center',
            marginTop: '60px',
          }}
        >
          {STATS.map(stat => (
            <div key={stat.label} style={{ textAlign: 'center' }}>
              <div style={{
                fontSize: '32px', fontWeight: 900,
                fontFamily: 'var(--font-display)',
                background: 'linear-gradient(135deg, #818cf8, #22d3ee)',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              }}>
                {stat.value}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>{stat.label}</div>
            </div>
          ))}
        </motion.div>
      </section>

      {/* â”€â”€ Features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      <section style={{ padding: '80px 24px', maxWidth: '1100px', margin: '0 auto' }}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          style={{ textAlign: 'center', marginBottom: '48px' }}
        >
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '36px', fontWeight: 800 }}>
            <span className="gradient-text">Cutting-edge</span> capabilities
          </h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: '12px' }}>
            Built with state-of-the-art AI and rendering technology
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
            gap: '20px',
          }}
        >
          {FEATURES.map(f => {
            const Icon = f.icon
            return (
              <motion.div
                key={f.title}
                variants={item}
                className="glass glass-hover"
                style={{ padding: '24px' }}
              >
                <div style={{
                  width: '48px', height: '48px', borderRadius: '12px',
                  background: `${f.color}22`,
                  border: `1px solid ${f.color}44`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  marginBottom: '16px',
                }}>
                  <Icon size={22} color={f.color} strokeWidth={1.8} />
                </div>
                <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '8px' }}>{f.title}</h3>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>{f.desc}</p>
              </motion.div>
            )
          })}
        </motion.div>
      </section>

      {/* â”€â”€ CTA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      <section style={{ padding: '80px 24px', textAlign: 'center' }}>
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="glass"
          style={{
            maxWidth: '700px', margin: '0 auto', padding: '60px 40px',
            background: 'linear-gradient(135deg, rgba(99,102,241,0.12), rgba(34,211,238,0.08))',
            border: '1px solid rgba(99,102,241,0.3)',
          }}
        >
          <div style={{
            width: '72px', height: '72px', borderRadius: '18px',
            background: 'linear-gradient(135deg, var(--color-primary), var(--color-secondary))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 20px',
            boxShadow: '0 8px 32px rgba(99,102,241,0.3)',
          }}>
            <Hand size={32} color="#fff" strokeWidth={2} />
          </div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '30px', fontWeight: 800, marginBottom: '16px' }}>
            Break communication barriers
          </h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '28px', lineHeight: 1.7 }}>
            Making sign language accessible through AI â€” for the deaf community, educators, and researchers.
          </p>
          <Link to="/translate" style={{ textDecoration: 'none' }}>
            <button className="btn btn-primary btn-lg" style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
              Get Started Now <ArrowRight size={18} />
            </button>
          </Link>
        </motion.div>
      </section>
    </div>
  )
}

export default LandingPage


