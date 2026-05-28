import React from 'react'
import { motion } from 'framer-motion'
import { Hand, Dna, Palette, Cog } from 'lucide-react'

const TEAM = [
  { name: 'ISL Research Initiative', role: 'Language & Linguistics', icon: Dna, color: '#6366f1' },
  { name: 'AI Animation Lab', role: 'Avatar & Rendering', icon: Palette, color: '#22d3ee' },
  { name: 'Accessibility Engineering', role: 'Backend & NLP', icon: Cog, color: '#a78bfa' },
]

const TECH_STACK = [
  { label: 'Frontend', items: ['React', 'TypeScript', 'Vite', 'Three.js', 'React Three Fiber', 'Framer Motion', 'Zustand'] },
  { label: 'Backend', items: ['FastAPI', 'Python', 'Whisper', 'NLTK', 'spaCy', 'MediaPipe'] },
  { label: 'AI/ML', items: ['OpenAI Whisper', 'PyTorch Hybrid Model', 'NLP Pipeline', 'MediaPipe Hands'] },
  { label: 'Rendering', items: ['WebGL', 'Post-processing', 'Bloom', 'Skeletal Animation', 'Particle Systems'] },
]

export const AboutPage: React.FC = () => {
  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '32px 24px' }}>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
          <div style={{
            width: '80px', height: '80px', borderRadius: '20px',
            background: 'linear-gradient(135deg, var(--color-primary), var(--color-secondary))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 16px',
            boxShadow: '0 8px 32px rgba(99,102,241,0.3)',
          }}>
            <Hand size={36} color="#fff" strokeWidth={2} />
          </div>
          <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '36px', fontWeight: 900, marginBottom: '16px' }}>
            About <span className="gradient-text">ISL Translate</span>
          </h1>
          <p style={{ fontSize: '16px', color: 'var(--text-secondary)', lineHeight: 1.8, maxWidth: '600px', margin: '0 auto' }}>
            An AI-powered research platform breaking communication barriers between the deaf community
            and the hearing world through real-time Indian Sign Language generation.
          </p>
        </div>

        <div className="glass" style={{ padding: '32px', marginBottom: '24px', borderLeft: '4px solid var(--color-primary)' }}>
          <h2 style={{ fontSize: '20px', fontWeight: 700, marginBottom: '12px' }}>Mission</h2>
          <p style={{ color: 'var(--text-secondary)', lineHeight: 1.8 }}>
            India has approximately <strong style={{ color: 'var(--text-primary)' }}>18 million</strong> deaf and hard-of-hearing individuals.
            Indian Sign Language is their primary mode of communication, yet access to ISL tools is severely limited.
            This platform combines cutting-edge AI with accessibility research to democratize communication.
          </p>
        </div>

        {/* Team */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '16px', marginBottom: '32px' }}>
          {TEAM.map(member => {
            const Icon = member.icon
            return (
              <div key={member.name} className="glass glass-hover" style={{ padding: '20px', display: 'flex', alignItems: 'center', gap: '14px' }}>
                <div style={{
                  width: '44px', height: '44px', borderRadius: '12px',
                  background: `${member.color}22`, border: `1px solid ${member.color}44`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                }}>
                  <Icon size={20} color={member.color} strokeWidth={1.8} />
                </div>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 600 }}>{member.name}</div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{member.role}</div>
                </div>
              </div>
            )
          })}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '16px', marginBottom: '32px' }}>
          {TECH_STACK.map(stack => (
            <div key={stack.label} className="glass" style={{ padding: '18px' }}>
              <div className="panel-label">{stack.label}</div>
              {stack.items.map(item => (
                <div key={item} style={{ fontSize: '12px', color: 'var(--text-secondary)', padding: '3px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                  {item}
                </div>
              ))}
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )
}

export default AboutPage
