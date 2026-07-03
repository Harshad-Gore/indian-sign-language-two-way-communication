import React from 'react'
import { useSettingsStore } from '@/stores/settingsStore'
import type { WhisperModel } from '@/stores/settingsStore'
import { Mic, Film, Palette, RotateCcw } from 'lucide-react'

const ColorInput = ({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) => (
  <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
    <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{label}</span>
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <input
        type="color"
        value={value}
        onChange={event => onChange(event.target.value)}
        style={{ width: '36px', height: '28px', border: 'none', borderRadius: '6px', cursor: 'pointer', background: 'none', padding: 0 }}
      />
      <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{value}</span>
    </div>
  </label>
)

const Toggle = ({ label, description, value, onToggle }: { label: string; description?: string; value: boolean; onToggle: () => void }) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
    <div>
      <div style={{ fontSize: '13px', color: 'var(--text-primary)' }}>{label}</div>
      {description && <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{description}</div>}
    </div>
    <button
      type="button"
      aria-label={`${label}: ${value ? 'on' : 'off'}`}
      aria-pressed={value}
      onClick={onToggle}
      style={{
        width: '40px', height: '22px', borderRadius: '11px', border: 0, padding: 0,
        background: value ? 'var(--color-primary)' : 'rgba(71,85,105,0.5)',
        position: 'relative', cursor: 'pointer', transition: 'background 0.2s', flexShrink: 0,
      }}
    >
      <span style={{
        position: 'absolute', top: '3px', left: value ? '20px' : '3px',
        width: '16px', height: '16px', borderRadius: '50%',
        background: '#fff', transition: 'left 0.2s',
      }} />
    </button>
  </div>
)

const Section = ({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) => (
  <div className="glass" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
    <h3 style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
      {icon} {title}
    </h3>
    {children}
  </div>
)

export const SettingsPage: React.FC = () => {
  const s = useSettingsStore()

  return (
    <div style={{ maxWidth: '700px', margin: '0 auto', padding: '32px 24px' }}>
      <div style={{ marginBottom: '28px' }}>
        <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '28px', fontWeight: 800 }}>Settings</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginTop: '4px' }}>Customise the translator and avatar appearance</p>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

        <Section title="Speech Recognition" icon={<Mic size={16} color="var(--color-primary)" />}>
          <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Whisper Model</span>
            <select
              value={s.whisperModel}
              onChange={e => s.setWhisperModel(e.target.value as WhisperModel)}
              style={{
                background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)',
                borderRadius: '6px', color: 'var(--text-primary)', fontSize: '12px', padding: '5px 10px',
              }}
            >
              {['tiny','base','small','medium','large'].map(m => (
                <option key={m} value={m}>{m.charAt(0).toUpperCase() + m.slice(1)}</option>
              ))}
            </select>
          </label>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
            Note: Model is loaded on the backend. Larger = more accurate but slower.
          </div>
        </Section>

        <Section title="Animation" icon={<Film size={16} color="var(--color-primary)" />}>
          <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>Speed: {s.animationSpeed}×</span>
            <input
              type="range" min={0.25} max={3} step={0.25}
              value={s.animationSpeed}
              onChange={e => s.setAnimationSpeed(parseFloat(e.target.value))}
              style={{ width: '140px' }}
            />
          </label>
          <Toggle label="Idle Breathing Animation" description="Subtle torso movement when idle" value={s.idleAnimation} onToggle={s.toggleIdleAnimation} />
        </Section>

        <Section title="Avatar Appearance" icon={<Palette size={16} color="var(--color-primary)" />}>
          <Toggle label="Show Skeleton Rig" description="Neon skeleton used for signing animation" value={s.showSkeleton} onToggle={s.toggleSkeleton} />
          <ColorInput label="Avatar Body Color" value={s.avatarColor} onChange={s.setAvatarColor} />
          <ColorInput label="Joint Sphere Color" value={s.jointColor} onChange={s.setJointColor} />
          <ColorInput label="Bone Cylinder Color" value={s.boneColor} onChange={s.setBoneColor} />
          <Toggle label="Show Particle Effects" description="Firefly sparks on hands during signing" value={s.showParticles} onToggle={s.toggleParticles} />
          <Toggle label="Show Landmark Spheres" description="Joint visualization" value={s.showLandmarks} onToggle={s.toggleLandmarks} />
          <Toggle label="Show Floor Grid" description="Holographic ground grid" value={s.showGrid} onToggle={s.toggleGrid} />
        </Section>

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn btn-danger btn-sm" onClick={s.reset} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <RotateCcw size={14} /> Reset to Defaults
          </button>
        </div>
      </div>
    </div>
  )
}

export default SettingsPage
