import React, { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { fetchSystemStatus } from '@/api/translateApi'
import { useSettingsStore } from '@/stores/settingsStore'
import type { SystemStatus } from '@/types'
import {
  Home,
  Languages,
  History,
  FlaskConical,
  Info,
  Settings,
  Sun,
  Moon,
  Hand,
  Camera,
} from 'lucide-react'

const NAV_LINKS = [
  { to: '/',           label: 'Home',        icon: Home },
  { to: '/translate',  label: 'Translator',  icon: Languages },
  { to: '/recognize',  label: 'Sign to Text', icon: Camera },
  { to: '/history',    label: 'History',     icon: History },
  { to: '/research',   label: 'Research',    icon: FlaskConical },
  { to: '/about',      label: 'About',       icon: Info },
  { to: '/settings',   label: 'Settings',    icon: Settings },
]

export const Navbar: React.FC = () => {
  const location = useLocation()
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [scrolled, setScrolled] = useState(false)
  const theme = useSettingsStore(s => s.theme)
  const setTheme = useSettingsStore(s => s.setTheme)

  useEffect(() => {
    fetchSystemStatus().then(setStatus).catch(() => null)
    const interval = setInterval(() => {
      fetchSystemStatus().then(setStatus).catch(() => null)
    }, 30000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 10)
    window.addEventListener('scroll', handler)
    return () => window.removeEventListener('scroll', handler)
  }, [])

  return (
    <nav style={{
      position: 'sticky',
      top: 0,
      zIndex: 100,
      height: '60px',
      display: 'flex',
      alignItems: 'center',
      padding: '0 24px',
      background: scrolled
        ? 'var(--bg-elevated)'
        : 'var(--bg-card)',
      backdropFilter: 'blur(20px)',
      borderBottom: '1px solid var(--border-subtle)',
      gap: '0',
      transition: 'background 0.3s',
    }}>
      {/* Logo */}
      <Link to="/" style={{ textDecoration: 'none', marginRight: '32px', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px', height: '32px',
            background: 'linear-gradient(135deg, var(--color-primary), var(--color-secondary))',
            borderRadius: '8px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: 'var(--glow-primary)',
          }}>
            <Hand size={16} color="#fff" strokeWidth={2.5} />
          </div>
          <span style={{
            fontFamily: 'var(--font-display)',
            fontSize: '16px',
            fontWeight: 800,
            background: 'linear-gradient(135deg, #818cf8, #22d3ee)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            ISL Translate
          </span>
        </div>
      </Link>

      {/* Nav links */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '2px', flex: 1 }}>
        {NAV_LINKS.map(link => {
          const active = location.pathname === link.to
          const Icon = link.icon
          return (
            <Link key={link.to} to={link.to} style={{ textDecoration: 'none' }}>
              <div style={{
                padding: '6px 14px',
                borderRadius: '8px',
                fontSize: '13px',
                fontWeight: active ? 600 : 400,
                color: active ? 'var(--color-primary-light)' : 'var(--text-secondary)',
                background: active ? 'rgba(99,102,241,0.15)' : 'transparent',
                border: active ? '1px solid rgba(99,102,241,0.3)' : '1px solid transparent',
                transition: 'all 0.15s',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
              onMouseEnter={e => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.background = 'rgba(99,102,241,0.08)';
                  (e.currentTarget as HTMLElement).style.color = 'var(--text-primary)'
                }
              }}
              onMouseLeave={e => {
                if (!active) {
                  (e.currentTarget as HTMLElement).style.background = 'transparent';
                  (e.currentTarget as HTMLElement).style.color = 'var(--text-secondary)'
                }
              }}
              >
                <Icon size={15} strokeWidth={active ? 2.2 : 1.8} />
                {link.label}
                {active && (
                  <motion.div
                    layoutId="nav-pill"
                    style={{ position: 'absolute', inset: 0, zIndex: -1 }}
                  />
                )}
              </div>
            </Link>
          )
        })}
      </div>

      {/* Status indicator + theme toggle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
        {status && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              className="status-dot"
              style={{
                background: status.status === 'healthy' ? '#10b981'
                  : status.status === 'degraded' ? '#f59e0b' : '#ef4444',
                boxShadow: `0 0 8px ${status.status === 'healthy' ? '#10b981' : '#f59e0b'}`,
              }}
            />
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {status.status === 'healthy' ? 'All systems online' : 'Degraded'}
            </span>
          </div>
        )}

        {/* Theme toggle */}
        <button
          id="theme-toggle"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          title="Toggle dark/light mode"
          style={{
            background: 'rgba(99,102,241,0.12)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '50%',
            width: '34px',
            height: '34px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {theme === 'dark' ? <Sun size={16} color="var(--text-primary)" /> : <Moon size={16} color="var(--text-primary)" />}
        </button>

        <div style={{
          padding: '4px 10px',
          background: 'rgba(99,102,241,0.12)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '999px',
          fontSize: '11px',
          color: 'var(--color-primary-light)',
        }}>
          v1.0
        </div>
      </div>
    </nav>
  )
}
