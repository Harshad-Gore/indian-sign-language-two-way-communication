import React, { Suspense, lazy, useEffect } from 'react'
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { Navbar } from '@/components/layout/Navbar'
import { useSettingsStore } from '@/stores/settingsStore'
import './index.css'

// Sync theme attribute to <html>
const ThemeSync = () => {
  const theme = useSettingsStore(s => s.theme)
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])
  return null
}

// Lazy-load pages for code splitting
const LandingPage    = lazy(() => import('@/pages/LandingPage'))
const TranslatorPage = lazy(() => import('@/pages/TranslatorPage'))
const HistoryPage    = lazy(() => import('@/pages/HistoryPage'))
const SettingsPage   = lazy(() => import('@/pages/SettingsPage'))
const AboutPage      = lazy(() => import('@/pages/AboutPage'))
const ResearchPage   = lazy(() => import('@/pages/ResearchPage'))
const SignRecognitionPage = lazy(() => import('@/pages/SignRecognitionPage'))

const PageLoader = () => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100vh - 60px)', flexDirection: 'column', gap: '16px' }}>
    <div style={{
      width: '48px', height: '48px', borderRadius: '50%',
      border: '3px solid rgba(99,102,241,0.2)',
      borderTop: '3px solid var(--color-primary)',
      animation: 'spin-slow 0.8s linear infinite',
    }} />
    <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Loading…</p>
  </div>
)

const AnimatedRoutes = () => {
  const location = useLocation()
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.22 }}
        style={{ height: '100%' }}
      >
        <Routes location={location}>
          <Route path="/"          element={<LandingPage />} />
          <Route path="/translate" element={<TranslatorPage />} />
          <Route path="/history"   element={<HistoryPage />} />
          <Route path="/settings"  element={<SettingsPage />} />
          <Route path="/about"     element={<AboutPage />} />
          <Route path="/research"  element={<ResearchPage />} />
          <Route path="/recognize" element={<SignRecognitionPage />} />
          <Route path="*"          element={<LandingPage />} />
        </Routes>
      </motion.div>
    </AnimatePresence>
  )
}

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean; error?: Error }> {
  constructor(props: { children: React.ReactNode }) {
    super(props)
    this.state = { hasError: false }
  }
  static getDerivedStateFromError(error: Error) { return { hasError: true, error } }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '40px', textAlign: 'center' }}>
          <h2 style={{ color: '#ef4444', marginBottom: '12px' }}>Something went wrong</h2>
          <pre style={{ color: 'var(--text-muted)', fontSize: '12px' }}>{this.state.error?.message}</pre>
          <button className="btn btn-primary" style={{ marginTop: '20px' }} onClick={() => window.location.reload()}>
            Reload
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <ErrorBoundary>
        <ThemeSync />
        <div style={{ position: 'relative', zIndex: 1 }}>
          <Navbar />
          <Suspense fallback={<PageLoader />}>
            <AnimatedRoutes />
          </Suspense>
        </div>
      </ErrorBoundary>
    </BrowserRouter>
  )
}

export default App
