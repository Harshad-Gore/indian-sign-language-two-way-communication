import { useEffect, useRef, useState, useCallback } from 'react'

// ── Types ────────────────────────────────────────────────────────────────────

interface CWASAConfig {
  width?: number
  height?: number
  background?: string
  initAvatar?: string
}

interface QueueItem {
  id: string
  value: string
  kind: string
  asset: string
  state: 'pending' | 'playing' | 'played'
}

interface CWASAState {
  avatarReady: boolean
  currentSign: string | null
  framesText: string
  engineStatus: string
  queue: QueueItem[]
  isPlaying: boolean
  isPaused: boolean
  playbackSpeed: number
}

// ── Globals bridge ───────────────────────────────────────────────────────────
// CWASA uses globals: window.CWASA, window.tuavatarLoaded, etc.

declare global {
  interface Window {
    CWASA: {
      init: (cfg: unknown) => void
      playSiGMLURL: (url: string, avIdx?: number) => string
      playSiGMLText: (text: string, avIdx?: number) => string
      stopSiGML: (avIdx?: number) => string
      getLogger: (name: string, level: string) => void
    }
    tuavatarLoaded: boolean
    initCfg: unknown
    playerAvailableToPlay: boolean
  }
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useCWASA(containerRef: React.RefObject<HTMLDivElement | null>, config?: CWASAConfig) {
  const [state, setState] = useState<CWASAState>({
    avatarReady: false,
    currentSign: null,
    framesText: '0/0',
    engineStatus: 'Initializing avatar...',
    queue: [],
    isPlaying: false,
    isPaused: false,
    playbackSpeed: 1,
  })

  const queueRef = useRef<QueueItem[]>([])
  const currentItemIdRef = useRef<string | null>(null)
  const pollIntervalRef = useRef<number | null>(null)
  const initDoneRef = useRef(false)
  const avatarReadyRef = useRef(false)
  const pausedRef = useRef(false)
  const speedStepsRef = useRef(0)

  // ── Single bootstrap effect ─────────────────────────────────────────────────
  useEffect(() => {
    if (initDoneRef.current) return
    initDoneRef.current = true

    const container = containerRef.current
    if (!container) return

    // 1. Build the DOM structure CWASA expects — BEFORE loading the script
    // The bridge inputs MUST be on the document body level (CWASA scans document.getElementsByClassName)
    // not inside a shadow DOM or scoped container.
    if (!document.querySelector('.cwasa-bridge-global')) {
      const bridge = document.createElement('div')
      bridge.className = 'cwasa-bridge-global'
      bridge.setAttribute('aria-hidden', 'true')
      bridge.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden;pointer-events:none;'
      bridge.innerHTML = `
        <input class="txtSF av0" value="0/0" type="text" tabindex="-1" aria-hidden="true">
        <input class="txtGloss av0" value="[none]" type="text" tabindex="-1" aria-hidden="true">
        <input class="txtFPS av0" value="00.00" type="text" tabindex="-1" aria-hidden="true">
        <input class="txtLogSpeed av0" value="+0.0" type="text" tabindex="-1" aria-hidden="true">
        <input type="text" class="statusExtra av0" value="Initializing avatar" tabindex="-1" aria-hidden="true">
        <input type="text" id="URLText" class="txtSiGMLURL av0" value="" tabindex="-1" aria-hidden="true">
        <textarea class="txtaSiGMLText av0" style="display:none;" tabindex="-1" aria-hidden="true"></textarea>
        <button type="button" class="bttnSuspend av0" tabindex="-1" aria-hidden="true"></button>
        <button type="button" class="bttnResume av0" tabindex="-1" aria-hidden="true"></button>
        <button type="button" class="bttnPrevF av0" tabindex="-1" aria-hidden="true"></button>
        <button type="button" class="bttnNextF av0" tabindex="-1" aria-hidden="true"></button>
        <button type="button" class="bttnSpeedDown av0" tabindex="-1" aria-hidden="true"></button>
        <button type="button" class="bttnSpeedUp av0" tabindex="-1" aria-hidden="true"></button>
        <button type="button" class="bttnSpeedReset av0" tabindex="-1" aria-hidden="true"></button>
        <button type="button" class="bttnStop av0" tabindex="-1" aria-hidden="true"></button>
        <a id="player"></a>
      `
      document.body.appendChild(bridge)
    }

    // Create the avatar canvas container inside our React container
    if (!container.querySelector('.CWASAAvatar.av0')) {
      const avatarDiv = document.createElement('div')
      avatarDiv.className = 'CWASAAvatar av0'
      avatarDiv.style.cssText = 'width:100%;height:100%;'
      container.appendChild(avatarDiv)
    }

    // 2. Set up the global initCfg — this is what CWASA.init() reads.
    //    CRITICAL: jasBase MUST be a full absolute URL with protocol://host
    //    because CWASA's Data.splitURI() parses it with a regex that extracts
    //    scheme and authority. A relative path like "/jas/loc2021/" returns
    //    scheme=undefined, authority=undefined → "undefined://undefined/..."
    const origin = window.location.origin  // e.g. "http://localhost:5173"
    const jasBase = `${origin}/jas/loc2021/`

    window.initCfg = {
      avsbsl: ['francoise', 'anna', 'marc', 'luna', 'siggi'],
      avSettings: {
        width: config?.width || 760,
        height: config?.height || 560,
        background: config?.background || '#dbe4ee',
        avList: 'avsbsl',
        initAv: config?.initAvatar || 'francoise',
        initCamera: [0, 0.18, 3.02, 5, 17, 26, -1, -1],
        allowFrameSteps: true,
        allowSiGMLText: false,
        // This is the key fix: provide jasBase as a full absolute URL
        jasBase: jasBase,
      },
    }

    window.playerAvailableToPlay = true
    window.tuavatarLoaded = false

    // 3. Load CWASA CSS
    if (!document.querySelector('link[href="/css/cwasa.css"]')) {
      const link = document.createElement('link')
      link.rel = 'stylesheet'
      link.href = '/css/cwasa.css'
      document.head.appendChild(link)
    }

    // 4. Load the CWASA script, then init on load
    const initCWASA = () => {
      console.log('[CWASA] Script loaded, calling CWASA.init()')
      try {
        window.CWASA.init(window.initCfg)
      } catch (err) {
        console.error('[CWASA] Init error:', err)
        setState(prev => ({ ...prev, engineStatus: 'CWASA init failed' }))
      }
    }

    if (window.CWASA && typeof window.CWASA.init === 'function') {
      initCWASA()
      return
    }

    const existingScript = document.querySelector<HTMLScriptElement>('script[src="/js/allcsa.js"]')
    if (existingScript) {
      existingScript.addEventListener('load', initCWASA, { once: true })
      return
    }

    const script = document.createElement('script')
    script.src = '/js/allcsa.js'  // Use the unminified version (same as the original text_to_isl)
    script.onload = initCWASA
    script.onerror = () => {
      console.error('[CWASA] Script failed to load')
      setState(prev => ({ ...prev, engineStatus: 'CWASA script failed to load' }))
    }
    document.head.appendChild(script)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Playback polling cycle ──────────────────────────────────────────────────
  useEffect(() => {
    pollIntervalRef.current = window.setInterval(() => {
      // Sync telemetry from hidden bridge inputs (global DOM)
      const framesInput = document.querySelector<HTMLInputElement>('.txtSF.av0')
      const statusInput = document.querySelector<HTMLInputElement>('.statusExtra.av0')

      const framesText = framesInput?.value?.trim() || '0/0'
      const statusText = statusInput?.value?.trim() || ''

      // Check if avatar has loaded
      if (window.tuavatarLoaded && !avatarReadyRef.current) {
        avatarReadyRef.current = true
        console.log('[CWASA] Avatar ready!')
      }

      // Check if current item finished playing
      if (currentItemIdRef.current) {
        const statusLower = statusText.toLowerCase()
        const isError =
          statusLower.includes('invalid') ||
          statusLower.includes('error') ||
          statusLower.includes('not loaded') ||
          statusLower.includes('undefined avatar')
        const isFinished = window.playerAvailableToPlay

        if (isError || isFinished) {
          const idx = queueRef.current.findIndex(q => q.id === currentItemIdRef.current)
          if (idx !== -1) queueRef.current[idx].state = 'played'
          currentItemIdRef.current = null
          window.playerAvailableToPlay = true
        }
      }

      // Try to play next pending item
      if (avatarReadyRef.current && !pausedRef.current && !currentItemIdRef.current && window.playerAvailableToPlay) {
        const nextItem = queueRef.current.find(q => q.state === 'pending')
        if (nextItem) {
          nextItem.state = 'playing'
          currentItemIdRef.current = nextItem.id
          window.playerAvailableToPlay = false

          // Resolve asset URL to absolute
          const resolvedURL = new URL(nextItem.asset, window.location.href).toString()
          const urlInput = document.querySelector<HTMLInputElement>('#URLText')
          if (urlInput) urlInput.value = resolvedURL

          try {
            // CWASA's exported API is playSiGMLURL(url, avatarIndex).
            // Passing (0, url) loads no sign because CWASA treats the URL as the avatar id.
            const playResult = window.CWASA.playSiGMLURL(resolvedURL, 0)
            if (typeof playResult === 'string' && playResult.toLowerCase().includes('undefined avatar')) {
              throw new Error(playResult)
            }
          } catch (err) {
            console.error('[CWASA] Play error:', err)
            if (currentItemIdRef.current) {
              const idx = queueRef.current.findIndex(q => q.id === currentItemIdRef.current)
              if (idx !== -1) queueRef.current[idx].state = 'played'
              currentItemIdRef.current = null
              window.playerAvailableToPlay = true
            }
          }
        }
      }

      // Update React state
      const hasPending = queueRef.current.some(q => q.state === 'pending')
      const playingItem = queueRef.current.find(q => q.state === 'playing')
      const hasActivePlayback = !!currentItemIdRef.current || hasPending
      setState({
        avatarReady: avatarReadyRef.current,
        currentSign: playingItem?.value || null,
        framesText,
        engineStatus: statusText || (avatarReadyRef.current ? 'Ready' : 'Initializing avatar...'),
        queue: [...queueRef.current],
        isPlaying: hasActivePlayback,
        isPaused: pausedRef.current,
        playbackSpeed: Math.pow(2, speedStepsRef.current / 2),
      })
    }, 300)

    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
    }
  }, [])

  // ── Public API ──────────────────────────────────────────────────────────────

  const enqueueSequence = useCallback((sequence: Array<{ value: string; kind: string; asset: string }>) => {
    const batchId = `${Date.now()}-${Math.random().toString(16).slice(2)}`
    const newItems: QueueItem[] = sequence.map((item, index) => ({
      id: `${batchId}-${index}`,
      value: item.value,
      kind: item.kind,
      asset: item.asset,
      state: 'pending' as const,
    }))

    // Clear old played items, keep last 6
    const played = queueRef.current.filter(q => q.state === 'played').slice(-6)
    const pending = queueRef.current.filter(q => q.state !== 'played')
    queueRef.current = [...played, ...pending, ...newItems]
  }, [])

  const stopPlayback = useCallback(() => {
    try {
      if (window.CWASA && typeof window.CWASA.stopSiGML === 'function') {
        window.CWASA.stopSiGML(0)
      }
    } catch (err) {
      console.warn('[CWASA] Stop error:', err)
    }
    window.playerAvailableToPlay = true
    currentItemIdRef.current = null
    pausedRef.current = false
    queueRef.current = []
    setState(prev => ({ ...prev, currentSign: null, queue: [], isPlaying: false, isPaused: false }))
  }, [])

  const replaySequence = useCallback((sequence: Array<{ value: string; kind: string; asset: string }>) => {
    stopPlayback()
    setTimeout(() => enqueueSequence(sequence), 100)
  }, [stopPlayback, enqueueSequence])

  const clickCWASAControl = useCallback((control: string) => {
    const button = document.querySelector<HTMLButtonElement>(`.bttn${control}.av0`)
    button?.click()
  }, [])

  const pausePlayback = useCallback(() => {
    if (!currentItemIdRef.current) return
    clickCWASAControl('Suspend')
    pausedRef.current = true
    setState(prev => ({ ...prev, isPaused: true }))
  }, [clickCWASAControl])

  const resumePlayback = useCallback(() => {
    clickCWASAControl('Resume')
    pausedRef.current = false
    setState(prev => ({ ...prev, isPaused: false }))
  }, [clickCWASAControl])

  const stepFrame = useCallback((direction: 'previous' | 'next') => {
    if (!pausedRef.current && currentItemIdRef.current) {
      clickCWASAControl('Suspend')
      pausedRef.current = true
    }
    window.setTimeout(() => {
      clickCWASAControl(direction === 'previous' ? 'PrevF' : 'NextF')
      setState(prev => ({ ...prev, isPaused: pausedRef.current }))
    }, 60)
  }, [clickCWASAControl])

  const setPlaybackSpeed = useCallback((speed: number) => {
    const clamped = Math.min(4, Math.max(0.25, speed))
    const steps = Math.max(-4, Math.min(4, Math.round(Math.log2(clamped) * 2)))
    clickCWASAControl('SpeedReset')
    const control = steps > 0 ? 'SpeedUp' : 'SpeedDown'
    for (let i = 0; i < Math.abs(steps); i += 1) {
      clickCWASAControl(control)
    }
    speedStepsRef.current = steps
    setState(prev => ({ ...prev, playbackSpeed: Math.pow(2, steps / 2) }))
  }, [clickCWASAControl])

  return {
    ...state,
    enqueueSequence,
    stopPlayback,
    replaySequence,
    pausePlayback,
    resumePlayback,
    stepFrame,
    setPlaybackSpeed,
  }
}
