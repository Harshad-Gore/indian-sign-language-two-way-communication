import React, { useState, useRef, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Send,
  Loader2,
  Square,
  RotateCcw,
  Mic,
  MicOff,
  Volume2,
  Type,
  Sparkles,
  Copy,
  Check,
  Zap,
  MessageSquareText,
  Hand,
  Play,
  Pause,
  Gauge,
  Wand2,
  ListChecks,
  Radio,
  StepBack,
  StepForward,
  Search,
  Languages,
  Captions,
  UploadCloud,
  FileAudio,
} from 'lucide-react'
import {
  getSigmlCatalog,
  translateAudioToSigml,
  translateLanguage,
  translateToSigml,
  type CatalogItem,
  type TranslationResult,
} from '@/api/sigmlApi'
import { useCWASA } from '@/hooks/useCWASA'
import styles from './TranslatorPage.module.css'

const QUICK_PHRASES = [
  { label: 'Hello', text: 'Hello, how are you?' },
  { label: 'Thank you', text: 'Thank you for your help' },
  { label: 'Please wait', text: 'Please wait a moment' },
  { label: 'My name', text: 'My name is' },
  { label: 'Good morning', text: 'Good morning, how are you?' },
  { label: 'I need help', text: 'I need help please' },
  { label: 'Sorry', text: 'I am sorry' },
  { label: 'Welcome', text: 'Welcome to our school' },
]

const SPEECH_LANGUAGES = [
  { label: 'English (India)', value: 'en-IN' },
  { label: 'Hindi', value: 'hi-IN' },
  { label: 'Bengali', value: 'bn-IN' },
  { label: 'Tamil', value: 'ta-IN' },
  { label: 'Telugu', value: 'te-IN' },
  { label: 'Marathi', value: 'mr-IN' },
]

const TARGET_LANGUAGES = [
  { label: 'Hindi', value: 'Hindi', speech: 'hi-IN' },
  { label: 'Bengali', value: 'Bengali', speech: 'bn-IN' },
  { label: 'Tamil', value: 'Tamil', speech: 'ta-IN' },
  { label: 'Telugu', value: 'Telugu', speech: 'te-IN' },
  { label: 'Marathi', value: 'Marathi', speech: 'mr-IN' },
  { label: 'Gujarati', value: 'Gujarati', speech: 'gu-IN' },
  { label: 'Kannada', value: 'Kannada', speech: 'kn-IN' },
  { label: 'Malayalam', value: 'Malayalam', speech: 'ml-IN' },
  { label: 'English', value: 'English', speech: 'en-IN' },
]

const SPEED_PRESETS = [0.5, 0.71, 1, 1.41, 2]

type SpeechRecognitionResultLike = {
  transcript: string
}

type SpeechRecognitionEventLike = {
  results: {
    [resultIndex: number]: {
      [alternativeIndex: number]: SpeechRecognitionResultLike
    }
  }
}

type SpeechRecognitionLike = {
  continuous: boolean
  interimResults: boolean
  lang: string
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
  onerror: (() => void) | null
  onend: (() => void) | null
  start: () => void
  stop: () => void
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike

type SpeechRecognitionWindow = Window & typeof globalThis & {
  SpeechRecognition?: SpeechRecognitionConstructor
  webkitSpeechRecognition?: SpeechRecognitionConstructor
}

const getApiErrorMessage = (err: unknown, fallback: string) => {
  const rawMessage = err instanceof Error ? err.message : fallback
  try {
    const parsed = JSON.parse(rawMessage) as { detail?: string }
    return parsed.detail || rawMessage
  } catch {
    return rawMessage
  }
}

export const TranslatorPage: React.FC = () => {
  const avatarContainerRef = useRef<HTMLDivElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)
  const cwasa = useCWASA(avatarContainerRef, {
    width: 760,
    height: 520,
    background: 'transparent',
    initAvatar: 'francoise',
  })

  const [inputText, setInputText] = useState('')
  const [isTranslating, setIsTranslating] = useState(false)
  const [lastResult, setLastResult] = useState<TranslationResult | null>(null)
  const [copied, setCopied] = useState(false)
  const [ttsEnabled, setTtsEnabled] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [speechLanguage, setSpeechLanguage] = useState('en-IN')
  const [targetLanguage, setTargetLanguage] = useState('Hindi')
  const [translatedText, setTranslatedText] = useState('')
  const [languageError, setLanguageError] = useState('')
  const [isLanguageTranslating, setIsLanguageTranslating] = useState(false)
  const [audioUploadStatus, setAudioUploadStatus] = useState('')
  const [audioError, setAudioError] = useState('')
  const [isAudioTranscribing, setIsAudioTranscribing] = useState(false)
  const [selectedAudioName, setSelectedAudioName] = useState('')
  const [catalog, setCatalog] = useState<CatalogItem[]>([])
  const [gestureSearch, setGestureSearch] = useState('')
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null)

  const queue = cwasa.queue
  const totalItems = queue.length
  const completedItems = queue.filter(item => item.state === 'played').length
  const playingIndex = queue.findIndex(item => item.state === 'playing')
  const activePosition = playingIndex >= 0 ? playingIndex + 1 : Math.min(completedItems + 1, totalItems)
  const progressPercent = totalItems > 0
    ? Math.min(100, ((completedItems + (cwasa.currentSign ? 0.45 : 0)) / totalItems) * 100)
    : 0
  const fingerspelledCount = lastResult?.sequence.filter(item => item.kind === 'fingerspell').length ?? 0
  const signedCount = lastResult?.sequence.filter(item => item.kind === 'sign').length ?? 0
  const selectedTargetLanguage = TARGET_LANGUAGES.find(lang => lang.value === targetLanguage) ?? TARGET_LANGUAGES[0]
  const whisperLanguage = speechLanguage.split('-')[0] || 'en'
  const filteredGestures = catalog
    .filter(item => {
      const query = gestureSearch.trim().toLowerCase()
      if (!query) return ['hello', 'thank', 'sorry', 'welcome', 'good', 'help', 'name', 'please'].some(
        hint => item.name.toLowerCase().includes(hint) || item.fileName.toLowerCase().includes(hint),
      )
      return item.name.toLowerCase().includes(query) || item.fileName.toLowerCase().includes(query)
    })
    .slice(0, 12)

  useEffect(() => {
    let cancelled = false
    getSigmlCatalog()
      .then(result => {
        if (!cancelled) setCatalog(result.items)
      })
      .catch(err => console.warn('Could not load sign catalog:', err))
    return () => {
      cancelled = true
    }
  }, [])

  const handleTranslate = useCallback(async (text?: string) => {
    const t = (text || inputText).trim()
    if (!t || isTranslating) return

    setIsTranslating(true)
    try {
      const result = await translateToSigml(t)
      setLastResult(result)
      if (result.sequence.length > 0) {
        cwasa.enqueueSequence(result.sequence)
      }
      if (ttsEnabled && result.gloss) {
        const utterance = new SpeechSynthesisUtterance(t)
        utterance.lang = speechLanguage
        utterance.rate = 0.9
        window.speechSynthesis.speak(utterance)
      }
    } catch (err) {
      console.error('Translation error:', err)
    } finally {
      setIsTranslating(false)
    }
  }, [inputText, isTranslating, cwasa, ttsEnabled, speechLanguage])

  const handleAudioUpload = useCallback(async (file?: File) => {
    if (!file || isAudioTranscribing) return

    setIsAudioTranscribing(true)
    setSelectedAudioName(file.name)
    setAudioUploadStatus('Transcribing audio and preparing avatar motion...')
    setAudioError('')

    try {
      const result = await translateAudioToSigml(file, whisperLanguage)
      const transcript = result.transcribedText || result.input

      setInputText(transcript)
      setLastResult(result)
      setTranslatedText('')
      setLanguageError('')

      if (result.sequence.length > 0) {
        cwasa.replaySequence(result.sequence)
      }

      const durationText = result.duration ? ` (${result.duration.toFixed(1)}s)` : ''
      setAudioUploadStatus(`Transcript${durationText}: ${transcript}`)
    } catch (err) {
      setAudioError(getApiErrorMessage(err, 'Audio transcription failed'))
      setAudioUploadStatus('')
    } finally {
      setIsAudioTranscribing(false)
      if (audioInputRef.current) {
        audioInputRef.current.value = ''
      }
    }
  }, [cwasa, isAudioTranscribing, whisperLanguage])

  const handleReplay = useCallback(() => {
    if (lastResult?.sequence) {
      cwasa.replaySequence(lastResult.sequence)
    }
  }, [lastResult, cwasa])

  const handleCopy = useCallback(() => {
    if (lastResult?.gloss) {
      navigator.clipboard.writeText(lastResult.gloss).then(() => {
        setCopied(true)
        window.setTimeout(() => setCopied(false), 2000)
      })
    }
  }, [lastResult])

  const toggleListening = useCallback(() => {
    if (isListening) {
      recognitionRef.current?.stop()
      setIsListening(false)
      return
    }

    const speechWindow = window as SpeechRecognitionWindow
    const SpeechRecognition = speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition
    if (!SpeechRecognition) return

    const recognition = new SpeechRecognition()
    recognition.continuous = false
    recognition.interimResults = false
    recognition.lang = speechLanguage

    recognition.onresult = event => {
      const transcript = event.results[0][0].transcript
      setInputText(transcript)
      setIsListening(false)
      window.setTimeout(() => handleTranslate(transcript), 100)
    }
    recognition.onerror = () => setIsListening(false)
    recognition.onend = () => setIsListening(false)

    recognitionRef.current = recognition
    recognition.start()
    setIsListening(true)
  }, [isListening, handleTranslate, speechLanguage])

  const handleLanguageTranslate = useCallback(async () => {
    const sourceText = (inputText || lastResult?.input || '').trim()
    if (!sourceText || isLanguageTranslating) return

    setIsLanguageTranslating(true)
    setLanguageError('')
    try {
      const result = await translateLanguage(sourceText, targetLanguage)
      setTranslatedText(result.translatedText)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Language translation failed'
      setLanguageError(message.includes('GROQ_API_KEY') ? 'Add GROQ_API_KEY in .env for multilingual translation.' : message)
    } finally {
      setIsLanguageTranslating(false)
    }
  }, [inputText, lastResult, targetLanguage, isLanguageTranslating])

  const speakText = useCallback((text: string, lang: string) => {
    if (!text.trim()) return
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = lang
    utterance.rate = 0.92
    window.speechSynthesis.speak(utterance)
  }, [])

  const handlePrimaryPlayback = useCallback(() => {
    if (cwasa.isPaused) {
      cwasa.resumePlayback()
      return
    }
    if (cwasa.isPlaying) {
      cwasa.pausePlayback()
      return
    }
    handleReplay()
  }, [cwasa, handleReplay])

  const handlePlayGesture = useCallback((item: CatalogItem) => {
    const fileName = item.fileName.endsWith('.sigml') ? item.fileName : `${item.fileName}.sigml`
    const value = item.name || fileName.replace(/\.sigml$/i, '')
    cwasa.replaySequence([
      {
        value,
        kind: 'sign',
        asset: `/SignFiles/${fileName}`,
      },
    ])
  }, [cwasa])

  return (
    <div className={styles.pageShell}>
      <section className={styles.pageHeader}>
        <div>
          <span className={styles.kicker}><Sparkles size={14} /> ISL Avatar Studio</span>
          <h1>Translate, preview and control every sign</h1>
          <p>
            A lighter workspace for voice-to-text, text-to-sign animation, gesture playback, speed control and accessible multilingual output.
          </p>
        </div>
        <div className={styles.headerStats}>
          <div>
            <strong>{catalog.length || '...'}</strong>
            <span>gesture library</span>
          </div>
          <div>
            <strong>{cwasa.avatarReady ? 'Ready' : 'Loading'}</strong>
            <span>avatar state</span>
          </div>
        </div>
      </section>

      <div className={styles.workspaceGrid}>
        <aside className={styles.controlStack}>
          <section className={`${styles.panel} ${styles.inputPanel}`}>
            <div className={styles.panelTitle}>
              <MessageSquareText size={18} />
              <div>
                <h2>Text to ISL</h2>
                <span>Type or speak English</span>
              </div>
            </div>

            <div className={styles.textareaWrap}>
              <textarea
                value={inputText}
                onChange={e => setInputText(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleTranslate()
                  }
                }}
                placeholder="Type a sentence to translate into ISL..."
                rows={5}
              />
              <div className={styles.inputActions}>
                <div className={styles.iconButtons}>
                  <select
                    className={styles.compactSelect}
                    value={speechLanguage}
                    onChange={e => setSpeechLanguage(e.target.value)}
                    aria-label="Speech recognition language"
                  >
                    {SPEECH_LANGUAGES.map(lang => (
                      <option key={lang.value} value={lang.value}>{lang.label}</option>
                    ))}
                  </select>
                  <button
                    className={`${styles.roundButton} ${isListening ? styles.dangerButton : ''}`}
                    onClick={toggleListening}
                    title={isListening ? 'Stop listening' : 'Speak to translate'}
                    type="button"
                  >
                    {isListening ? <MicOff size={16} /> : <Mic size={16} />}
                  </button>
                  <button
                    className={`${styles.roundButton} ${ttsEnabled ? styles.activeButton : ''}`}
                    onClick={() => setTtsEnabled(!ttsEnabled)}
                    title={ttsEnabled ? 'Disable voice output' : 'Enable voice output'}
                    type="button"
                  >
                    <Volume2 size={16} />
                  </button>
                </div>
                <motion.button
                  className={styles.primaryButton}
                  onClick={() => handleTranslate()}
                  disabled={!inputText.trim() || isTranslating}
                  whileTap={{ scale: 0.96 }}
                  type="button"
                >
                  {isTranslating ? <Loader2 size={16} className="spin-icon" /> : <Send size={16} />}
                  Translate
                </motion.button>
              </div>
            </div>

            <div className={styles.audioUploadBox}>
              <input
                ref={audioInputRef}
                className={styles.fileInput}
                type="file"
                accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm,.flac"
                onChange={e => handleAudioUpload(e.target.files?.[0])}
              />
              <button
                onClick={() => audioInputRef.current?.click()}
                disabled={isAudioTranscribing}
                type="button"
              >
                {isAudioTranscribing ? <Loader2 size={15} className="spin-icon" /> : <UploadCloud size={15} />}
                Upload audio
              </button>
              <span>
                <FileAudio size={14} />
                {selectedAudioName || `Whisper transcript to signs (${whisperLanguage})`}
              </span>
            </div>

            {(audioUploadStatus || audioError) && (
              <div className={audioError ? styles.audioError : styles.audioStatus}>
                {audioError || audioUploadStatus}
              </div>
            )}
          </section>

          <section className={styles.panel}>
            <div className={styles.sectionHeader}>
              <span><Wand2 size={14} /> Quick phrases</span>
            </div>
            <div className={styles.quickGrid}>
              {QUICK_PHRASES.map(phrase => (
                <motion.button
                  key={phrase.label}
                  onClick={() => {
                    setInputText(phrase.text)
                    handleTranslate(phrase.text)
                  }}
                  whileTap={{ scale: 0.95 }}
                  type="button"
                >
                  {phrase.label}
                </motion.button>
              ))}
            </div>
          </section>

          <section className={`${styles.panel} ${styles.accessPanel}`}>
            <div className={styles.sectionHeader}>
              <span><Languages size={14} /> Accessibility output</span>
            </div>
            <div className={styles.languageTools}>
              <select
                value={targetLanguage}
                onChange={e => setTargetLanguage(e.target.value)}
                aria-label="Translate output language"
              >
                {TARGET_LANGUAGES.map(lang => (
                  <option key={lang.value} value={lang.value}>{lang.label}</option>
                ))}
              </select>
              <button onClick={handleLanguageTranslate} disabled={isLanguageTranslating || !(inputText || lastResult?.input)} type="button">
                {isLanguageTranslating ? <Loader2 size={14} className="spin-icon" /> : <Captions size={14} />}
                Translate text
              </button>
            </div>
            {(translatedText || languageError) && (
              <div className={languageError ? styles.translationError : styles.translationBox}>
                <p>{languageError || translatedText}</p>
                {!languageError && (
                  <button onClick={() => speakText(translatedText, selectedTargetLanguage.speech)} type="button">
                    <Volume2 size={13} /> Speak {selectedTargetLanguage.label}
                  </button>
                )}
              </div>
            )}
          </section>

          <AnimatePresence>
            {lastResult && (
              <motion.section
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                className={styles.panel}
              >
                <div className={styles.sectionHeader}>
                  <span><Type size={14} /> ISL gloss</span>
                  <div className={styles.smallActions}>
                    <button onClick={handleCopy} type="button">
                      {copied ? <Check size={13} /> : <Copy size={13} />}
                      {copied ? 'Copied' : 'Copy'}
                    </button>
                    <button onClick={handleReplay} type="button">
                      <RotateCcw size={13} /> Replay
                    </button>
                  </div>
                </div>

                <div className={styles.glossBox}>
                  {lastResult.sequence.map((token, index) => (
                    <motion.span
                      key={`${token.value}-${index}`}
                      initial={{ opacity: 0, scale: 0.88 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.025 }}
                      className={token.kind === 'fingerspell' ? styles.fingerToken : styles.signToken}
                    >
                      {token.value}
                      {token.kind === 'fingerspell' && <em>ABC</em>}
                    </motion.span>
                  ))}
                </div>

                <div className={styles.miniMetrics}>
                  <span>{signedCount} direct signs</span>
                  <span>{fingerspelledCount} fingerspelled</span>
                  <span>{lastResult.gloss || 'No gloss'}</span>
                </div>
              </motion.section>
            )}
          </AnimatePresence>
        </aside>

        <main className={styles.stageColumn}>
          <section className={styles.avatarCard}>
            <div className={styles.stageTopbar}>
              <div className={styles.avatarStatus}>
                <span className={`${styles.statusDot} ${cwasa.avatarReady ? styles.readyDot : styles.loadingDot}`} />
                <div>
                  <strong>{cwasa.avatarReady ? 'Avatar Ready' : 'Loading Avatar'}</strong>
                  <span>{cwasa.engineStatus}</span>
                </div>
              </div>
              <div className={styles.frameBadge}>{cwasa.framesText}</div>
            </div>

            <div className={styles.avatarViewport}>
              <div ref={avatarContainerRef} className={styles.cwasaMount} />
              <AnimatePresence>
                {cwasa.currentSign && (
                  <motion.div
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 12 }}
                    className={styles.currentSignPill}
                  >
                    <Hand size={18} />
                    <span>{cwasa.currentSign}</span>
                  </motion.div>
                )}
              </AnimatePresence>
              {!cwasa.isPlaying && !cwasa.currentSign && cwasa.avatarReady && (
                <div className={styles.emptyHint}>Enter a sentence to start avatar signing</div>
              )}
            </div>
          </section>

          <section className={styles.motionDeck}>
            <div className={styles.deckHeader}>
              <div>
                <span><Radio size={14} /> Motion playback</span>
                <strong>{cwasa.currentSign ? `Signing ${cwasa.currentSign}` : totalItems ? 'Sequence queued' : 'Idle'}</strong>
              </div>
              <div className={styles.deckCount}>{totalItems ? `${activePosition}/${totalItems}` : '0/0'}</div>
            </div>

            <div className={styles.progressTrack} aria-label="Playback progress">
              <motion.div
                className={styles.progressFill}
                animate={{ width: `${progressPercent}%` }}
                transition={{ duration: 0.2 }}
              />
            </div>

            <div className={styles.transportPanel} aria-label="Avatar playback controller">
              <button
                className={styles.transportPrimary}
                onClick={handlePrimaryPlayback}
                disabled={!cwasa.isPlaying && !lastResult?.sequence.length}
                type="button"
              >
                {cwasa.isPaused ? <Play size={18} /> : cwasa.isPlaying ? <Pause size={18} /> : <Play size={18} />}
                {cwasa.isPaused ? 'Resume' : cwasa.isPlaying ? 'Pause' : 'Play'}
              </button>
              <button onClick={() => cwasa.stepFrame('previous')} disabled={!cwasa.currentSign} type="button">
                <StepBack size={15} /> Frame
              </button>
              <button onClick={() => cwasa.stepFrame('next')} disabled={!cwasa.currentSign} type="button">
                Frame <StepForward size={15} />
              </button>
              <button onClick={handleReplay} disabled={!lastResult?.sequence.length} type="button">
                <RotateCcw size={15} /> Replay
              </button>
              <button onClick={cwasa.stopPlayback} disabled={!cwasa.isPlaying && totalItems === 0} type="button">
                <Square size={15} /> Stop
              </button>
            </div>

            <div className={styles.speedPanel}>
              <span><Gauge size={13} /> Speed {cwasa.playbackSpeed.toFixed(2)}x</span>
              <div>
                {SPEED_PRESETS.map(speed => (
                  <button
                    key={speed}
                    className={Math.abs(cwasa.playbackSpeed - speed) < 0.03 ? styles.speedActive : ''}
                    onClick={() => cwasa.setPlaybackSpeed(speed)}
                    type="button"
                  >
                    {speed.toFixed(speed === 1 ? 0 : 2)}x
                  </button>
                ))}
              </div>
            </div>

            <div className={styles.gestureDock}>
              <div className={styles.gestureSearch}>
                <Search size={14} />
                <input
                  value={gestureSearch}
                  onChange={e => setGestureSearch(e.target.value)}
                  placeholder="Play a specific gesture..."
                />
              </div>
              <div className={styles.gestureResults}>
                {filteredGestures.length > 0 ? filteredGestures.map(item => (
                  <button key={`${item.sid}-${item.fileName}`} onClick={() => handlePlayGesture(item)} type="button">
                    {item.name}
                  </button>
                )) : (
                  <span>No matching gesture in the SiGML library</span>
                )}
              </div>
            </div>

            <div className={styles.queueStrip}>
              {queue.length > 0 ? queue.slice(-18).map(item => (
                <span key={item.id} className={styles[item.state]}>
                  {item.state === 'playing' && <Zap size={11} />}
                  {item.value}
                </span>
              )) : (
                <span className={styles.queuePlaceholder}><ListChecks size={13} /> Queue will appear here</span>
              )}
            </div>

            <div className={styles.deckMetrics}>
              <span><Gauge size={13} /> CWASA SiGML runtime</span>
              <span>{lastResult ? `${lastResult.tokenCount} assets` : 'No active sequence'}</span>
              <span>{cwasa.isPaused ? 'Paused for review' : cwasa.avatarReady ? 'Renderer online' : 'Renderer loading'}</span>
            </div>
          </section>
        </main>
      </div>
    </div>
  )
}

export default TranslatorPage

