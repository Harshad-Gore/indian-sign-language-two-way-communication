import React, { useState, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import { useTranslatorStore } from '@/stores/translatorStore'
import { Zap, Mic, Square, Loader2 } from 'lucide-react'

interface TextInputPanelProps {
  onTranslate: (text: string) => void
}

const EXAMPLE_PHRASES = [
  'Hello, how are you?',
  'What is your name?',
  'I need help',
  'Where is the hospital?',
  'Thank you very much',
  'I am happy today',
  'Please sit down',
  'Good morning',
]

const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'Hindi' },
]

export const TextInputPanel: React.FC<TextInputPanelProps> = ({ onTranslate }) => {
  const { inputText, setInputText, isProcessing, islGrammar, setIslGrammar, language, setLanguage } = useTranslatorStore()
  const [isRecording, setIsRecording] = useState(false)
  const recognitionRef = useRef<any>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = () => {
    if (inputText.trim()) onTranslate(inputText.trim())
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) handleSubmit()
  }

  // Web Speech API
  const toggleRecording = useCallback(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) {
      alert('Speech recognition not supported in this browser.')
      return
    }

    if (isRecording) {
      recognitionRef.current?.stop()
      setIsRecording(false)
      return
    }

    const recog = new SpeechRecognition()
    recog.continuous = false
    recog.interimResults = true
    recog.lang = language === 'hi' ? 'hi-IN' : 'en-US'
    recognitionRef.current = recog

    recog.onresult = (e: any) => {
      const transcript = Array.from(e.results)
        .map((r: any) => r[0].transcript)
        .join('')
      setInputText(transcript)
    }
    recog.onend = () => {
      setIsRecording(false)
      if (inputText.trim()) onTranslate(inputText.trim())
    }
    recog.onerror = () => setIsRecording(false)

    recog.start()
    setIsRecording(true)
  }, [isRecording, language, inputText, setInputText, onTranslate])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span className="panel-label">Input</span>
        <select
          value={language}
          onChange={e => setLanguage(e.target.value)}
          style={{
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '6px',
            color: 'var(--text-secondary)',
            fontSize: '12px',
            padding: '4px 8px',
          }}
        >
          {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.label}</option>)}
        </select>
      </div>

      {/* Textarea */}
      <div style={{ position: 'relative' }}>
        <textarea
          ref={textareaRef}
          className="input"
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a sentence to translate into ISL…&#10;&#10;Press Ctrl+Enter to translate"
          rows={5}
          style={{ resize: 'none', paddingRight: '40px' }}
        />
        <span style={{
          position: 'absolute', bottom: '10px', right: '12px',
          fontSize: '10px', color: 'var(--text-muted)',
        }}>
          {inputText.length}/2000
        </span>
      </div>

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: '8px' }}>
        <motion.button
          className="btn btn-primary"
          style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
          onClick={handleSubmit}
          disabled={!inputText.trim() || isProcessing}
          whileTap={{ scale: 0.97 }}
        >
          {isProcessing ? (
            <>
              <Loader2 size={16} className="spin-icon" /> Translating…
            </>
          ) : (
            <>
              <Zap size={16} /> Translate to ISL
            </>
          )}
        </motion.button>

        <motion.button
          className={`btn btn-icon ${isRecording ? 'btn-danger' : 'btn-secondary'}`}
          onClick={toggleRecording}
          title={isRecording ? 'Stop recording' : 'Start voice input'}
          whileTap={{ scale: 0.9 }}
          animate={isRecording ? { scale: [1, 1.1, 1] } : {}}
          transition={{ repeat: isRecording ? Infinity : 0, duration: 0.8 }}
          style={{
            background: isRecording ? 'rgba(239,68,68,0.2)' : undefined,
            boxShadow: isRecording ? '0 0 16px rgba(239,68,68,0.5)' : undefined,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          {isRecording ? <Square size={16} /> : <Mic size={16} />}
        </motion.button>
      </div>

      {/* ISL Grammar toggle */}
      <label style={{
        display: 'flex', alignItems: 'center', gap: '10px',
        cursor: 'pointer', padding: '8px 12px',
        background: 'var(--bg-elevated)', borderRadius: '8px',
        border: '1px solid var(--border-subtle)',
      }}>
        <div
          onClick={() => setIslGrammar(!islGrammar)}
          style={{
            width: '36px', height: '20px', borderRadius: '10px',
            background: islGrammar ? 'var(--color-primary)' : 'var(--text-muted)',
            position: 'relative', transition: 'background 0.2s', cursor: 'pointer',
          }}
        >
          <div style={{
            position: 'absolute', top: '3px', left: islGrammar ? '18px' : '3px',
            width: '14px', height: '14px', borderRadius: '50%',
            background: '#fff', transition: 'left 0.2s',
          }} />
        </div>
        <div>
          <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>
            ISL Grammar
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            {islGrammar ? 'SOV reordering enabled' : 'Raw token output'}
          </div>
        </div>
      </label>

      {/* Example phrases */}
      <div>
        <div className="panel-label">Quick Examples</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {EXAMPLE_PHRASES.map((phrase) => (
            <motion.button
              key={phrase}
              onClick={() => { setInputText(phrase); onTranslate(phrase) }}
              whileHover={{ x: 4 }}
              style={{
                background: 'transparent',
                border: '1px solid var(--border-subtle)',
                borderRadius: '6px',
                color: 'var(--text-secondary)',
                fontSize: '12px',
                padding: '6px 10px',
                textAlign: 'left',
                cursor: 'pointer',
                transition: 'all 0.15s',
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.borderColor = 'var(--color-primary)';
                (e.currentTarget as HTMLElement).style.color = 'var(--text-primary)'
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-subtle)';
                (e.currentTarget as HTMLElement).style.color = 'var(--text-secondary)'
              }}
            >
              {phrase}
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  )
}
