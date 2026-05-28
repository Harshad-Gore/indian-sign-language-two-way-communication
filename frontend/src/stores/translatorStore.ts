import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import type { AnimationData, NLPBreakdown, LogEntry, LogLevel } from '@/types'

interface TranslatorState {
  // Input
  inputText: string
  language: string
  islGrammar: boolean

  // Processing
  isProcessing: boolean
  processingStage: 'idle' | 'transcribing' | 'nlp' | 'animation' | 'done' | 'error'
  processingTimeMs: number

  // Results
  originalText: string
  simplifiedText: string
  glossSequence: string[]
  nlpBreakdown: NLPBreakdown | null
  animation: AnimationData | null
  confidence: number

  // Logs
  logs: LogEntry[]

  // Animation playback
  isPlaying: boolean
  currentFrame: number
  playbackSpeed: number

  // Actions
  setInputText: (text: string) => void
  setLanguage: (lang: string) => void
  setIslGrammar: (v: boolean) => void
  setProcessing: (stage: TranslatorState['processingStage']) => void
  setNLPResult: (nlp: NLPBreakdown) => void
  setAnimation: (anim: AnimationData) => void
  setResult: (result: Partial<TranslatorState>) => void
  addLog: (level: LogLevel, message: string) => void
  clearLogs: () => void
  setPlaying: (v: boolean) => void
  setCurrentFrame: (f: number) => void
  setPlaybackSpeed: (s: number) => void
  reset: () => void
}

const initialState = {
  inputText: '',
  language: 'en',
  islGrammar: true,
  isProcessing: false,
  processingStage: 'idle' as const,
  processingTimeMs: 0,
  originalText: '',
  simplifiedText: '',
  glossSequence: [],
  nlpBreakdown: null,
  animation: null,
  confidence: 0,
  logs: [],
  isPlaying: false,
  currentFrame: 0,
  playbackSpeed: 1.0,
}

export const useTranslatorStore = create<TranslatorState>()(
  devtools(
    (set, get) => ({
      ...initialState,

      setInputText: (text) => set({ inputText: text }),
      setLanguage:  (lang) => set({ language: lang }),
      setIslGrammar: (v)   => set({ islGrammar: v }),

      setProcessing: (stage) => set({
        processingStage: stage,
        isProcessing: stage !== 'idle' && stage !== 'done' && stage !== 'error',
      }),

      setNLPResult: (nlp) => set({
        glossSequence: nlp.gloss_sequence,
        simplifiedText: nlp.simplified,
        nlpBreakdown: nlp,
      }),

      setAnimation: (anim) => set({
        animation: anim,
        isPlaying: true,
        currentFrame: 0,
      }),

      setResult: (result) => set((state) => ({ ...state, ...result })),

      addLog: (level, message) => set((state) => ({
        logs: [
          ...state.logs.slice(-199),  // keep last 200
          { id: crypto.randomUUID(), timestamp: Date.now(), level, message },
        ],
      })),

      clearLogs: () => set({ logs: [] }),

      setPlaying:       (v) => set({ isPlaying: v }),
      setCurrentFrame:  (f) => set({ currentFrame: f }),
      setPlaybackSpeed: (s) => set({ playbackSpeed: s }),

      reset: () => set(initialState),
    }),
    { name: 'translator-store' },
  ),
)
