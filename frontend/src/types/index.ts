// Central type definitions for the ISL Translation System

export interface LandmarkPoint {
  x: number
  y: number
  z: number
  visibility?: number
}

export interface PoseFrame {
  frame_index: number
  timestamp_ms: number
  body: LandmarkPoint[]
  left_hand: LandmarkPoint[]
  right_hand: LandmarkPoint[]
}

export interface GlossTimestamp {
  gloss: string
  start_frame: number
  end_frame: number
}

export interface AnimationData {
  fps: number
  total_frames: number
  duration_ms: number
  frames: PoseFrame[]
  gloss_timeline: GlossTimestamp[]
}

export interface TokenAnalysis {
  token: string
  pos: string
  dep: string
  lemma: string
  is_stopword: boolean
  kept_in_isl: boolean
}

export interface NLPBreakdown {
  original: string
  simplified: string
  gloss_sequence: string[]
  tokens: TokenAnalysis[]
  sentence_structure: string
}

export interface TranslationResult {
  id: string
  timestamp: string
  original_text: string
  transcribed_text?: string
  simplified_text: string
  gloss_sequence: string[]
  nlp_breakdown: NLPBreakdown
  animation: AnimationData
  confidence: number
  processing_time_ms: number
}

export interface TranslationHistoryItem {
  id: string
  timestamp: string
  original_text: string
  gloss_sequence: string[]
  confidence: number
}

export interface ModelStatus {
  name: string
  status: 'ready' | 'loading' | 'error' | 'unavailable' | 'not_loaded'
  version?: string
  message?: string
}

export interface SystemStatus {
  status: 'healthy' | 'degraded' | 'error'
  version: string
  uptime_seconds: number
  models: ModelStatus[]
}

// WebSocket event types
export type WSEvent =
  | { event: 'log';       data: { level: string; message: string } }
  | { event: 'nlp';       data: NLPBreakdown }
  | { event: 'animation'; data: AnimationData }
  | { event: 'done';      data: { processing_time_ms: number } }
  | { event: 'error';     data: { message: string } }
  | { event: 'pong';      data: Record<string, never> }

export type LogLevel = 'info' | 'success' | 'warning' | 'error'

export interface LogEntry {
  id: string
  timestamp: number
  level: LogLevel
  message: string
}
