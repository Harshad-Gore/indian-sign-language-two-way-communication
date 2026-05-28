import axios from 'axios'
import type { TranslationResult, TranslationHistoryItem, AnimationData, SystemStatus } from '@/types'

const api = axios.create({
  baseURL: '',   // uses Vite proxy
  timeout: 60000,
})

// ── Text Translation ──────────────────────────────────────────────────────────

export async function translateText(
  text: string,
  options: { language?: string; isl_grammar?: boolean } = {},
): Promise<TranslationResult> {
  const res = await api.post('/translate/text', {
    text,
    language: options.language ?? 'en',
    isl_grammar: options.isl_grammar ?? true,
  })
  return res.data
}

// ── Voice Translation ─────────────────────────────────────────────────────────

export async function translateVoice(
  audioBlob: Blob,
  options: { language?: string; isl_grammar?: boolean } = {},
): Promise<TranslationResult> {
  const form = new FormData()
  form.append('audio', audioBlob, 'recording.wav')
  form.append('language', options.language ?? 'en')
  form.append('isl_grammar', String(options.isl_grammar ?? true))
  const res = await api.post('/translate/voice', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

// ── History ───────────────────────────────────────────────────────────────────

export async function fetchHistory(limit = 20, offset = 0): Promise<TranslationHistoryItem[]> {
  const res = await api.get('/translate/history', { params: { limit, offset } })
  return res.data
}

export async function clearHistory(): Promise<void> {
  await api.delete('/translate/history')
}

// ── Animation ─────────────────────────────────────────────────────────────────

export async function generateAnimation(
  glossSequence: string[],
  fps = 30,
  interpolationFrames = 10,
): Promise<AnimationData> {
  const res = await api.post('/animation/generate', {
    gloss_sequence: glossSequence,
    fps,
    interpolation_frames: interpolationFrames,
  })
  return res.data
}

// ── System ────────────────────────────────────────────────────────────────────

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await api.get('/system/status')
  return res.data
}

export async function fetchAvailableSigns(): Promise<{ signs: string[]; count: number }> {
  const res = await api.get('/system/signs')
  return res.data
}
