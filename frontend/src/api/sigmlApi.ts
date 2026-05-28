/**
 * API client for the SiGML translation backend.
 */

const API_BASE = '/api/sigml'

export interface SignToken {
  id: number
  value: string
  kind: 'sign' | 'fingerspell'
  asset: string
}

export interface TranslationResult {
  input: string
  gloss: string
  tokenCount: number
  sequence: SignToken[]
  transcribedText?: string
  detectedLanguage?: string
  duration?: number
}

export interface CatalogItem {
  fileName: string
  name: string
  sid: number
}

export interface LanguageTranslationResult {
  input: string
  translatedText: string
  sourceLanguage: string
  targetLanguage: string
}

/**
 * Translate English text to ISL sign sequence.
 */
export async function translateToSigml(text: string): Promise<TranslationResult> {
  const res = await fetch(`${API_BASE}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) throw new Error(`Translation failed: ${res.statusText}`)
  return res.json()
}

/**
 * Transcribe an uploaded audio clip and translate it to ISL sign sequence.
 */
export async function translateAudioToSigml(audio: File, language = 'en'): Promise<TranslationResult> {
  const form = new FormData()
  form.append('audio', audio)
  form.append('language', language)

  const res = await fetch(`${API_BASE}/voice`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const message = await res.text()
    throw new Error(message || `Audio translation failed: ${res.statusText}`)
  }
  return res.json()
}

/**
 * Get the full catalog of available sign files.
 */
export async function getSigmlCatalog(): Promise<{ count: number; items: CatalogItem[] }> {
  const res = await fetch(`${API_BASE}/catalog`)
  if (!res.ok) throw new Error(`Catalog fetch failed: ${res.statusText}`)
  return res.json()
}

/**
 * Translate text into another spoken/written language for accessibility.
 */
export async function translateLanguage(
  text: string,
  targetLanguage: string,
  sourceLanguage = 'auto',
): Promise<LanguageTranslationResult> {
  const res = await fetch(`${API_BASE}/translate-language`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      target_language: targetLanguage,
      source_language: sourceLanguage,
    }),
  })
  if (!res.ok) {
    const message = await res.text()
    throw new Error(message || `Language translation failed: ${res.statusText}`)
  }
  return res.json()
}
