import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface RecognitionResult {
  sign: string | null
  confidence: number
  top_k: { sign: string; confidence: number }[]
  frame_count: number
  buffer_size: number
}

interface RecognizerStatus {
  loaded: boolean
  num_classes: number
  class_names: string[]
  model_path: string
  window_size: number
  confidence_threshold: number
}

export async function sendLandmarks(
  rightHand: number[][] | null,
  leftHand: number[][] | null,
  pose: number[][] | null,
): Promise<RecognitionResult> {
  const { data } = await axios.post<RecognitionResult>(`${API_BASE}/api/recognize/frame`, {
    right_hand: rightHand,
    left_hand: leftHand,
    pose: pose,
  })
  return data
}

export async function resetRecognizer(): Promise<void> {
  await axios.post(`${API_BASE}/api/recognize/reset`)
}

export async function getRecognizerStatus(): Promise<RecognizerStatus> {
  const { data } = await axios.get<RecognizerStatus>(`${API_BASE}/api/recognize/status`)
  return data
}

export async function getSignClasses(): Promise<{ classes: string[]; count: number }> {
  const { data } = await axios.get<{ classes: string[]; count: number }>(`${API_BASE}/api/recognize/classes`)
  return data
}

export async function completeSentence(words: string[]): Promise<{ sentence: string }> {
  const { data } = await axios.post<{ sentence: string }>(`${API_BASE}/api/recognize/complete-sentence`, {
    words,
  })
  return data
}

export async function recognizeVision(imageBase64: string): Promise<RecognitionResult> {
  const { data } = await axios.post<RecognitionResult>(`${API_BASE}/api/recognize/vision`, {
    image_base64: imageBase64,
  })
  return data
}
