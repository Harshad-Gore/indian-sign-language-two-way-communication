import React, { useRef, useEffect, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Camera,
  CameraOff,
  Volume2,
  VolumeX,
  RotateCcw,
  Hand,
  Activity,
  Type,
  Copy,
  Check,
  AlertCircle,
  Loader2,
  Zap,
} from 'lucide-react'
import { sendLandmarks, resetRecognizer, getRecognizerStatus, completeSentence, recognizeVision } from '@/api/recognizeApi'
import { FilesetResolver, HandLandmarker, PoseLandmarker, GestureRecognizer, DrawingUtils } from '@mediapipe/tasks-vision'

// ── Types ────────────────────────────────────────────────────────────────────

interface TopKResult {
  sign: string
  confidence: number
}

interface RecognitionState {
  currentSign: string | null
  confidence: number
  topK: TopKResult[]
  sentence: string[]
  frameCount: number
  bufferSize: number
}

// ── Styles ───────────────────────────────────────────────────────────────────

const pageStyle: React.CSSProperties = {
  maxWidth: '1200px',
  margin: '0 auto',
  padding: '24px',
  display: 'grid',
  gridTemplateColumns: '1fr 340px',
  gap: '20px',
  height: 'calc(100vh - 60px)',
}

const videoContainerStyle: React.CSSProperties = {
  position: 'relative',
  borderRadius: '16px',
  overflow: 'hidden',
  background: '#000',
  aspectRatio: '16/9',
  maxHeight: 'calc(100vh - 200px)',
}

const canvasOverlayStyle: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%',
  pointerEvents: 'none',
}

// ── Component ────────────────────────────────────────────────────────────────

export const SignRecognitionPage: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const animFrameRef = useRef<number>(0)
  const handLandmarkerRef = useRef<HandLandmarker | null>(null)
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null)
  const gestureRecognizerRef = useRef<GestureRecognizer | null>(null)
  const lastVideoTimeRef = useRef<number>(-1)
  const lastSendTimeRef = useRef<number>(0)
  const lastVisionScanRef = useRef<number>(0)
  const staticFrameCountRef = useRef<number>(0)
  const lastWristPosRef = useRef<{x: number, y: number} | null>(null)

  const [cameraActive, setCameraActive] = useState(false)
  const [modelsLoading, setModelsLoading] = useState(true)
  const [isCompleting, setIsCompleting] = useState(false)
  const [isVisionScanning, setIsVisionScanning] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [ttsEnabled, setTtsEnabled] = useState(true)
  const [copied, setCopied] = useState(false)
  const [modelStatus, setModelStatus] = useState<{ loaded: boolean; numClasses: number } | null>(null)

  const [recognition, setRecognition] = useState<RecognitionState>({
    currentSign: null,
    confidence: 0,
    topK: [],
    sentence: [],
    frameCount: 0,
    bufferSize: 0,
  })

  // Track last spoken sign for TTS dedup
  const lastSpokenRef = useRef<string | null>(null)
  // Track last committed sign for dedup
  const lastCommittedRef = useRef<string | null>(null)
  const holdStartRef = useRef<number>(0)
  const holdSignRef = useRef<string | null>(null)

  const HOLD_THRESHOLD_MS = 800  // Must hold sign for 800ms to commit

  // ── Check model status on mount ──────────────────────────────────────────
  useEffect(() => {
    getRecognizerStatus()
      .then(s => setModelStatus({ loaded: s.loaded, numClasses: s.num_classes }))
      .catch(() => setModelStatus({ loaded: false, numClasses: 0 }))

    const initModels = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        )
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2,
        })
        poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
        })
        gestureRecognizerRef.current = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2,
        })
        setModelsLoading(false)
      } catch (err) {
        console.error("Failed to load MediaPipe models:", err)
      }
    }
    initModels()
  }, [])

  // ── Start camera ─────────────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    try {
      setCameraError(null)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'user' },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setCameraActive(true)

      // Reset recognizer on backend
      await resetRecognizer().catch(() => {})

    } catch (err: any) {
      setCameraError(err.message || 'Camera access denied')
      setCameraActive(false)
    }
  }, [])

  // ── Stop camera ──────────────────────────────────────────────────────────
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    cancelAnimationFrame(animFrameRef.current)
    setCameraActive(false)
  }, [])

  // ── MediaPipe Inference & Render Loop ────────────────────────────────────
  useEffect(() => {
    if (!cameraActive) return
    let active = true

    const renderLoop = async () => {
      if (!active) return

      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas || video.readyState < 2) {
        animFrameRef.current = requestAnimationFrame(renderLoop)
        return
      }

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      }

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      ctx.save()
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      
      // Mirror canvas context because video is mirrored horizontally via CSS
      ctx.translate(canvas.width, 0)
      ctx.scale(-1, 1)

      const handLandmarker = handLandmarkerRef.current
      const poseLandmarker = poseLandmarkerRef.current
      const gestureRecognizer = gestureRecognizerRef.current

      if (handLandmarker && poseLandmarker && gestureRecognizer && video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime
        const startTimeMs = performance.now()

        const handResult = handLandmarker.detectForVideo(video, startTimeMs)
        const poseResult = poseLandmarker.detectForVideo(video, startTimeMs)
        const gestureResult = gestureRecognizer.recognizeForVideo(video, startTimeMs)

        const drawingUtils = new DrawingUtils(ctx)
        
        let currentWrist: { x: number; y: number } | null = null
        if (handResult.landmarks && handResult.landmarks.length > 0) {
          currentWrist = { x: handResult.landmarks[0][0].x, y: handResult.landmarks[0][0].y }
        }
        
        if (currentWrist) {
          if (lastWristPosRef.current) {
            const dx = currentWrist.x - lastWristPosRef.current.x
            const dy = currentWrist.y - lastWristPosRef.current.y
            const dist = Math.sqrt(dx*dx + dy*dy)
            if (dist < 0.015) {
              staticFrameCountRef.current += 1
            } else {
              staticFrameCountRef.current = 0
            }
          }
          lastWristPosRef.current = currentWrist
        } else {
          staticFrameCountRef.current = 0
          lastWristPosRef.current = null
        }

        if (staticFrameCountRef.current > 20 && Date.now() - lastVisionScanRef.current > 3000) {
          lastVisionScanRef.current = Date.now()
          staticFrameCountRef.current = 0
          
          const offCanvas = document.createElement('canvas')
          offCanvas.width = video.videoWidth
          offCanvas.height = video.videoHeight
          const offCtx = offCanvas.getContext('2d')
          if (offCtx) {
            offCtx.translate(offCanvas.width, 0)
            offCtx.scale(-1, 1)
            offCtx.drawImage(video, 0, 0, offCanvas.width, offCanvas.height)
            const b64 = offCanvas.toDataURL('image/jpeg', 0.6)
            
            setIsVisionScanning(true)
            recognizeVision(b64).then(res => {
              if (!active) return
              if (res.sign && res.sign !== lastCommittedRef.current) {
                setRecognition(prev => ({
                  ...prev,
                  sentence: [...prev.sentence, res.sign!],
                }))
                lastCommittedRef.current = res.sign
                if (ttsEnabled && res.sign !== lastSpokenRef.current) {
                  const utterance = new SpeechSynthesisUtterance(res.sign)
                  utterance.rate = 0.9
                  speechSynthesis.speak(utterance)
                  lastSpokenRef.current = res.sign
                }
              }
            }).catch(() => {}).finally(() => {
              if (active) setIsVisionScanning(false)
            })
          }
        }

        // Draw Pose (Thinner, elegant violet lines)
        if (poseResult.landmarks && poseResult.landmarks.length > 0) {
          const pose = poseResult.landmarks[0]
          drawingUtils.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, {
            color: 'rgba(167, 139, 250, 0.4)',
            lineWidth: 1.5
          })
          drawingUtils.drawLandmarks(pose, {
            color: 'rgba(167, 139, 250, 0.9)',
            lineWidth: 0,
            radius: 2
          })
        }

        // Draw Hands (Thinner, glowing cyan lines)
        if (handResult.landmarks && handResult.landmarks.length > 0) {
          for (const landmarks of handResult.landmarks) {
            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
              color: 'rgba(34, 211, 238, 0.6)',
              lineWidth: 1.5
            })
            drawingUtils.drawLandmarks(landmarks, {
              color: '#22d3ee',
              lineWidth: 0,
              radius: 2.5
            })
          }
        }

        // Throttle API sends to ~15fps (approx every 66ms)
        const now = Date.now()
        if (now - lastSendTimeRef.current > 66) {
          lastSendTimeRef.current = now

          let rightHand: number[][] | null = null
          let leftHand: number[][] | null = null

          if (handResult.landmarks && handResult.handednesses) {
            for (let i = 0; i < handResult.landmarks.length; i++) {
              const h = handResult.handednesses[i][0].categoryName
              const coords = handResult.landmarks[i].map(l => [l.x, l.y, l.z])
              if (h === 'Right') leftHand = coords
              if (h === 'Left') rightHand = coords
            }
          }

          let poseCoords: number[][] | null = null
          if (poseResult.landmarks && poseResult.landmarks.length > 0) {
            poseCoords = poseResult.landmarks[0].map(l => [l.x, l.y, l.z, l.visibility || 0])
          }

          sendLandmarks(rightHand, leftHand, poseCoords)
            .then(result => {
              if (!active) return

              // Look for robust static gestures if dynamic model failed to find one
              let activeSign = result.sign
              let isStatic = false
              if (!activeSign && gestureResult.gestures && gestureResult.gestures.length > 0) {
                const bestGesture = gestureResult.gestures[0][0]
                if (bestGesture && bestGesture.categoryName !== 'None' && bestGesture.score > 0.6) {
                   activeSign = bestGesture.categoryName.replace('_', ' ')
                   isStatic = true
                }
              }

              if (activeSign) {
                if (activeSign === holdSignRef.current) {
                  const elapsed = Date.now() - holdStartRef.current
                  const threshold = isStatic ? 500 : HOLD_THRESHOLD_MS
                  if (elapsed >= threshold && activeSign !== lastCommittedRef.current) {
                    setRecognition(prev => ({
                      ...prev,
                      sentence: [...prev.sentence, activeSign!],
                    }))
                    lastCommittedRef.current = activeSign

                    if (ttsEnabled && activeSign !== lastSpokenRef.current) {
                      const utterance = new SpeechSynthesisUtterance(activeSign!)
                      utterance.rate = 0.9
                      utterance.pitch = 1.0
                      speechSynthesis.speak(utterance)
                      lastSpokenRef.current = activeSign
                    }
                  }
                } else {
                  holdSignRef.current = activeSign
                  holdStartRef.current = Date.now()
                }
              } else {
                holdSignRef.current = null
                holdStartRef.current = 0
              }

              setRecognition(prev => ({
                ...prev,
                currentSign: activeSign,
                confidence: isStatic ? 1.0 : result.confidence,
                topK: result.top_k,
                frameCount: result.frame_count,
                bufferSize: result.buffer_size,
              }))
            })
            .catch(() => {})
        }
      }

      ctx.restore()
      animFrameRef.current = requestAnimationFrame(renderLoop)
    }

    animFrameRef.current = requestAnimationFrame(renderLoop)
    return () => {
      active = false
      cancelAnimationFrame(animFrameRef.current)
    }
  }, [cameraActive, ttsEnabled])

  // ── Cleanup on unmount ───────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [stopCamera])

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handleReset = async () => {
    setRecognition({
      currentSign: null,
      confidence: 0,
      topK: [],
      sentence: [],
      frameCount: 0,
      bufferSize: 0,
    })
    lastCommittedRef.current = null
    lastSpokenRef.current = null
    holdSignRef.current = null
    await resetRecognizer().catch(() => {})
  }

  const handleCopy = () => {
    const text = recognition.sentence.join(' ')
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  const handleUndo = () => {
    setRecognition(prev => ({
      ...prev,
      sentence: prev.sentence.slice(0, -1),
    }))
    lastCommittedRef.current = null
  }

  const handleCompleteSentence = async () => {
    if (recognition.sentence.length === 0) return
    setIsCompleting(true)
    try {
      const res = await completeSentence(recognition.sentence)
      setRecognition(prev => ({
        ...prev,
        sentence: [res.sentence] // Replace raw words with completed sentence
      }))
      
      if (ttsEnabled) {
        const utterance = new SpeechSynthesisUtterance(res.sentence)
        utterance.rate = 0.9
        speechSynthesis.speak(utterance)
      }
    } catch (err) {
      console.error('Sentence completion failed', err)
    } finally {
      setIsCompleting(false)
    }
  }

  return (
    <div style={pageStyle}>
      {/* ── Left: Video Feed ──────────────────────────────────────────── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h1 style={{
            fontFamily: 'var(--font-display)',
            fontSize: '24px',
            fontWeight: 800,
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}>
            <Camera size={22} color="var(--color-primary)" />
            Sign Language Recognition
          </h1>
          <div style={{ display: 'flex', gap: '8px' }}>
            {modelStatus && (
              <div style={{
                padding: '4px 10px',
                borderRadius: '999px',
                fontSize: '11px',
                fontWeight: 600,
                background: modelStatus.loaded ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.12)',
                color: modelStatus.loaded ? '#10b981' : '#ef4444',
                border: `1px solid ${modelStatus.loaded ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`,
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}>
                <Activity size={12} />
                {modelStatus.loaded ? `${modelStatus.numClasses} signs` : 'Model offline'}
              </div>
            )}
          </div>
        </div>

        {/* Video container */}
        <div className="glass" style={{ padding: '0', overflow: 'hidden', borderRadius: '16px' }}>
          <div style={videoContainerStyle}>
            <video
              ref={videoRef}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                transform: 'scaleX(-1)', // mirror
                display: cameraActive ? 'block' : 'none',
              }}
              playsInline
              muted
            />
            <canvas ref={canvasRef} style={canvasOverlayStyle} />

            {/* Camera off state */}
            {!cameraActive && (
              <div style={{
                position: 'absolute', inset: 0,
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center',
                background: 'var(--bg-card)',
                gap: '16px',
              }}>
                <div style={{
                  width: '80px', height: '80px', borderRadius: '20px',
                  background: 'rgba(99,102,241,0.12)',
                  border: '1px solid rgba(99,102,241,0.3)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  {modelsLoading ? (
                    <Loader2 size={32} className="spin-icon" color="var(--color-primary)" strokeWidth={2} />
                  ) : (
                    <CameraOff size={32} color="var(--text-muted)" strokeWidth={1.5} />
                  )}
                </div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '14px', textAlign: 'center', maxWidth: '300px' }}>
                  {modelsLoading 
                    ? "Downloading AI models... This may take a moment."
                    : "Enable your camera to start recognizing Indian Sign Language gestures in real-time"}
                </p>
                {cameraError && (
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    color: '#ef4444', fontSize: '12px',
                    background: 'rgba(239,68,68,0.1)',
                    padding: '6px 12px', borderRadius: '8px',
                  }}>
                    <AlertCircle size={14} /> {cameraError}
                  </div>
                )}
              </div>
            )}

            {isVisionScanning && (
              <motion.div
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 1.5, ease: 'easeInOut' }}
                style={{
                  position: 'absolute', top: 0, left: 0, right: 0,
                  height: '3px',
                  background: 'linear-gradient(90deg, #6366f1, #38bdf8, #6366f1)',
                  transformOrigin: 'left',
                  borderRadius: '0 0 2px 2px',
                  boxShadow: '0 0 12px rgba(99,102,241,0.5)',
                }}
              />
            )}

            {/* Live detection overlay */}
            {cameraActive && recognition.currentSign && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                style={{
                  position: 'absolute', bottom: '16px', left: '16px', right: '16px',
                  background: 'rgba(0,0,0,0.75)',
                  backdropFilter: 'blur(12px)',
                  borderRadius: '12px',
                  padding: '12px 16px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}
              >
                <div>
                  <div style={{ fontSize: '20px', fontWeight: 800, color: '#fff' }}>
                    {recognition.currentSign}
                  </div>
                  <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)' }}>
                    Detected sign
                  </div>
                </div>
                <div style={{
                  width: '48px', height: '48px', borderRadius: '50%',
                  background: `conic-gradient(#10b981 ${recognition.confidence * 360}deg, rgba(255,255,255,0.1) 0deg)`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <div style={{
                    width: '40px', height: '40px', borderRadius: '50%',
                    background: 'rgba(0,0,0,0.8)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '12px', fontWeight: 700, color: '#10b981',
                  }}>
                    {Math.round(recognition.confidence * 100)}%
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', gap: '8px' }}>
          <motion.button
            className={`btn ${cameraActive ? 'btn-danger' : 'btn-primary'}`}
            onClick={cameraActive ? stopCamera : startCamera}
            disabled={modelsLoading}
            whileTap={modelsLoading ? {} : { scale: 0.97 }}
            style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
          >
            {cameraActive ? <><CameraOff size={16} /> Stop Camera</> : <><Camera size={16} /> Start Camera</>}
          </motion.button>
          <button
            className="btn btn-secondary"
            onClick={() => setTtsEnabled(!ttsEnabled)}
            title={ttsEnabled ? 'Mute voice output' : 'Enable voice output'}
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '44px' }}
          >
            {ttsEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
          </button>
          <button
            className="btn btn-secondary"
            onClick={handleReset}
            title="Reset recognition"
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '44px' }}
          >
            <RotateCcw size={16} />
          </button>
        </div>

        {/* Sentence output */}
        <div className="glass" style={{ padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span className="panel-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Type size={13} /> Recognized Sentence
            </span>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button
                className="btn btn-secondary btn-sm"
                onClick={handleUndo}
                disabled={recognition.sentence.length === 0}
                style={{ fontSize: '11px', padding: '3px 8px' }}
              >
                Undo
              </button>
              <button
                className="btn btn-secondary btn-sm"
                onClick={handleCopy}
                disabled={recognition.sentence.length === 0}
                style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', padding: '3px 8px' }}
              >
                {copied ? <Check size={12} /> : <Copy size={12} />}
                {copied ? 'Copied!' : 'Copy'}
              </button>
            </div>
          </div>
          <div style={{
            minHeight: '48px',
            background: 'var(--bg-elevated)',
            borderRadius: '8px',
            padding: '12px',
            display: 'flex',
            flexWrap: 'wrap',
            gap: '6px',
            alignItems: 'center',
          }}>
            {recognition.sentence.length === 0 ? (
              <span style={{ color: 'var(--text-muted)', fontSize: '13px', fontStyle: 'italic' }}>
                Signs will appear here as you perform them…
              </span>
            ) : (
              recognition.sentence.map((word, i) => (
                <motion.span
                  key={`${word}-${i}`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="chip chip-primary"
                  style={{ fontSize: '13px', fontWeight: 600, padding: '4px 10px' }}
                >
                  {word}
                </motion.span>
              ))
            )}
          </div>
          
          <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'flex-end' }}>
             <button
                className="btn btn-primary"
                onClick={handleCompleteSentence}
                disabled={recognition.sentence.length === 0 || isCompleting}
                style={{ 
                  display: 'flex', alignItems: 'center', gap: '6px', 
                  fontSize: '13px', padding: '6px 14px',
                  background: 'linear-gradient(135deg, #10b981, #059669)',
                  color: 'white'
                }}
              >
                {isCompleting ? <Loader2 size={14} className="spin-icon" /> : <Zap size={14} />}
                Complete with AI
              </button>
          </div>
        </div>
      </div>

      {/* ── Right: Detection Panel ────────────────────────────────────── */}
      <aside style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        {/* Current detection */}
        <div className="glass" style={{ padding: '20px', textAlign: 'center' }}>
          <div className="panel-label" style={{ marginBottom: '12px' }}>Current Detection</div>
          <AnimatePresence mode="wait">
            <motion.div
              key={recognition.currentSign || 'none'}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.15 }}
            >
              {recognition.currentSign ? (
                <div>
                  <div style={{
                    width: '72px', height: '72px', borderRadius: '18px',
                    background: 'linear-gradient(135deg, var(--color-primary), var(--color-secondary))',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    margin: '0 auto 12px',
                    boxShadow: '0 8px 32px rgba(99,102,241,0.3)',
                  }}>
                    <Hand size={32} color="#fff" strokeWidth={2} />
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: 800, fontFamily: 'var(--font-display)' }}>
                    {recognition.currentSign}
                  </div>
                  <div style={{
                    fontSize: '14px', fontWeight: 700,
                    color: recognition.confidence > 0.8 ? '#10b981' : recognition.confidence > 0.6 ? '#f59e0b' : '#ef4444',
                    marginTop: '4px',
                  }}>
                    {Math.round(recognition.confidence * 100)}% confidence
                  </div>
                </div>
              ) : (
                <div>
                  <div style={{
                    width: '72px', height: '72px', borderRadius: '18px',
                    background: 'rgba(99,102,241,0.08)',
                    border: '2px dashed rgba(99,102,241,0.2)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    margin: '0 auto 12px',
                  }}>
                    <Hand size={28} color="var(--text-muted)" strokeWidth={1.5} />
                  </div>
                  <div style={{ fontSize: '14px', color: 'var(--text-muted)' }}>
                    {cameraActive ? 'Waiting for sign…' : 'Camera is off'}
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Top-K predictions */}
        <div className="glass" style={{ padding: '16px' }}>
          <div className="panel-label" style={{ marginBottom: '10px' }}>Top Predictions</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {recognition.topK.length > 0 ? (
              recognition.topK.map((pred, i) => (
                <div key={pred.sign} style={{
                  display: 'flex', alignItems: 'center', gap: '10px',
                  padding: '6px 8px', borderRadius: '6px',
                  background: i === 0 ? 'rgba(99,102,241,0.08)' : 'transparent',
                }}>
                  <div style={{
                    width: '22px', height: '22px', borderRadius: '6px',
                    background: i === 0 ? 'var(--color-primary)' : 'var(--bg-elevated)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '10px', fontWeight: 700,
                    color: i === 0 ? '#fff' : 'var(--text-muted)',
                  }}>
                    {i + 1}
                  </div>
                  <span style={{ flex: 1, fontSize: '13px', fontWeight: i === 0 ? 600 : 400 }}>{pred.sign}</span>
                  <span style={{
                    fontSize: '12px', fontFamily: 'var(--font-mono)', fontWeight: 600,
                    color: pred.confidence > 0.5 ? '#10b981' : 'var(--text-muted)',
                  }}>
                    {Math.round(pred.confidence * 100)}%
                  </span>
                </div>
              ))
            ) : (
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', textAlign: 'center', padding: '12px' }}>
                No predictions yet
              </div>
            )}
          </div>
        </div>

        {/* Stats */}
        <div className="glass" style={{ padding: '16px' }}>
          <div className="panel-label" style={{ marginBottom: '10px' }}>Session Stats</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            {[
              { label: 'Frames', value: recognition.frameCount.toString() },
              { label: 'Buffer', value: `${recognition.bufferSize}/30` },
              { label: 'Signs', value: recognition.sentence.length.toString() },
              { label: 'TTS', value: ttsEnabled ? 'On' : 'Off' },
            ].map(stat => (
              <div key={stat.label} style={{
                background: 'var(--bg-elevated)', borderRadius: '8px', padding: '10px',
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '18px', fontWeight: 800, fontFamily: 'var(--font-display)', color: 'var(--color-primary-light)' }}>
                  {stat.value}
                </div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>{stat.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Instructions */}
        <div className="glass" style={{ padding: '16px' }}>
          <div className="panel-label" style={{ marginBottom: '8px' }}>How to Use</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {[
              'Click "Start Camera" to begin',
              'Position your hands clearly in frame',
              'Perform ISL signs steadily',
              'Hold each sign for ~1 second',
              'Signs are committed to the sentence',
            ].map((step, i) => (
              <div key={i} style={{ display: 'flex', gap: '8px', alignItems: 'flex-start', fontSize: '12px', color: 'var(--text-secondary)' }}>
                <span style={{
                  width: '18px', height: '18px', borderRadius: '50%',
                  background: 'rgba(99,102,241,0.12)', flexShrink: 0,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '10px', fontWeight: 700, color: 'var(--color-primary)',
                }}>
                  {i + 1}
                </span>
                {step}
              </div>
            ))}
          </div>
        </div>
      </aside>
    </div>
  )
}

export default SignRecognitionPage
