import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface SettingsState {
  theme: 'dark' | 'light'
  whisperModel: 'tiny' | 'base' | 'small' | 'medium' | 'large'
  animationSpeed: number
  showSkeleton: boolean
  showLandmarks: boolean
  showParticles: boolean
  showGrid: boolean
  idleAnimation: boolean
  fpsLimit: number
  avatarColor: string
  jointColor: string
  boneColor: string
  cameraFov: number

  // Actions
  setTheme: (t: 'dark' | 'light') => void
  setWhisperModel: (m: SettingsState['whisperModel']) => void
  setAnimationSpeed: (s: number) => void
  toggleSkeleton: () => void
  toggleLandmarks: () => void
  toggleParticles: () => void
  toggleGrid: () => void
  toggleIdleAnimation: () => void
  setFpsLimit: (f: number) => void
  setAvatarColor: (c: string) => void
  setJointColor: (c: string) => void
  setBoneColor: (c: string) => void
  setCameraFov: (f: number) => void
  reset: () => void
}

const defaults = {
  theme: 'dark' as const,
  whisperModel: 'base' as const,
  animationSpeed: 1.0,
  showSkeleton: true,
  showLandmarks: true,
  showParticles: true,
  showGrid: true,
  idleAnimation: true,
  fpsLimit: 60,
  avatarColor: '#818cf8',
  jointColor: '#22d3ee',
  boneColor: '#6366f1',
  cameraFov: 60,
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      ...defaults,
      setTheme:          (t) => set({ theme: t }),
      setWhisperModel:   (m) => set({ whisperModel: m }),
      setAnimationSpeed: (s) => set({ animationSpeed: s }),
      toggleSkeleton:    () => set((s) => ({ showSkeleton: !s.showSkeleton })),
      toggleLandmarks:   () => set((s) => ({ showLandmarks: !s.showLandmarks })),
      toggleParticles:   () => set((s) => ({ showParticles: !s.showParticles })),
      toggleGrid:        () => set((s) => ({ showGrid: !s.showGrid })),
      toggleIdleAnimation: () => set((s) => ({ idleAnimation: !s.idleAnimation })),
      setFpsLimit:       (f) => set({ fpsLimit: f }),
      setAvatarColor:    (c) => set({ avatarColor: c }),
      setJointColor:     (c) => set({ jointColor: c }),
      setBoneColor:      (c) => set({ boneColor: c }),
      setCameraFov:      (f) => set({ cameraFov: f }),
      reset:             () => set(defaults),
    }),
    { name: 'isl-settings' },
  ),
)
