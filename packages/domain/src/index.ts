/**
 * shared Rocky behavior contracts
 * these values are consumed by training, Brain, Studio, Runtime, and motion layers
 **/

export const Emotions = ["neutral", "excited", "alarmed", "sad", "happy", "curious"] as const
export type Emotion = (typeof Emotions)[number]

/** Gestures grounded in Project Hail Mary narration, not invented human body language */
export const Gestures = [
  "none",
  "jazz_hands",
  "tap_carapace",
  "tap_divider",
  "point",
  "claw_on_divider",
  "fold_claws",
  "perk_up",
  "hunker_carapace",
  "raise_carapace",
  "sink_carapace",
  "shift_carapace",
  "cock_carapace",
  "wave_arms",
  "wave_absently",
  "bounce",
  "spider_walk",
  "skitter",
] as const
export type Gesture = (typeof Gestures)[number]

export interface Metadata {
  readonly emotion: Emotion
  readonly intensity: number
  readonly gesture: Gesture
}

export interface BodyModel {
  readonly supportedGestures: ReadonlyArray<Gesture>
}

export const defaultMetadata = (): Metadata => ({
  emotion: "neutral",
  intensity: 0.5,
  gesture: "none",
})

export const isEmotion = (value: string): value is Emotion =>
  (Emotions as readonly string[]).includes(value)

export const isGesture = (value: string): value is Gesture =>
  (Gestures as readonly string[]).includes(value)

export const isIntensity = (value: number): boolean => value >= 0 && value <= 1

export const supportsGesture = (bodyModel: BodyModel, gesture: Gesture): boolean =>
  bodyModel.supportedGestures.includes(gesture)

export const parseMetadata = (raw: unknown): Metadata => {
  if (typeof raw !== "object" || raw === null) {
    return defaultMetadata()
  }

  const record = raw as Record<string, unknown>
  const fallback = defaultMetadata()

  const emotion =
    typeof record.emotion === "string" && isEmotion(record.emotion)
      ? record.emotion
      : fallback.emotion

  const gesture =
    typeof record.gesture === "string" && isGesture(record.gesture)
      ? record.gesture
      : fallback.gesture

  const intensity =
    typeof record.intensity === "number" && isIntensity(record.intensity)
      ? record.intensity
      : fallback.intensity

  return { emotion, intensity, gesture }
}
