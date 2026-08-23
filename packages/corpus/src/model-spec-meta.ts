/** shared ModelSpec allowlists synced to the Python contract */

export const ADAPTER_METHODS = ["lora", "qlora"] as const
export const TRAIN_PRECISIONS = ["bf16", "fp16"] as const
export const TRAIN_QUANTS = ["none", "nf4"] as const
export const EXPORT_QUANTS = ["bf16", "q4_k_m"] as const
export const SCHEDULERS = ["cosine", "linear"] as const
/** distinguishes in-loop loss from post-training release gates */
export const CHECKPOINT_METRICS = ["eval_loss", "composite_gates"] as const
export const TARGET_MODULES = [
  "q_proj",
  "k_proj",
  "v_proj",
  "o_proj",
  "gate_proj",
  "up_proj",
  "down_proj",
] as const

export const MODEL_SPEC_META = {
  adapterMethods: [...ADAPTER_METHODS],
  trainPrecisions: [...TRAIN_PRECISIONS],
  trainQuants: [...TRAIN_QUANTS],
  exportQuants: [...EXPORT_QUANTS],
  schedulers: [...SCHEDULERS],
  checkpointMetrics: [...CHECKPOINT_METRICS],
  targetModules: [...TARGET_MODULES],
} as const
