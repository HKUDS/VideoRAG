export interface ServiceStateSnapshot {
  isRunning: boolean
  imagebindLoaded: boolean
  message?: string
  error?: string
}

export function preserveServiceStateOnStopFailure(
  previous: ServiceStateSnapshot,
  error: string | undefined
): ServiceStateSnapshot {
  return { ...previous, error }
}
