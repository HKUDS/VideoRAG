export function createLatestRequestGuard() {
  let currentRequest = 0

  return {
    begin(): number {
      currentRequest += 1
      return currentRequest
    },
    isCurrent(requestId: number): boolean {
      return requestId === currentRequest
    },
    invalidate(): void {
      currentRequest += 1
    },
  }
}
