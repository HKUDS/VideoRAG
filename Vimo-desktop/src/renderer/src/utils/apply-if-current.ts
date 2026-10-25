export async function applyIfCurrent<T>(
  resultPromise: Promise<T>,
  isCurrent: () => boolean,
  apply: (result: T) => void,
): Promise<void> {
  const result = await resultPromise
  if (isCurrent()) apply(result)
}
