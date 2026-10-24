type SessionSaveResult = { success: boolean; error?: string }
type SessionSave = (chatId: string, sessionData: unknown) => Promise<SessionSaveResult>

export async function persistSession(
  save: SessionSave,
  chatId: string,
  sessionData: unknown,
  onSaved?: () => void,
): Promise<void> {
  const result = await save(chatId, sessionData)
  if (!result.success) {
    throw new Error(result.error || 'Failed to save chat session')
  }
  onSaved?.()
}
