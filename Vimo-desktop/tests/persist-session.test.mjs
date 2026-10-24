import assert from 'node:assert/strict'
import test from 'node:test'

import { persistSession } from '../src/renderer/src/utils/persist-session.ts'

test('failed persistence rejects without notifying listeners', async () => {
  let notifications = 0
  await assert.rejects(
    persistSession(
      async () => ({ success: false, error: 'Storage unavailable' }),
      'chat-test',
      { messages: [] },
      () => notifications++,
    ),
    /Storage unavailable/,
  )
  assert.equal(notifications, 0)
})

test('successful persistence notifies listeners once', async () => {
  let notifications = 0
  await persistSession(
    async () => ({ success: true }),
    'chat-test',
    { messages: [] },
    () => notifications++,
  )
  assert.equal(notifications, 1)
})
