import assert from 'node:assert/strict'
import test from 'node:test'

import { shouldFollowMessages } from '../src/renderer/src/utils/scroll-follow.ts'

test('message updates keep the view pinned when already near the bottom', () => {
  assert.equal(shouldFollowMessages(1200, 540, 600), true)
  assert.equal(shouldFollowMessages(1200, 520, 600), true)
})

test('message updates do not pull the view down when the reader is reviewing history', () => {
  assert.equal(shouldFollowMessages(1200, 300, 600), false)
})
