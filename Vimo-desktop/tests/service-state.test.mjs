import assert from 'node:assert/strict'
import test from 'node:test'

import { preserveServiceStateOnStopFailure } from '../src/renderer/src/lib/service-state.ts'

test('failed stop preserves the last known running and ImageBind state', () => {
  const previous = {
    isRunning: true,
    imagebindLoaded: true,
    message: 'Service ready',
  }

  assert.deepEqual(
    preserveServiceStateOnStopFailure(previous, 'Stop request failed'),
    {
      ...previous,
      error: 'Stop request failed',
    },
  )
})
