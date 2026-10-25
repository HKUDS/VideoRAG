import assert from 'node:assert/strict'
import test from 'node:test'

import { createLatestRequestGuard } from '../src/renderer/src/lib/latest-request.ts'

function deferred() {
  let resolve
  const promise = new Promise((done) => {
    resolve = done
  })
  return { promise, resolve }
}

test('stale session responses cannot replace the latest result', async () => {
  const guard = createLatestRequestGuard()
  const oldResponse = deferred()
  const newResponse = deferred()
  let sessionInfo

  const load = async (response) => {
    const requestId = guard.begin()
    const result = await response.promise
    if (guard.isCurrent(requestId)) sessionInfo = result
  }

  const oldLoad = load(oldResponse)
  const newLoad = load(newResponse)
  newResponse.resolve('new-session')
  await newLoad
  oldResponse.resolve('old-session')
  await oldLoad

  assert.equal(sessionInfo, 'new-session')
})

test('unmount invalidation prevents an outstanding response from applying', async () => {
  const guard = createLatestRequestGuard()
  const response = deferred()
  let sessionInfo
  const requestId = guard.begin()
  const load = response.promise.then((result) => {
    if (guard.isCurrent(requestId)) sessionInfo = result
  })

  guard.invalidate()
  response.resolve('unmounted-session')
  await load

  assert.equal(sessionInfo, undefined)
})
