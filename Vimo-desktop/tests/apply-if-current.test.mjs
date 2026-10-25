import assert from 'node:assert/strict'
import test from 'node:test'

import { applyIfCurrent } from '../src/renderer/src/utils/apply-if-current.ts'

function deferred() {
  let resolve
  const promise = new Promise((done) => {
    resolve = done
  })
  return { promise, resolve }
}

test('out-of-order chat loads apply only the result for the active route', async () => {
  const oldLoad = deferred()
  const newLoad = deferred()
  let oldRouteIsCurrent = true
  let newRouteIsCurrent = true
  let messages

  const oldRequest = applyIfCurrent(oldLoad.promise, () => oldRouteIsCurrent, (value) => {
    messages = value
  })
  oldRouteIsCurrent = false
  const newRequest = applyIfCurrent(newLoad.promise, () => newRouteIsCurrent, (value) => {
    messages = value
  })

  newLoad.resolve(['new-chat-message'])
  await newRequest
  oldLoad.resolve(['old-chat-message'])
  await oldRequest

  assert.deepEqual(messages, ['new-chat-message'])
})

test('only the newest overlapping refresh for one chat can update messages', async () => {
  const oldLoad = deferred()
  const newLoad = deferred()
  let latestRequest = 0
  let messages
  const load = (response) => {
    const requestId = ++latestRequest
    return applyIfCurrent(response.promise, () => latestRequest === requestId, (value) => {
      messages = value
    })
  }

  const oldRequest = load(oldLoad)
  const newRequest = load(newLoad)
  newLoad.resolve(['newest-chat-message'])
  await newRequest
  oldLoad.resolve(['stale-chat-message'])
  await oldRequest

  assert.deepEqual(messages, ['newest-chat-message'])
})

test('route cleanup prevents a pending result from updating state', async () => {
  const pendingLoad = deferred()
  let routeIsCurrent = true
  let messages
  const request = applyIfCurrent(pendingLoad.promise, () => routeIsCurrent, (value) => {
    messages = value
  })

  routeIsCurrent = false
  pendingLoad.resolve(['unmounted-chat-message'])
  await request

  assert.equal(messages, undefined)
})
