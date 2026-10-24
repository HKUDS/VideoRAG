import assert from 'node:assert/strict';
import test from 'node:test';
import { shouldSubmitOnEnter } from '../src/renderer/src/utils/chat-input.ts';

const keyEvent = ({ key = 'Enter', shiftKey = false, isComposing = false, keyCode = 13 } = {}) => ({
  key,
  shiftKey,
  nativeEvent: { isComposing, keyCode },
});

test('submits on plain Enter', () => {
  assert.equal(shouldSubmitOnEnter(keyEvent()), true);
});

test('preserves Shift+Enter for a newline', () => {
  assert.equal(shouldSubmitOnEnter(keyEvent({ shiftKey: true })), false);
});

test('does not submit while an IME composition is active', () => {
  assert.equal(shouldSubmitOnEnter(keyEvent({ isComposing: true })), false);
});

test('does not submit for the legacy IME confirmation key code', () => {
  assert.equal(shouldSubmitOnEnter(keyEvent({ keyCode: 229 })), false);
});

test('does not submit for other keys', () => {
  assert.equal(shouldSubmitOnEnter(keyEvent({ key: 'a' })), false);
});
