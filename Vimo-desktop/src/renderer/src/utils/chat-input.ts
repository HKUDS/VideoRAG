type ChatInputKeyboardEvent = {
  key: string;
  shiftKey: boolean;
  nativeEvent: {
    isComposing?: boolean;
    keyCode: number;
  };
};

export function shouldSubmitOnEnter(event: ChatInputKeyboardEvent): boolean {
  return event.key === 'Enter'
    && !event.shiftKey
    && !event.nativeEvent.isComposing
    && event.nativeEvent.keyCode !== 229;
}
