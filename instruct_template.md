### Abstract Template

```
[
    {
        "role": "system",
        "content": [
            { "type":"text", "text": ""},
            {"type": "audio", "audio": ""},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": ""},
            {"type": "audio", "audio": ""},
        ]
    },
    {
        "role": "assistant",
        "thinking": True/False/None,
        "Chain-of-thought": "",
        "content": [
            {"type": "text", "text": ""},
            {"type": "audio", "audio": ""},
        ]
    },
]
```

### ASR Example

```
[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Convert this speech recording to text" / "识别并转写这段语音"},
            {"type": "audio", "audio": ""},
        ]
    },
    {
        "role": "assistant",
        "thinking": False,
        "content": [
            {"type": "text", "text": ""},
        ]
    },
]
```

### TTS

```
[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please convert this text to speech: ..."},
        ]
    },
    {
        "role": "assistant",
        "thinking": None,
        "content": [
            {"type": "text", "text": ""},
            {"type": "audio", "audio": ""},
        ]
    },
]
```

### Instruct TTS

```
[
    {
        "role": "system",
        "content": [
            { "type":"text", "text": "你需要根据指定的风格指令和文本内容来生成和语音prompt具有相同音色的语音。你的音色应该是："},
            {"type": "audio", "audio": ""},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please convert this text to speech: ..."},
            {"type": "audio", "audio": ""},
        ]
    },
    {
        "role": "assistant",
        "thinking": True,
        "Chain-of-thought": "",
        "content": [
            {"type": "text", "text": ""},
            {"type": "audio", "audio": ""},
        ]
    },
]
```

### Audio Understanding

```
[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": ""},
            {"type": "audio", "audio": ""},
        ]
    },
    {
        "role": "assistant",
        "thinking": False,
        "content": [
            {"type": "text", "text": ""},
        ]
    },
]
```

### Audio Understanding with Thinking

```
[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": ""},
            {"type": "audio", "audio": ""},
        ]
    },
    {
        "role": "assistant",
        "thinking": True,
        "Chain-of-thought": "",
        "content": [
            {"type": "text", "text": ""},
        ]
    },
]
```

### Spoken Dialogue

```
[
    {
        "role": "system",
        "content": [
            {"type":"text", "text": "Your voice should be:"},
            {"type": "audio", "audio": ""},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": ""},
        ]
    },
    {
        "role": "assistant",
        "thinking": None,
        "content": [
            {"type": "text", "text": ""},
            {"type": "audio", "audio": ""},
        ]
    },
]
```