# Sugar-AI API contracts

Every endpoint states what it accepts and what it returns. Requests are
validated before a model is called, and failures come back in one shape
whatever caused them.

## Authentication

Send your key as `X-API-Key`. Requests without a valid key are refused
before any validation happens, so a bad key and a bad body report the
key.

## Errors

Every error, from any layer, arrives as:

```json
{"error": {"code": "validation_error", "message": "question: field required"}}
```

| Code | Status | Meaning |
|---|---|---|
| `unauthorized` | 401 | Key missing or unknown |
| `forbidden` | 403 | Key lacks permission for this action |
| `not_found` | 404 | No such route |
| `validation_error` | 422 | The request did not match the contract |
| `modality_not_supported` | 422 | The active model cannot accept this media |
| `quota_exceeded` | 429 | Daily request limit reached |
| `internal_error` | 500 | Unhandled failure |

## Endpoints

### POST /ask

Answers a question using retrieval over the bundled documentation.

```json
{"question": "How do I move a sprite?"}
```

Returns `{"answer", "user", "quota": {"remaining", "total"}}`.

### POST /ask-llm

Answers a question with no retrieval. Accepts `attachments`, so the
question can refer to a picture or a recording.

```json
{"question": "What is in this picture?", "attachments": [ ... ]}
```

Returns the same shape as `/ask`.

### POST /ask-llm-prompted

Two modes, selected by `chat`.

**Prompted** (`chat: false`) needs `question` and `custom_prompt`, and
takes optional `attachments`. When attachments are present the custom
prompt is sent as the system message, because media can only travel as
chat content.

```json
{"chat": false, "question": "What is this?", "custom_prompt": "Explain simply."}
```

Returns `{"answer", "user", "quota", "generation_params"}`.

**Chat** (`chat: true`) needs `messages`. Each message's `content` is a
string or a list of parts.

```json
{"chat": true, "messages": [{"role": "user", "content": "Hello"}]}
```

Returns `{"choices": [{"message", "index", "finish_reason"}], "user", "quota", "generation_params"}`.

Generation parameters, valid in both modes: `max_length` (1–8192),
`temperature` (0–2), `top_p` (0–1], `top_k` (≥0), `repetition_penalty`
(0–2], `truncation`.

### POST /debug

```json
{"code": "print(1", "context": false}
```

`context: true` explains what the code does; `false` debugs it. Returns
the `/ask` shape.

### GET /health

Returns `{"status", "provider", "model", "modalities"}`. Read
`modalities` to learn what the running model accepts before sending
media.

### Query parameters

`/ask`, `/ask-llm` and `/debug` still accept their old query parameters
(`?question=`, `?code=`) so existing callers keep working. The JSON body
is the contract; the parameters are deprecated.

## Multimodal input

A message's `content` is either a string or a list of typed parts:

```json
{"type": "text",  "text": "What is this?"}
{"type": "image", "mime_type": "image/png", "data": "<base64>"}
{"type": "audio", "mime_type": "audio/wav", "data": "<base64>"}
```

`data` is raw base64 with no `data:` prefix. Accepted types are
`image/png`, `image/jpeg`, `image/webp`, `audio/wav`, `audio/mpeg` and
`audio/ogg`. Images are limited to 5 MB and audio to 20 MB after
decoding. Malformed base64 is rejected at the edge.

Audio is input only: the model listens and answers in text. Spoken
replies are not part of this contract.

### What each provider accepts

Support varies by model, not by provider, so Sugar-AI asks the backend
where it can and falls back to a per-provider default where it cannot.

| Provider | Accepts | How it is known |
|---|---|---|
| Ollama | per model | Asked via `/api/show` at startup; `vision` means image. No audio input |
| Gemini | text, image, audio | Default; every current `generateContent` model takes all three |
| OpenAI-compatible | text | Default; depends on the model, so widen it explicitly |
| HuggingFace | text | Local text-generation pipeline |

Sending media a model cannot accept returns 422 `modality_not_supported`
naming what you sent and what it accepts, rather than failing upstream.
If a backend cannot be asked (server down at startup, unknown model),
the provider's default stands.

To widen an OpenAI-compatible deployment whose model does handle media:

```
AI_SUPPORTED_MODALITIES=text,image,audio
```

Text is always accepted and cannot be configured away.

## Examples

Ask a question:

```bash
curl -X POST http://localhost:8000/ask-llm \
  -H "X-API-Key: $SUGAR_AI_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is a loop?"}'
```

Ask about a picture:

```bash
IMAGE=$(base64 -w0 drawing.png)
curl -X POST http://localhost:8000/ask-llm \
  -H "X-API-Key: $SUGAR_AI_KEY" -H "Content-Type: application/json" \
  -d "{\"question\": \"What did I draw?\",
       \"attachments\": [{\"type\": \"image\",
                          \"mime_type\": \"image/png\",
                          \"data\": \"$IMAGE\"}]}"
```

Ask about a recording:

```bash
CLIP=$(base64 -w0 question.wav)
curl -X POST http://localhost:8000/ask-llm \
  -H "X-API-Key: $SUGAR_AI_KEY" -H "Content-Type: application/json" \
  -d "{\"question\": \"Answer what I asked.\",
       \"attachments\": [{\"type\": \"audio\",
                          \"mime_type\": \"audio/wav\",
                          \"data\": \"$CLIP\"}]}"
```

A picture inside a conversation:

```bash
curl -X POST http://localhost:8000/ask-llm-prompted \
  -H "X-API-Key: $SUGAR_AI_KEY" -H "Content-Type: application/json" \
  -d "{\"chat\": true,
       \"messages\": [{\"role\": \"user\",
                       \"content\": [{\"type\": \"text\", \"text\": \"What is this?\"},
                                     {\"type\": \"image\",
                                      \"mime_type\": \"image/png\",
                                      \"data\": \"$IMAGE\"}]}]}"
```
