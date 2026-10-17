# E2E Italian TTS → STT benchmark

Strictly sequential pipeline over HTTP:

```
Piper it_IT-paola-medium ──POST /v1/audio/speech──▶ WAV
      │
      └── same bytes ──POST /v1/audio/transcriptions──▶ Italian Whisper (CT2 int8)
```

`e2e_tts_stt.py` waits for the complete TTS response, then sends that exact WAV
to STT, and waits for the transcript before starting the next sample. Only one
request is in flight at a time (`WHISPER_MAX_CONCURRENT=1`). STT models are
booted one at a time so large models fit in RAM.

## Run

```sh
# Both servers are started automatically; sibling stt-openai-server repo required
uv run python benchmarks/e2e_tts_stt.py \
  --models base,small,large-v3-turbo --runs 3 --warmup 1

# Single-word / very short replies (Ciao, Sì, No, Forse, Va bene, ...)
uv run python benchmarks/e2e_tts_stt.py \
  --models base,small,large-v3-turbo --suite short --runs 1

# Reuse already-running servers (single STT model)
uv run python benchmarks/e2e_tts_stt.py \
  --tts-url http://127.0.0.1:18000 \
  --stt-url http://127.0.0.1:18001 --models base
```

Generated audio/results go to `artifacts/e2e-tts-stt/<timestamp>/`
(`artifacts/` is gitignored): `results.json`, `report.md`, `audio/`, `logs/`.
Use `--out DIR` to place them elsewhere.

The default STT models are the Italian YODAS-only CT2 INT8 builds:

```text
base            LocalAI-io/whisper-base-it-yodas-only-ct2-int8
small           LocalAI-io/whisper-small-it-yodas-only-ct2-int8
large-v3-turbo  LocalAI-io/whisper-large-v3-turbo-it-yodas-only-ct2-int8
```

## Measured on the reference host (2026-10-17)

Intel N150, 4 logical CPUs but a **3-CPU cgroup quota**, 4 GiB RAM limit,
`beam_size=5`, `compute_type=int8`, `language=it`, Piper Paola 22.05 kHz WAV.
RTFs are audio-weighted (sum wall / sum audio); throughput is audio seconds per
wall second for TTS + STT combined.

### Sentences (1.4–32.6 s)

| STT model | startup | TTS RTF | STT RTF | E2E RTF | throughput | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 1.77 s | 0.138 | 0.317 | 0.455 | 2.17× | 11.5% |
| `small` | 1.73 s | 0.180 | 0.806 | 0.987 | 1.18× | 6.7% |
| `large-v3-turbo` | 6.42 s | 0.192 | 3.015 | 3.207 | 0.30× | 4.8% |

Whisper processes a fixed ~30 s window per encoder pass. The 32.6 s story
crosses into a second window; on this CPU only `base` is comfortably faster
than real time end-to-end, `small` is borderline, and `large-v3-turbo` is
~3.2× slower than real time at beam 5.

### Single words / short replies (0.22–1.02 s)

| STT model | median STT latency | median E2E latency | exact transcripts |
| --- | ---: | ---: | ---: |
| `base` | 2.60 s | 2.68 s | 6/7 |
| `small` | 4.91 s | 4.98 s | 5/7 |
| `large-v3-turbo` | 18.82 s | 18.90 s | 5/7 |

Latency is almost independent of utterance length: a 0.3 s "Sì" still pays the
full encoder pass. For sub-second replies, use absolute latency and batching,
not RTF. All models transcribed `Ciao`, `Sì`, `No`, `Non lo so` and
`Grazie mille`; all failed on the isolated word **`Forse`** (heard as
`Posso`, `Posè` or `Cos'è?` depending on model/run).
This is a one-run synthetic-Piper sanity check, not a human-recording quality
claim — verify `Forse` with real audio before drawing ASR conclusions.

Absolute numbers vary with CPU pressure; an earlier identical `large-v3-turbo`
run measured 2.08 STT RTF vs 3.01 here. Treat the numbers as ratios on this
box, not universal constants.
