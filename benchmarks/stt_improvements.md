# STT tuning, real-audio accuracy, and short-command study

Results from three experiments on the reference host (Intel N150, 3-CPU cgroup
quota, 4 GiB RAM, `language=it`, `compute_type=int8`). All requests are strictly
sequential.

## 1. Tuning matrix (`small`, synthetic Italian)

`stt_tuning_matrix.py` runs a staged matrix through the real TTS→STT pipeline.
Sentence suite: 5 clips, 1.4–32.6 s, 2 runs. Short suite: 7 clips, 0.2–1.0 s,
1 run. STT RTF is audio-weighted over each suite.

| config | beam | threads | batch | VAD | sentence STT RTF | sentence WER | short STT RTF |
| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| baseline | 5 | 0 | 0 | no | 0.764 | 9.6% | 9.163 |
| threads3 | 5 | 3 | 0 | no | 0.766 | 10.8% | 8.133 |
| threads3_beam2 | 2 | 3 | 0 | no | 0.875 | 13.0% | 9.087 |
| threads3_beam1 | 1 | 3 | 0 | no | **0.631** | 12.3% | 9.622 |
| threads3_beam2_batch4 | 2 | 3 | 4 | no | **0.627** | 9.6% | 8.390 |
| threads3_beam5_batch4 | 5 | 3 | 4 | no | 0.682 | **7.0%** | 8.238 |
| threads3_beam2_temps | 2 | 3 | 0 | no | 0.632 | 8.4% | 10.806 |
| threads3_vad | 5 | 3 | 0 | yes | 0.705 | **6.7%** | 9.217 |

Takeaways:

- Matching `cpu_threads` to the 3-CPU quota did **not** improve throughput.
- Beam 1, or beam 2 + batch 4, is ~18% faster on the sentence suite, but beam
  1/2 hurt short-clip accuracy and produced a repetition loop on the long story.
- On this synthetic suite, `beam5 + batch4` is the best speed/accuracy
  compromise: ~11% faster than baseline with the lowest sentence WER in the
  faster half of the table. The real-audio FLEURS result below shows that
  batch 4 can regress WER on long clips, so treat this as synthetic-only.
- VAD's 0.705 vs 0.764 is within run-to-run noise; on clean TTS it is not a
  clear win and adds a VAD pass.
- Short-clip latency is ~5 s on `small` regardless of config; tuning cannot
  remove Whisper's fixed 30 s encoder window.

Caveat: 2 runs and a noisy/shared host. Treat differences under ~10% as noise.

## 2. Real Italian audio: FLEURS

`prepare_fleurs_it.py` + `stt_accuracy.py`, 24 real human clips (5.3–52.9 s,
374.5 s total), beam 5, threads 3. This is real read speech, not Piper.

| model | startup | STT RTF | WER | mean clip WER | RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 1.54 s | **0.162** | 13.0% | 14.1% | 316 MB |
| `small` | 1.52 s | 0.418 | 7.0% | 7.6% | 656 MB |
| `medium` | 3.62 s | 1.052 | 4.8% | 5.8% | 1381 MB |
| `large-v3-turbo` | 4.29 s | 1.342 | **4.2%** | 4.5% | 1167 MB |
| `small` + batch 4 | 2.63 s | **0.352** | 9.3% | 8.5% | 677 MB |

Takeaways:

- On real audio, `small` is the best real-time compromise: ~2.4× faster than
  `medium` and ~3.2× faster than turbo, for +2.2/+2.8 pp WER.
- `base` is ~2.6× faster than `small` but costs +6 pp WER.
- `medium` and turbo are accuracy-first; both are around or slower than
  real time on this CPU.
- **Batch 4 is a throughput tool, not a free win.** On real audio it cut
  `small` RTF 0.418→0.352, but the 52.9 s clip went 21.57→10.29 s while WER
  jumped 3.6%→32.1% (aggregate 7.0%→9.3%). Batched mode changes segmentation/
  conditioning; validate WER on your domain before enabling it.
- Real-audio WER is higher than the synthetic-TTS numbers (e.g., `small`
  7.0% vs 6.7% earlier; `turbo` 4.2% vs 4.8%), and the ranking differs, so
  don't select a model from synthetic audio alone.

## 3. Short-command layer

`stt_command_eval.py` applies a normalized edit-distance match against:
`ciao`, `sì`, `no`, `forse`, `va bene`, `non lo so`, `grazie mille`.
Threshold 2. Also tested `WHISPER_INITIAL_PROMPT` with the command list.

| run | raw exact | mapped | false accepts |
| --- | ---: | ---: | ---: |
| baseline `base` | 6/7 | 6/7 | — |
| baseline `small` | 5/7 | 6/7 | — |
| baseline `large-v3-turbo` | 5/7 | **7/7** | — |
| prompt `base` | 6/7 | 6/7 | — |
| prompt `small` | 6/7 | 6/7 | — |
| prompt `large-v3-turbo` | 6/7 | **7/7** | — |
| FLEURS non-command clips | — | — | **0/96 (0.0%)** |

Takeaways:

- The initial prompt fixed most confusions (`Va bene` vs `Fa bene`,
  `Non lo so` vs `Non la so`) but **not `Forse`** for base/small.
- The fuzzy layer recovers `Fa bene`, `Posè`, `Non la so`, and turbo's `Forza`
  → `Forse`; turbo reaches 7/7 with zero false accepts on 96 real FLEURS
  sentences.
- Base/small still fail `Forse` (`Posso`/`Posa`); a trained command classifier
  or a Vosk-style grammar is needed for robust closed-set recognition.
- The command test is synthetic TTS and one run. The `Forse` audio may be a
  Piper pronunciation artifact; validate with real human recordings.

## Recommendation

- **General real-time transcription:** `small`, beam 5, **batch 0** for
  accuracy; enable batch 4 only for long/offline throughput after checking WER.
- **Accuracy-first:** `large-v3-turbo` (or `medium` if memory is tighter),
  batch 0.
- **Maximum CPU headroom:** `base`, accepting the WER penalty.
- **Short replies:** initial prompt + closed-set matcher helps, but use a
  dedicated command/streaming model if `sì/no/forse` reliability is critical.
- **Do not** rely on thread tuning or VAD for speed on this host.
- OpenVINO GPU remains an optional experiment; the CPU `small` path is
  already near real time on real audio (RTF 0.418).

## Reproduce

```sh
# 1. tuning matrix
uv run python benchmarks/stt_tuning_matrix.py \
  --models small --runs-main 2 --runs-short 1

# 2. real audio (downloads FLEURS it_it test split)
/workspace/stt-openai-server/.venv/bin/python \
  benchmarks/prepare_fleurs_it.py --out artifacts/fleurs-it --per-bucket 8
/workspace/stt-openai-server/.venv/bin/python benchmarks/stt_accuracy.py \
  --manifest artifacts/fleurs-it/manifest.jsonl \
  --models base,small,medium,large-v3-turbo --runs 1 --cpu-threads 3

# 3. command layer
/workspace/stt-openai-server/.venv/bin/python benchmarks/stt_command_eval.py \
  --short-results artifacts/.../results.json \
  --fleurs-results artifacts/.../results.json --threshold 2
```
