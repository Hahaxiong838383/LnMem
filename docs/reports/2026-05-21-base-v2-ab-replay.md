# 2026-05-21 base_v2 A/B replay

- queries_glob: `['/Users/rayman.chen/.lnmem/feedback/queries_2026-05-09.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-10.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-11.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-12.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-13.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-14.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-15.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-16.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-17.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-18.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-19.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-20.jsonl']`
- signals_dir: `/Users/rayman.chen/.lnmem/data`
- ckpt_v1: `poc/data/checkpoints/day12_with_self/checkpoint_day12.pt`
- ckpt_v2: `poc/data/checkpoints/base_v2/checkpoint_base_v2_best.pt`
- evaluable_events_total: **67**
- replayed_events: **64**
- skipped_events: **3** ({'missing_window': 3})
- response_label_counts: **{'mismatch': 64}**
- true_state_counts: **{'distracted': 42, 'focused': 22}**
- v2_mapping_assumption: class0=`neutral`, classes1-8=`low_energy,focused,distracted,fatigued,emotional_positive,emotional_negative,neutral,transitioning`; if class0 top-1 then use next-highest logit

## Summary

| metric | base_v1 | base_v2 | delta |
|---|---:|---:|---:|
| mismatch_rate | 100.00% | 100.00% | +0.00% |
| improvement_vs_v1 | - | 0.00% | - |
| v2 neutral-top fallback count | - | 64 | - |

## Caveat

- Current filter requires `user_response.true_state != null`; in the live feedback schema that mostly means user-explicit mismatch corrections, so `base_v1 mismatch_rate` is a tautological 100% on this slice and is **not** directly comparable to the historical 30.7% trigger-level mismatch rate from ADR-007.

## Bootstrap CI

| metric | 95% CI |
|---|---|
| mismatch_v1 | 100.00% ~ 100.00% |
| mismatch_v2 | 100.00% ~ 100.00% |
| improvement | 0.00% ~ 0.00% |

## Confusion Matrix

### base_v1

| true_state | low_energy | focused | distracted | fatigued | emotional_positive | emotional_negative | neutral | transitioning |
|---|---|---|---|---|---|---|---|---|
| low_energy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| focused | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 11 |
| distracted | 0 | 0 | 0 | 0 | 3 | 19 | 0 | 20 |
| fatigued | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| emotional_positive | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| emotional_negative | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| neutral | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| transitioning | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### base_v2

| true_state | low_energy | focused | distracted | fatigued | emotional_positive | emotional_negative | neutral | transitioning |
|---|---|---|---|---|---|---|---|---|
| low_energy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| focused | 0 | 0 | 1 | 19 | 2 | 0 | 0 | 0 |
| distracted | 1 | 0 | 0 | 39 | 2 | 0 | 0 | 0 |
| fatigued | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| emotional_positive | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| emotional_negative | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| neutral | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| transitioning | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Group Analysis

### persona_idx

| persona_idx | n | mismatch_v1 | mismatch_v2 | improvement |
|---|---:|---:|---:|---:|
| 30 | 64 | 100.00% | 100.00% | 0.00% |

### signal_path

| signal_path | n | mismatch_v1 | mismatch_v2 | improvement |
|---|---:|---:|---:|---:|
| missing | 17 | 100.00% | 100.00% | 0.00% |
| raw | 47 | 100.00% | 100.00% | 0.00% |

### vlm_label

| vlm_label | n | mismatch_v1 | mismatch_v2 | improvement |
|---|---:|---:|---:|---:|
| none | 64 | 100.00% | 100.00% | 0.00% |

## ADR-013 Suggestion

- recommendation: **不上线：base_v2 无改善或反向**
- note: historical ADR-007 target is 30.7% -> <15%; this replay used `64` evaluable brow_burst events.
