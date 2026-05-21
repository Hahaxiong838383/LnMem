# 2026-05-21 base_v1 / base_v2 / base_v2.1 replay

- queries_glob: `['/Users/rayman.chen/.lnmem/feedback/queries_2026-05-09.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-10.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-11.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-12.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-13.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-14.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-15.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-16.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-17.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-18.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-19.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-20.jsonl', '/Users/rayman.chen/.lnmem/feedback/queries_2026-05-21.jsonl']`
- signals_dir: `/Users/rayman.chen/.lnmem/data`
- evaluable_events_total: **69**
- replayed_events: **66**
- skipped_events: **3** ({'missing_window': 3})
- response_label_counts: **{'mismatch': 66}**
- true_state_counts: **{'distracted': 44, 'focused': 22}**
- compared_models: **base_v1, base_v2, checkpoint_base_v2_1_20260521T112745Z**

## Summary

| model | mismatch_rate | focused_acc | distracted_acc | mean_russell | mean_entropy | top1_neutral_count |
|---|---:|---:|---:|---:|---:|---:|
| base_v1 | 100.00% | 0.00% | 0.00% | 2.3205 | 0.6262 | 0 |
| base_v2 | 100.00% | 0.00% | 0.00% | 1.0948 | 1.6793 | 66 |
| checkpoint_base_v2_1_20260521T112745Z | 89.39% | 0.00% | 15.91% | 1.0710 | 1.7194 | 66 |

## Collapse Monitor

| model | mean_entropy | min_entropy | max_entropy | top1_neutral_count | status |
|---|---:|---:|---:|---:|---|
| base_v1 | 0.6262 | 0.0043 | 1.3737 | 0 | OK |
| base_v2 | 1.6793 | 1.5512 | 1.7965 | 66 | WARN collapse |
| checkpoint_base_v2_1_20260521T112745Z | 1.7194 | 1.5721 | 1.8416 | 66 | WARN collapse |

## Confusion Matrix

### base_v1

| true_state | low_energy | focused | distracted | fatigued | emotional_positive | emotional_negative | neutral | transitioning |
|---|---|---|---|---|---|---|---|---|
| low_energy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| focused | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 11 |
| distracted | 0 | 0 | 0 | 0 | 3 | 20 | 0 | 21 |
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
| distracted | 1 | 0 | 0 | 41 | 2 | 0 | 0 | 0 |
| fatigued | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| emotional_positive | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| emotional_negative | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| neutral | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| transitioning | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### checkpoint_base_v2_1_20260521T112745Z

| true_state | low_energy | focused | distracted | fatigued | emotional_positive | emotional_negative | neutral | transitioning |
|---|---|---|---|---|---|---|---|---|
| low_energy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| focused | 3 | 0 | 1 | 17 | 1 | 0 | 0 | 0 |
| distracted | 10 | 0 | 7 | 25 | 2 | 0 | 0 | 0 |
| fatigued | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| emotional_positive | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| emotional_negative | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| neutral | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| transitioning | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Russell Distance

### base_v1

- overall_mean: **2.3205**

| true_state | mean_arc_length |
|---|---:|
| low_energy | 0.0000 |
| focused | 2.3562 |
| distracted | 2.3026 |
| fatigued | 0.0000 |
| emotional_positive | 0.0000 |
| emotional_negative | 0.0000 |
| neutral | 0.0000 |
| transitioning | 0.0000 |

### base_v2

- overall_mean: **1.0948**

| true_state | mean_arc_length |
|---|---:|
| low_energy | 0.0000 |
| focused | 1.6065 |
| distracted | 0.8389 |
| fatigued | 0.0000 |
| emotional_positive | 0.0000 |
| emotional_negative | 0.0000 |
| neutral | 0.0000 |
| transitioning | 0.0000 |

### checkpoint_base_v2_1_20260521T112745Z

- overall_mean: **1.0710**

| true_state | mean_arc_length |
|---|---:|
| low_energy | 0.0000 |
| focused | 1.4637 |
| distracted | 0.8746 |
| fatigued | 0.0000 |
| emotional_positive | 0.0000 |
| emotional_negative | 0.0000 |
| neutral | 0.0000 |
| transitioning | 0.0000 |

