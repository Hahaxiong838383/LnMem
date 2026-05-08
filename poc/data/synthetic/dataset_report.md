# 数据集质量报告

- 画像数: **3**
- 剧本数: **3**
- 总时长: **24.00h**
- 平均时长/画像: **8.00h**

## 1. 画像总览

| persona_id | scripts | events | duration_h | errors |
|------------|---------|--------|-----------|--------|
| vboss_001 | 1 | 55 | 8.00 | 0 |
| vboss_002 | 1 | 52 | 8.00 | 0 |
| vboss_003 | 1 | 48 | 8.00 | 0 |

## 2. 状态分布（事件计数）

| persona_id | low_energy | focused | distracted | fatigued | emotional_positive | emotional_negative | neutral | transitioning |
|---|---|---|---|---|---|---|---|---|
| vboss_001 | 5 | 23 | 6 | 4 | 9 | 2 | 4 | 2 |
| vboss_002 | 5 | 13 | 4 | 5 | 5 | 7 | 5 | 8 |
| vboss_003 | 7 | 12 | 2 | 7 | 8 | 6 | 3 | 3 |

## 3. 活动分布（事件计数）

| persona_id | work_focus | work_meeting | break_short | break_long | commute | leisure | sleep | transition |
|---|---|---|---|---|---|---|---|---|
| vboss_001 | 26 | 10 | 6 | 1 | 0 | 7 | 0 | 5 |
| vboss_002 | 14 | 17 | 4 | 4 | 4 | 2 | 0 | 7 |
| vboss_003 | 18 | 7 | 8 | 2 | 1 | 10 | 0 | 2 |

## 4. HEXACO 6 维欧式距离矩阵

> 标准（data-generation-todo.md §4 #1）：两两 ≥ 0.4。

| | vboss_001 | vboss_002 | vboss_003 |
|---|---|---|---|
| **vboss_001** | — | 0.615 | 0.527 |
| **vboss_002** | 0.615 | — | 0.708 |
| **vboss_003** | 0.527 | 0.708 | — |

## 5. 校验问题汇总

✅ 无错误。

