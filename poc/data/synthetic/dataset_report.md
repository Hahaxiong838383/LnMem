# 数据集质量报告

- 画像数: **3**
- 剧本数: **3**
- 总时长: **24.00h**
- 平均时长/画像: **8.00h**

## 1. 画像总览

| persona_id | scripts | events | duration_h | errors |
|------------|---------|--------|-----------|--------|
| vboss_001 | 1 | 50 | 8.00 | 0 |
| vboss_002 | 1 | 55 | 8.00 | 0 |
| vboss_003 | 1 | 47 | 8.00 | 0 |

## 2. 状态分布（事件计数）

| persona_id | low_energy | focused | distracted | fatigued | emotional_positive | emotional_negative | neutral | transitioning |
|---|---|---|---|---|---|---|---|---|
| vboss_001 | 7 | 21 | 4 | 4 | 3 | 2 | 5 | 4 |
| vboss_002 | 4 | 23 | 2 | 3 | 5 | 7 | 5 | 6 |
| vboss_003 | 5 | 11 | 3 | 7 | 5 | 5 | 7 | 4 |

## 3. 活动分布（事件计数）

| persona_id | work_focus | work_meeting | break_short | break_long | commute | leisure | sleep | transition |
|---|---|---|---|---|---|---|---|---|
| vboss_001 | 26 | 9 | 5 | 2 | 0 | 2 | 0 | 6 |
| vboss_002 | 15 | 18 | 7 | 1 | 3 | 1 | 0 | 10 |
| vboss_003 | 24 | 3 | 7 | 2 | 0 | 7 | 0 | 4 |

## 4. HEXACO 6 维欧式距离矩阵

> 标准（data-generation-todo.md §4 #1）：两两 ≥ 0.4。

| | vboss_001 | vboss_002 | vboss_003 |
|---|---|---|---|
| **vboss_001** | — | 0.615 | 0.527 |
| **vboss_002** | 0.615 | — | 0.708 |
| **vboss_003** | 0.527 | 0.708 | — |

## 5. 校验问题汇总

✅ 无错误。

