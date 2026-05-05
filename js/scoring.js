const HEX_DIMENSIONS = [
  'honesty_humility',
  'emotionality',
  'extraversion',
  'agreeableness',
  'conscientiousness',
  'openness'
];

const SIGNAL_DIMENSIONS = {
  H: 'honesty_humility',
  Em: 'emotionality',
  N: 'emotionality',
  X: 'extraversion',
  E: 'extraversion',
  A: 'agreeableness',
  C: 'conscientiousness',
  O: 'openness'
};

export function scoreAssessment(questions, answers) {
  const hex = scoreHex(questions.hexaco || [], answers);
  const attachment = scoreAttachment(questions.ecr || [], answers);
  const motivation = scoreMotivation(questions, answers);
  const situational = scoreSituational(questions.situational || [], answers);
  return { hex, attachment, motivation, situational };
}

export function scoreHex(items, answers) {
  return Object.fromEntries(HEX_DIMENSIONS.map((dimension) => {
    const rows = items.filter((item) => item.dimension === dimension);
    const scores = rows.map((item) => scoreLikert(item, answers[item.id], 5));
    return [dimension, average(scores)];
  }));
}

export function scoreAttachment(items, answers) {
  const anxiety = averageByDimension(items, answers, 'anxiety', 5);
  const avoidance = averageByDimension(items, answers, 'avoidance', 5);
  const style = attachmentStyle(anxiety, avoidance);
  return { anxiety, avoidance, style };
}

export function scoreMotivation(questions, answers) {
  const motivationItems = questions.motivation || [];
  const groups = groupScores(motivationItems, answers, 'group', 7);
  const values = scoreValues(questions.values || [], answers);
  const selfEfficacy = groups.gse ?? average([]);
  return {
    self_efficacy: selfEfficacy,
    groups,
    values,
    top: values.slice(0, 2).map((item) => item.value),
    low: values.slice(-2).map((item) => item.value)
  };
}

export function scoreValues(items, answers) {
  return items.map((item) => ({
    value: item.value || item.id,
    score: scoreLikert(item, answers[item.id], 7),
    raw: readLikertValue(answers[item.id], item.default ?? midpoint(maxLikert(item, 7)))
  })).sort((a, b) => b.score - a.score);
}

export function scoreSituational(items, answers) {
  const selected = items.map((item) => {
    const answer = answers[item.id];
    const key = readOptionKey(answer);
    const option = (item.options || []).find((row) => row.key === key);
    return {
      id: item.id,
      scenario: item.scenario || item.text || item.title,
      key,
      text: option?.text || '',
      signals: normalizeSignals(option?.signals || option?.signal || answer?.signals)
    };
  }).filter((item) => item.key);

  const totals = {};
  for (const item of selected) {
    for (const signal of item.signals) {
      totals[signal.dimension] = (totals[signal.dimension] || 0) + signal.weight;
    }
  }

  const sorted = Object.entries(totals)
    .map(([dimension, score]) => ({ dimension, score }))
    .sort((a, b) => b.score - a.score);

  return { selected, totals: sorted };
}

export function scoreLikert(question, answer, fallbackMax) {
  const max = maxLikert(question, fallbackMax);
  let value = readLikertValue(answer, midpoint(max));
  if (question?.reverse) value = max + 1 - value;
  return normalizeLikert(value, max);
}

export function normalizeSignals(input) {
  if (!input) return [];
  if (Array.isArray(input)) return input.flatMap((item) => normalizeSignals(item));
  if (typeof input === 'object') return normalizeSignalObject(input);
  return String(input)
    .split(/[，,、/;；]/)
    .map((token) => token.trim())
    .filter(Boolean)
    .map(parseSignalToken)
    .filter(Boolean);
}

function normalizeSignalObject(input) {
  return Object.entries(input).map(([key, value]) => ({
    label: `${key}:${value}`,
    dimension: SIGNAL_DIMENSIONS[key] || key,
    weight: Number(value) || 0
  }));
}

function parseSignalToken(token) {
  const match = token.match(/^(中高|高|中|低)?\s*([A-Za-z]+|责任|自我管理|自我关怀|自我接纳)/);
  if (!match) return null;
  const level = match[1] || '中';
  const code = match[2];
  return {
    label: token,
    dimension: SIGNAL_DIMENSIONS[code] || code,
    weight: signalWeight(level)
  };
}

function signalWeight(level) {
  if (level === '高') return 1;
  if (level === '中高') return 0.75;
  if (level === '低') return -0.5;
  return 0.5;
}

function averageByDimension(items, answers, dimension, fallbackMax) {
  const rows = items.filter((item) => item.dimension === dimension);
  return average(rows.map((item) => scoreLikert(item, answers[item.id], fallbackMax)));
}

function groupScores(items, answers, key, fallbackMax) {
  const groups = {};
  for (const item of items) {
    const group = item[key] || 'default';
    groups[group] ||= [];
    groups[group].push(scoreLikert(item, answers[item.id], fallbackMax));
  }
  return Object.fromEntries(Object.entries(groups).map(([group, scores]) => [group, average(scores)]));
}

function attachmentStyle(anxiety, avoidance) {
  if (anxiety < 0.5 && avoidance < 0.5) return '安全型';
  if (anxiety >= 0.5 && avoidance < 0.5) return '焦虑型';
  if (anxiety < 0.5 && avoidance >= 0.5) return '回避型';
  return '混乱型';
}

function readOptionKey(answer) {
  if (typeof answer === 'string') return answer;
  if (!answer || typeof answer !== 'object') return '';
  return answer.key || answer.option || answer.value || '';
}

function readLikertValue(answer, fallback) {
  if (answer == null || answer === '') return fallback;
  if (typeof answer === 'number') return answer;
  if (typeof answer === 'string' && !Number.isNaN(Number(answer))) return Number(answer);
  if (typeof answer !== 'object') return fallback;
  const value = answer.likert ?? answer.score ?? answer.value ?? answer.rating;
  return Number.isFinite(Number(value)) ? Number(value) : fallback;
}

function maxLikert(question, fallback) {
  return question?.likert || question?.scale_max || question?.max || fallback;
}

function midpoint(max) {
  return Math.ceil(max / 2);
}

function normalizeLikert(value, max) {
  if (max <= 1) return 0;
  return clamp((value - 1) / (max - 1));
}

function average(values) {
  const safe = values.filter((value) => Number.isFinite(value));
  if (safe.length === 0) return 0;
  return safe.reduce((sum, value) => sum + value, 0) / safe.length;
}

function clamp(value) {
  return Math.max(0, Math.min(1, value));
}
