const STORAGE_KEY = 'lnmem_persona_tool';

export const QUESTION_FILES = {
  hexaco: 'data/hexaco-questions.json',
  ecr: 'data/ecr-questions.json',
  motivation: 'data/motivation-questions.json',
  values: 'data/values-questions.json',
  situational: 'data/situational-questions.json'
};

export const SCALE_META = {
  hexaco: {
    id: 'hexaco',
    name: 'HEXACO 简化版',
    time: '约 12 分钟',
    theory: 'Lee & Ashton (2004)',
    likert: 5
  },
  ecr: {
    id: 'ecr',
    name: 'ECR-R 简化版',
    time: '约 4 分钟',
    theory: 'Brennan et al. (1998)',
    likert: 5
  },
  motivation: {
    id: 'motivation',
    name: '自我效能与动机',
    time: '约 6 分钟',
    theory: 'Bandura GSE + SDT',
    likert: 7
  },
  values: {
    id: 'values',
    name: 'Schwartz 价值观',
    time: '约 3 分钟',
    theory: 'Schwartz (2012) PVQ-10',
    likert: 7
  },
  situational: {
    id: 'situational',
    name: '情境化签名',
    time: '约 5 分钟',
    theory: 'CAPS',
    likert: 4
  }
};

export const FALLBACK_CHAPTERS = [
  {
    id: 'start',
    title: '第 1 章：测评设置',
    subtitle: '选择模式、模型供应商和本地保存策略。',
    scales: []
  },
  {
    id: 'hexaco',
    title: '第 2 章：人格基底',
    subtitle: '用 HEXACO 捕捉稳定特质。',
    scales: ['hexaco']
  },
  {
    id: 'attachment',
    title: '第 3 章：关系安全感',
    subtitle: '识别焦虑与回避两条依恋轴。',
    scales: ['ecr']
  },
  {
    id: 'agency',
    title: '第 4 章：行动效能',
    subtitle: '观察目标推进和自我效能。',
    scales: ['motivation']
  },
  {
    id: 'values',
    title: '第 5 章：价值排序',
    subtitle: '排序真实价值偏好，不使用假数据。',
    scales: ['values']
  },
  {
    id: 'situational',
    title: '第 6 章：情境反应',
    subtitle: '收集具体场景下的行为信号。',
    scales: ['situational']
  },
  {
    id: 'result',
    title: '第 7 章：画像与生成',
    subtitle: '汇总图表、导出画像，并交给 LLM 生成。',
    scales: []
  }
];

export const state = {
  view: 'home',
  mode: 'full',
  provider: 'openai',
  apiKey: '',
  currentChapter: 0,
  currentScale: 'hexaco',
  currentQ: 0,
  answers: {},
  debug: false,
  result: null,
  fullProfile: null,
  generated: [],
  questions: emptyQuestions(),
  chapters: FALLBACK_CHAPTERS
};

export function emptyQuestions() {
  return Object.fromEntries(Object.keys(QUESTION_FILES).map((id) => [id, []]));
}

export async function initState() {
  Object.assign(state.questions, await loadQuestions());
  state.chapters = await loadChapters();
  loadState();
  alignCurrentScale();
  saveState();
}

export async function loadQuestions() {
  const entries = await Promise.all(
    Object.entries(QUESTION_FILES).map(async ([id, url]) => [id, await fetchJson(url, [])])
  );
  return Object.fromEntries(entries);
}

export async function loadChapters() {
  const chapters = await fetchJson('data/chapters.json', null);
  if (!Array.isArray(chapters) || chapters.length === 0) {
    return FALLBACK_CHAPTERS;
  }
  return chapters;
}

export function loadState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {
    const saved = JSON.parse(raw);
    const safe = { ...saved };
    delete safe.questions;
    delete safe.chapters;
    Object.assign(state, safe);
  } catch (error) {
    console.warn('读取本地进度失败，已忽略。', error);
  }
}

export function saveState() {
  const { questions, chapters, ...serializable } = state;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
}

export function resetState() {
  const fresh = {
    view: 'home',
    mode: 'full',
    provider: 'openai',
    apiKey: '',
    currentChapter: 0,
    currentScale: 'hexaco',
    currentQ: 0,
    answers: {},
    debug: state.debug,
    result: null,
    fullProfile: null,
    generated: []
  };
  Object.assign(state, fresh);
  alignCurrentScale();
  saveState();
}

export function activeAssessmentChapters() {
  return state.chapters.filter((chapter) => {
    const scales = chapterScaleIds(chapter);
    if (state.mode === 'fast' && scales.includes('values')) return false;
    return chapterQuestions(chapter).length > 0;
  });
}

export function assessmentFlow() {
  return activeAssessmentChapters().map((chapter) => chapter.id);
}

export function currentChapter() {
  return activeAssessmentChapters()[state.currentChapter] || activeAssessmentChapters()[0];
}

export function findChapterByScale(scaleId) {
  return state.chapters.find((chapter) => chapter.scales?.includes(scaleId));
}

export function chapterQuestions(chapter) {
  if (!chapter) return [];
  if (Array.isArray(chapter.question_ids) && chapter.question_ids.length) {
    return chapter.question_ids.map(findQuestionById).filter(Boolean);
  }
  return (chapter.scales || []).flatMap((scale) => {
    return (state.questions[scale] || []).map((question) => ({ ...question, __scale: scale }));
  });
}

export function findQuestionById(questionId) {
  for (const [scale, questions] of Object.entries(state.questions)) {
    const question = questions.find((item) => item.id === questionId);
    if (question) return { ...question, __scale: scale };
  }
  return null;
}

export function setView(view) {
  state.view = view;
  saveState();
}

export function setAnswer(questionId, value) {
  state.answers[questionId] = value;
  saveState();
}

export function answeredCount() {
  return Object.keys(state.answers).length;
}

export function totalQuestionCount() {
  return activeAssessmentChapters().reduce((total, chapter) => total + chapterQuestions(chapter).length, 0);
}

export function alignCurrentScale() {
  const chapters = activeAssessmentChapters();
  state.currentChapter = Math.max(0, Math.min(state.currentChapter || 0, chapters.length - 1));
  const question = chapterQuestions(chapters[state.currentChapter])[state.currentQ];
  state.currentScale = question?.__scale || state.currentScale || 'hexaco';
}

async function fetchJson(url, fallback) {
  try {
    const response = await fetch(url);
    if (!response.ok) return fallback;
    return await response.json();
  } catch {
    return fallback;
  }
}

function chapterScaleIds(chapter) {
  if (Array.isArray(chapter.scales) && chapter.scales.length) return chapter.scales;
  if (!Array.isArray(chapter.question_ids)) return [];
  return [...new Set(chapter.question_ids.map(findQuestionById).filter(Boolean).map((q) => q.__scale))];
}
