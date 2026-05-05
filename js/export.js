import { callLLM } from './llm-client.js';
import { ANALYSIS_SYSTEM, buildAnalysisPrompt, mergeProfile } from './profile.js';

export function buildProfile(state) {
  if (state.fullProfile) return state.fullProfile;
  return mergeProfile(localProfileSeed(state));
}

export async function analyzeProfileWithLLM(state, runtime = {}) {
  if (!state.provider || !state.apiKey) {
    throw new Error('请先在首页填写 provider 和 API Key。');
  }

  const { signal, onStatus } = runtime;
  const provider = String(state.provider || '').toLowerCase();
  const seed = localProfileSeed(state);
  const prompt = buildAnalysisPrompt(state.answers, seed);
  const providerModel = provider === 'minimax' ? 'MiniMax-M2.7-highspeed' : undefined;
  const providerMaxTokens = provider === 'minimax' ? 2200 : 1800;
  const base = {
    responseFormat: { type: 'json_object' },
    temperature: 0.4,
    maxTokens: providerMaxTokens,
    model: providerModel,
    signal
  };

  let response;
  try {
    onStatus?.('调用 LLM 分析中...（第 1 次请求，最长约 45 秒）');
    response = await callLLM(state.provider, state.apiKey, ANALYSIS_SYSTEM, prompt, {
      ...base,
      timeoutMs: 45000
    });
  } catch (error) {
    if ((!isTimeoutError(error) && !isNoTextError(error)) || signal?.aborted) throw error;
    onStatus?.('首轮结果不完整，正在重试...（第 2 次请求，最长约 90 秒）');
    response = await callLLM(state.provider, state.apiKey, ANALYSIS_SYSTEM, prompt, {
      ...base,
      timeoutMs: 90000,
      maxTokens: provider === 'minimax' ? 2800 : providerMaxTokens,
      model: provider === 'minimax' ? 'MiniMax-M2.7' : providerModel,
      temperature: 0.25
    });
  }

  return mergeProfile(seed, response.text);
}

function localProfileSeed(state) {
  return {
    ...(state.result || {}),
    profile_id: 'boss_self',
    version: '0.2',
    is_real_person: true,
    raw_assessment_data: state.answers,
    metadata: {
      assessment_version: state.mode,
      provider: state.provider,
      analysis_status: 'local_only'
    }
  };
}

function isTimeoutError(error) {
  const message = String(error?.message || '');
  return /timed out|timeout/i.test(message);
}

function isNoTextError(error) {
  const message = String(error?.message || '');
  return /did not contain text content|no text/i.test(message);
}

export function downloadJson(name, payload) {
  const text = JSON.stringify(payload, null, 2);
  const blob = new Blob([text], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  try {
    downloadUrl(name, url);
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}

const DIMENSION_LABELS = {
  honesty_humility: '诚实谦逊',
  emotionality: '情绪敏感',
  extraversion: '外向表达',
  agreeableness: '宜人合作',
  conscientiousness: '尽责执行',
  openness: '开放探索'
};

const ATTACHMENT_STYLE_LABELS = {
  secure: '安全型',
  anxious_preoccupied: '焦虑型',
  dismissive_avoidant: '回避型',
  fearful_avoidant: '恐惧-回避型',
  安全型: '安全型',
  焦虑型: '焦虑型',
  回避型: '回避型',
  混乱型: '混乱型'
};

const VALUE_LABELS = {
  self_direction: '自我导向',
  stimulation: '刺激冒险',
  hedonism: '享乐体验',
  achievement: '成就驱动',
  power: '权力影响',
  security: '安全稳定',
  conformity: '规范一致',
  tradition: '传统秩序',
  benevolence: '仁慈关怀',
  universalism: '普遍主义'
};

export function buildChineseAnalysisView(profile = {}) {
  const hexaco = profile.trait_layer?.hexaco_scores || {};
  const rankedTraits = Object.entries(hexaco)
    .filter(([, score]) => Number.isFinite(Number(score)))
    .sort((a, b) => Number(b[1]) - Number(a[1]));

  const topTraits = rankedTraits.slice(0, 2).map(([id, score]) => {
    return `${DIMENSION_LABELS[id] || id}（${toPercent(score)}）`;
  });
  const lowTraits = rankedTraits.slice(-2).map(([id, score]) => {
    return `${DIMENSION_LABELS[id] || id}（${toPercent(score)}）`;
  });

  const attachmentDims = profile.attachment_layer?.attachment_dimensions || {};
  const attachmentStyle = profile.attachment_layer?.attachment_style?.primary;
  const attachmentZh = ATTACHMENT_STYLE_LABELS[attachmentStyle] || attachmentStyle || '未识别';

  const values = profile.motivation_layer?.values_hierarchy?.ranked_values || [];
  const topValues = values
    .slice(0, 3)
    .map((item) => {
      const key = item?.value || item?.id || '';
      return VALUE_LABELS[key] || key;
    })
    .filter(Boolean);

  const summary = profile.narrative_layer?.summary_second_person?.trim()
    || buildFallbackSummary(topTraits, attachmentZh);

  return [
    '【中文画像解读】',
    '',
    `1. 核心特征：${topTraits.join('、') || '数据不足'}。`,
    `2. 相对弱项：${lowTraits.join('、') || '数据不足'}。`,
    `3. 依恋关系：${attachmentZh}（焦虑 ${toPercent(attachmentDims.anxiety)} / 回避 ${toPercent(attachmentDims.avoidance)}）。`,
    `4. 价值倾向：${topValues.length ? topValues.join('、') : '暂未形成稳定排序'}。`,
    `5. 行为建议：${buildBehaviorSuggestion(profile, topTraits, attachmentZh)}`,
    '',
    '【个性化总结】',
    summary
  ].join('\n');
}

function buildBehaviorSuggestion(profile, topTraits, attachmentZh) {
  const efficacy = Number(profile.motivation_layer?.self_efficacy?.general ?? 0.5);
  const efficacyHint = efficacy >= 0.65 ? '你有较强的推进能力' : '你在推进任务时更需要外部节奏';
  const traitHint = topTraits[0] ? `你的优势维度是${topTraits[0]}` : '当前优势维度尚不明确';
  const attachmentHint = attachmentZh === '回避型'
    ? '在高压协作中建议提前声明边界与反馈频率'
    : '在关系协作中建议保持稳定反馈与情绪标注';
  return `${traitHint}；${efficacyHint}；${attachmentHint}。`;
}

function buildFallbackSummary(topTraits, attachmentZh) {
  const first = topTraits[0] || '你的稳定特质';
  return `你在测评中呈现出以${first}为核心的风格，在关系上偏向${attachmentZh}。建议把优势用于关键任务，同时通过固定复盘和明确沟通降低协作摩擦。`;
}

function toPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '0%';
  return `${Math.round(number * 100)}%`;
}

export function downloadUrl(name, url) {
  const link = document.createElement('a');
  link.download = name;
  link.href = url;
  link.rel = 'noopener';
  link.style.display = 'none';
  document.body.appendChild(link);
  try {
    link.click();
  } catch {
    window.open(url, '_blank', 'noopener');
  } finally {
    document.body.removeChild(link);
  }
}

export async function exportScreenshot(target, name = 'persona-report.png') {
  if (!window.html2canvas) throw new Error('html2canvas 未加载');
  const canvas = await window.html2canvas(target);
  downloadUrl(name, canvas.toDataURL('image/png'));
}

export async function generateProfilesWithLLM({ state, prompt, count }) {
  const profile = buildProfile(state);
  const payload = {
    provider: state.provider,
    apiKey: state.apiKey,
    prompt,
    userPrompt: prompt,
    count,
    anchorProfile: profile,
    profile,
    result: state.result
  };

  const batch = await import('./batch.js');
  return batch.generateBatch(payload);
}
