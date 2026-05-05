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
  const seed = localProfileSeed(state);
  const prompt = buildAnalysisPrompt(state.answers, seed);
  const base = {
    responseFormat: { type: 'json_object' },
    temperature: 0.4,
    maxTokens: 2200,
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
    if (!isTimeoutError(error) || signal?.aborted) throw error;
    onStatus?.('首轮超时，正在重试...（第 2 次请求，最长约 60 秒）');
    response = await callLLM(state.provider, state.apiKey, ANALYSIS_SYSTEM, prompt, {
      ...base,
      timeoutMs: 60000,
      temperature: 0.3
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
