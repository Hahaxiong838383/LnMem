import { callLLM } from './llm-client.js';
import { ANALYSIS_SYSTEM, buildAnalysisPrompt, mergeProfile } from './profile.js';

export function buildProfile(state) {
  if (state.fullProfile) return state.fullProfile;
  return mergeProfile(localProfileSeed(state));
}

export async function analyzeProfileWithLLM(state) {
  if (!state.provider || !state.apiKey) {
    throw new Error('请先在首页填写 provider 和 API Key。');
  }

  const seed = localProfileSeed(state);
  const prompt = buildAnalysisPrompt(state.answers, seed);
  const response = await callLLM(state.provider, state.apiKey, ANALYSIS_SYSTEM, prompt, {
    responseFormat: { type: 'json_object' },
    temperature: 0.4
  });

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

export function downloadJson(name, payload) {
  const text = JSON.stringify(payload, null, 2);
  downloadUrl(name, `data:application/json;charset=utf-8,${encodeURIComponent(text)}`);
}

export function downloadUrl(name, url) {
  const link = document.createElement('a');
  link.download = name;
  link.href = url;
  link.click();
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
