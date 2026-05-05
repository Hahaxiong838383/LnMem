import { callLLM } from './llm-client.js';
import { HEXACO_DIMENSIONS, profileHasRequiredSchema } from './profile.js';

const DEFAULT_BATCH_SYSTEM = [
  '你是一位人格心理学专家，负责生成虚构但心理结构自洽的人格画像。',
  '输出必须是严格 JSON 数组，不要 Markdown，不要解释。',
  '每个 profile 必须满足 v0.2 schema，并且所有 HEXACO 分数在 0 到 1 之间。'
].join('\n');

function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function extractJson(text) {
  const cleaned = String(text || '')
    .trim()
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/i, '')
    .replace(/```$/i, '')
    .trim();
  try {
    return JSON.parse(cleaned);
  } catch {
    const startArray = cleaned.indexOf('[');
    const endArray = cleaned.lastIndexOf(']');
    if (startArray !== -1 && endArray > startArray) {
      return JSON.parse(cleaned.slice(startArray, endArray + 1));
    }
    const startObject = cleaned.indexOf('{');
    const endObject = cleaned.lastIndexOf('}');
    if (startObject !== -1 && endObject > startObject) {
      return JSON.parse(cleaned.slice(startObject, endObject + 1));
    }
    throw new Error('LLM response did not contain valid JSON');
  }
}

function buildBatchPrompt(ctx) {
  const count = Number(ctx.count ?? ctx.N ?? 10);
  const anchorProfile = ctx.anchorProfile || ctx.anchor_profile_json || null;
  const userPrompt = ctx.userPrompt || ctx.prompt || '生成结构差异明显、职业和生活阶段多样的人格画像。';

  return [
    `请生成 ${count} 个虚构人物的人格画像 JSON 数组。`,
    '',
    '参考真实锚点画像：',
    JSON.stringify(anchorProfile, null, 2),
    '',
    '用户要求：',
    userPrompt,
    '',
    '硬性 schema：每个对象一级字段必须包含 profile_id、version、created_date、is_real_person、',
    'trait_layer、attachment_layer、motivation_layer、situational_signatures、multimodal_signature、',
    'ai_interaction_design、narrative_layer、raw_assessment_data、metadata。',
    '',
    '质量约束：',
    '1. is_real_person 必须为 false。',
    '2. trait_layer.hexaco_scores 必须包含 6 个 HEXACO 维度，且每个值在 0 到 1。',
    '3. situational_signatures.if_then_patterns 至少 5 条，每条要有 if_situation 与 then_response。',
    '4. 批次内部 HEXACO 两两 cosine distance 必须大于 0.2，请主动拉开分布。',
    '5. attachment_style、价值观、AI 互动偏好、叙事层必须随特质差异变化，不能模板化。',
    '6. 输出只能是 JSON 数组，不要包裹在对象里，不要 Markdown。'
  ].join('\n');
}

function hexacoVector(profile) {
  const scores = profile?.trait_layer?.hexaco_scores || {};
  return HEXACO_DIMENSIONS.map((dimension) => Number(scores[dimension]));
}

function inUnitInterval(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 && number <= 1;
}

function cosineDistance(a, b) {
  const dot = a.reduce((sum, value, index) => sum + value * b[index], 0);
  const normA = Math.sqrt(a.reduce((sum, value) => sum + value * value, 0));
  const normB = Math.sqrt(b.reduce((sum, value) => sum + value * value, 0));
  if (normA === 0 || normB === 0) return 1;
  return 1 - dot / (normA * normB);
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function stddev(values) {
  const avg = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - avg) ** 2)));
}

function distributionStats(profiles) {
  const byDimension = {};
  for (const dimension of HEXACO_DIMENSIONS) {
    const values = profiles
      .map((profile) => Number(profile.trait_layer?.hexaco_scores?.[dimension]))
      .filter(Number.isFinite);
    byDimension[dimension] = {
      mean: Number(mean(values).toFixed(4)),
      min: values.length ? Math.min(...values) : null,
      max: values.length ? Math.max(...values) : null,
      range: values.length ? Number((Math.max(...values) - Math.min(...values)).toFixed(4)) : null,
      stddev: Number(stddev(values).toFixed(4))
    };
  }

  const attachmentStyles = profiles
    .map((profile) => profile.attachment_layer?.attachment_style?.primary)
    .filter(Boolean);
  return {
    count: profiles.length,
    hexaco: byDimension,
    attachment_style_count: new Set(attachmentStyles).size,
    unique_profile_ids: new Set(profiles.map((profile) => profile.profile_id)).size
  };
}

export async function generateBatch(ctx = {}) {
  const provider = ctx.provider;
  const apiKey = ctx.apiKey;
  if (!provider || !apiKey) {
    throw new Error('generateBatch requires ctx.provider and ctx.apiKey for a real LLM call');
  }

  const systemPrompt = ctx.systemPrompt || DEFAULT_BATCH_SYSTEM;
  const userPrompt = ctx.userPromptText || buildBatchPrompt(ctx);
  const result = await callLLM(provider, apiKey, systemPrompt, userPrompt, ctx.options || {});
  const parsed = extractJson(result.text);
  const profiles = Array.isArray(parsed) ? parsed : (Array.isArray(parsed.profiles) ? parsed.profiles : []);

  if (!profiles.length) {
    throw new Error('LLM response did not contain a profile array');
  }

  const validation = validateBatch(profiles);
  if (ctx.strict && !validation.valid) {
    throw new Error(`Generated batch failed validation: ${validation.issues.join('; ')}`);
  }

  return {
    profiles,
    validation,
    raw: result
  };
}

export function validateBatch(profiles) {
  const issues = [];
  const warnings = [];
  const distances = [];

  if (!Array.isArray(profiles)) {
    return { valid: false, issues: ['batch must be an array'], warnings, stats: null, distances };
  }

  if (profiles.length === 0) {
    issues.push('batch must contain at least one profile');
  }

  const ids = new Set();
  profiles.forEach((profile, index) => {
    const label = profile?.profile_id || `index_${index}`;
    if (!profileHasRequiredSchema(profile)) {
      issues.push(`${label}: missing required v0.2 schema fields`);
    }

    if (ids.has(profile?.profile_id)) issues.push(`${label}: duplicate profile_id`);
    if (profile?.profile_id) ids.add(profile.profile_id);

    const vector = hexacoVector(profile);
    vector.forEach((value, dimensionIndex) => {
      if (!inUnitInterval(value)) {
        issues.push(`${label}: HEXACO ${HEXACO_DIMENSIONS[dimensionIndex]} must be in [0,1]`);
      }
    });

    const patterns = profile?.situational_signatures?.if_then_patterns;
    if (!Array.isArray(patterns) || patterns.length < 5) {
      issues.push(`${label}: situational_signatures.if_then_patterns must contain at least 5 items`);
    }
  });

  for (let i = 0; i < profiles.length; i += 1) {
    for (let j = i + 1; j < profiles.length; j += 1) {
      const left = hexacoVector(profiles[i]);
      const right = hexacoVector(profiles[j]);
      if (left.every(Number.isFinite) && right.every(Number.isFinite)) {
        const distance = cosineDistance(left, right);
        const record = {
          pair: [profiles[i].profile_id || i, profiles[j].profile_id || j],
          cosine_distance: Number(distance.toFixed(4))
        };
        distances.push(record);
        if (distance <= 0.2) {
          issues.push(
            `${record.pair[0]} vs ${record.pair[1]}: HEXACO cosine distance `
            + `${record.cosine_distance} must be > 0.2`
          );
        }
      }
    }
  }

  const stats = distributionStats(profiles.filter(isPlainObject));
  if (stats.unique_profile_ids !== profiles.length) {
    issues.push('profile_id values must be unique');
  }

  if (profiles.length >= 4 && stats.attachment_style_count < 2) {
    warnings.push('basic distribution: attachment styles should include at least 2 primary styles for batches of 4+');
  }

  if (profiles.length >= 5) {
    for (const [dimension, dimensionStats] of Object.entries(stats.hexaco)) {
      if (dimensionStats.range !== null && dimensionStats.range < 0.15) {
        warnings.push(`basic distribution: ${dimension} range ${dimensionStats.range} is narrow; prefer >= 0.15`);
      }
      if (dimensionStats.mean < 0.2 || dimensionStats.mean > 0.8) {
        warnings.push(`basic distribution: ${dimension} mean ${dimensionStats.mean} is extreme; prefer 0.2-0.8`);
      }
    }
  }

  return {
    valid: issues.length === 0,
    issues,
    warnings,
    stats,
    distances
  };
}
