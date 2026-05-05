export const HEXACO_DIMENSIONS = [
  'honesty_humility',
  'emotionality',
  'extraversion',
  'agreeableness',
  'conscientiousness',
  'openness'
];

export const REQUIRED_PROFILE_FIELDS = [
  'profile_id',
  'version',
  'created_date',
  'is_real_person',
  'trait_layer',
  'attachment_layer',
  'motivation_layer',
  'situational_signatures',
  'multimodal_signature',
  'ai_interaction_design',
  'narrative_layer',
  'raw_assessment_data',
  'metadata'
];

export const ANALYSIS_SYSTEM = [
  '你是一位严谨的人格心理学分析师和 AI 交互设计师。',
  '你的任务是把结构化测评分数与题目答案整合为可被产品直接使用的人格画像 JSON。',
  '请坚持以下原则：分数层只解释，不篡改。',
  '所有推论都必须能回到 HEXACO、ECR、GSE/SDT/PVQ 或情境题证据。',
  '不做临床诊断，不使用病理化标签。',
  '输出必须是严格 JSON，不要 Markdown、注释或额外解释。'
].join('\n');

function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function safeParseJson(value) {
  if (isPlainObject(value)) return value;
  if (typeof value !== 'string') return {};
  const trimmed = value
    .trim()
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/i, '')
    .replace(/```$/i, '')
    .trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    return {};
  }
}

function clamp01(value, fallback = 0.5) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(1, Math.max(0, number));
}

function normalizeHexaco(source = {}) {
  const candidate = source.hexaco_scores
    || source.hexaco
    || source.hex
    || source.trait_layer?.hexaco_scores
    || {};
  return Object.fromEntries(
    HEXACO_DIMENSIONS.map((dimension) => [dimension, clamp01(candidate[dimension])])
  );
}

function normalizeAttachment(source = {}) {
  const candidate = source.attachment_dimensions
    || source.attachment
    || source.attachment_layer?.attachment_dimensions
    || {};
  const anxiety = clamp01(candidate.anxiety);
  const avoidance = clamp01(candidate.avoidance);
  let primary = source.attachment_style?.primary || source.attachment_layer?.attachment_style?.primary;

  if (!primary) {
    if (anxiety < 0.45 && avoidance < 0.45) primary = 'secure';
    else if (anxiety >= 0.45 && avoidance < 0.45) primary = 'anxious_preoccupied';
    else if (anxiety < 0.45 && avoidance >= 0.45) primary = 'dismissive_avoidant';
    else primary = 'fearful_avoidant';
  }

  return {
    attachment_dimensions: { anxiety, avoidance },
    attachment_style: {
      primary,
      secondary: source.attachment_style?.secondary || null,
      confidence: clamp01(
        source.attachment_style?.confidence
        ?? source.attachment_layer?.attachment_style?.confidence
        ?? 0.72
      )
    }
  };
}

function normalizeMotivation(source = {}) {
  const candidate = source.motivation_layer || source.motivation || {};
  const selfEfficacy = candidate.self_efficacy || { general: source.gse ?? candidate.gse ?? 0.5 };
  const sdt = candidate.self_determination || candidate.sdt || {};
  const values = candidate.values_hierarchy || candidate.values || {};

  return {
    _theory_basis: candidate._theory_basis || 'GSE + Self-Determination Theory + Schwartz PVQ',
    self_efficacy: {
      general: clamp01(selfEfficacy.general),
      professional: clamp01(selfEfficacy.professional ?? selfEfficacy.domain_professional),
      relationship: clamp01(selfEfficacy.relationship ?? selfEfficacy.domain_relationship),
      emotion_regulation: clamp01(selfEfficacy.emotion_regulation ?? selfEfficacy.domain_emotion),
      health_management: clamp01(selfEfficacy.health_management ?? selfEfficacy.domain_health),
      creative_expression: clamp01(selfEfficacy.creative_expression ?? selfEfficacy.domain_creativity)
    },
    self_determination: {
      autonomy: clamp01(sdt.autonomy),
      competence: clamp01(sdt.competence),
      relatedness: clamp01(sdt.relatedness)
    },
    values_hierarchy: {
      top_values: Array.isArray(values.top_values)
        ? values.top_values
        : (Array.isArray(candidate.top_values) ? candidate.top_values : []),
      low_values: Array.isArray(values.low_values)
        ? values.low_values
        : (Array.isArray(candidate.low_values) ? candidate.low_values : []),
      ranked_values: Array.isArray(values.ranked_values) ? values.ranked_values : []
    }
  };
}

function defaultSituationalSignatures() {
  return {
    _theory_basis: 'CAPS cognitive-affective processing system',
    if_then_patterns: [],
    stress_triggers: [],
    recovery_conditions: [],
    collaboration_mode: 'insufficient_data'
  };
}

function defaultMultimodalSignature() {
  return {
    visual_cues: [],
    voice_cues: [],
    behavioral_cues: [],
    digital_behavior_cues: [],
    confidence: 0.3
  };
}

function defaultAiDesign() {
  return {
    interaction_principles: [],
    preferred_tone: 'direct_supportive',
    proactivity_level: 'moderate',
    trust_building_path: [],
    do: [],
    avoid: [],
    memory_strategy: []
  };
}

function defaultNarrative() {
  return {
    summary_second_person: '',
    core_tension: '',
    growth_direction: '',
    relationship_to_ai: '',
    evidence_notes: []
  };
}

function mergeObject(defaults, value) {
  return isPlainObject(value) ? { ...defaults, ...value } : { ...defaults };
}

export function buildAnalysisPrompt(answers, scoreResult) {
  const payload = {
    score_result: scoreResult || {},
    answers: answers || {}
  };

  const schema = {
    trait_layer: {
      hexaco_interpretation: {},
      facet_highlights: [],
      trait_risks: [],
      trait_strengths: []
    },
    attachment_layer: {
      relationship_pattern: '',
      needs_in_close_relationships: [],
      risk_under_stress: []
    },
    motivation_layer: {
      drive_summary: '',
      value_conflicts: [],
      activation_conditions: []
    },
    situational_signatures: {
      if_then_patterns: ['至少 5 条'],
      stress_triggers: [],
      recovery_conditions: [],
      collaboration_mode: ''
    },
    multimodal_signature: {
      visual_cues: [],
      voice_cues: [],
      behavioral_cues: [],
      digital_behavior_cues: [],
      confidence: '0 到 1'
    },
    ai_interaction_design: {
      interaction_principles: [],
      preferred_tone: '',
      proactivity_level: '',
      trust_building_path: [],
      do: [],
      avoid: [],
      memory_strategy: []
    },
    narrative_layer: {
      summary_second_person: '300 字内',
      core_tension: '',
      growth_direction: '',
      relationship_to_ai: '',
      evidence_notes: []
    },
    metadata: {
      llm_confidence: '0 到 1',
      analysis_notes: []
    }
  };

  return [
    '请基于以下测评分数与原始答案，生成完整 v0.2 人格画像分析 JSON。',
    '',
    '输入数据：',
    JSON.stringify(payload, null, 2),
    '',
    '输出必须是单个 JSON 对象，并尽量包含以下字段：',
    JSON.stringify(schema, null, 2),
    '',
    '硬性要求：只输出 JSON；不要修改已给出的量表分数。',
    'if_then_patterns 每条包含 id、if_situation、then_response 和 supporting_evidence。',
    'then_response 必须包含 behavioral、emotional、cognitive。'
  ].join('\n');
}

export function mergeProfile(scoreResult = {}, analysisJson = {}) {
  const analysis = safeParseJson(analysisJson);
  const now = new Date().toISOString().slice(0, 10);
  const hexaco = normalizeHexaco(scoreResult);
  const attachment = normalizeAttachment(scoreResult);
  const motivation = normalizeMotivation(scoreResult);

  return {
    profile_id: analysis.profile_id || scoreResult.profile_id || `profile_${Date.now()}`,
    version: analysis.version || scoreResult.version || '0.2',
    created_date: analysis.created_date || scoreResult.created_date || now,
    is_real_person: Boolean(analysis.is_real_person ?? scoreResult.is_real_person ?? true),
    trait_layer: {
      _theory_basis: 'HEXACO six-factor personality model',
      hexaco_scores: hexaco,
      facet_scores: scoreResult.facet_scores
        || scoreResult.facets
        || scoreResult.trait_layer?.facet_scores
        || {},
      ...(isPlainObject(analysis.trait_layer) ? analysis.trait_layer : {}),
      hexaco_scores: hexaco
    },
    attachment_layer: {
      _theory_basis: 'ECR attachment dimensions',
      ...attachment,
      ...(isPlainObject(analysis.attachment_layer) ? analysis.attachment_layer : {}),
      attachment_dimensions: attachment.attachment_dimensions,
      attachment_style: analysis.attachment_layer?.attachment_style || attachment.attachment_style
    },
    motivation_layer: {
      ...motivation,
      ...(isPlainObject(analysis.motivation_layer) ? analysis.motivation_layer : {})
    },
    situational_signatures: mergeObject(defaultSituationalSignatures(), analysis.situational_signatures),
    multimodal_signature: mergeObject(defaultMultimodalSignature(), analysis.multimodal_signature),
    ai_interaction_design: mergeObject(defaultAiDesign(), analysis.ai_interaction_design),
    narrative_layer: mergeObject(defaultNarrative(), analysis.narrative_layer),
    raw_assessment_data: analysis.raw_assessment_data || scoreResult.raw_assessment_data || scoreResult.answers || {},
    metadata: {
      assessment_version: 'v0.2',
      generated_by: 'personalt-test',
      confidence_score: clamp01(
        analysis.metadata?.llm_confidence
        ?? analysis.metadata?.confidence_score
        ?? scoreResult.confidence_score
        ?? 0.7
      ),
      ...(isPlainObject(scoreResult.metadata) ? scoreResult.metadata : {}),
      ...(isPlainObject(analysis.metadata) ? analysis.metadata : {})
    }
  };
}

export function profileHasRequiredSchema(profile) {
  if (!isPlainObject(profile)) return false;
  if (!REQUIRED_PROFILE_FIELDS.every((field) => Object.prototype.hasOwnProperty.call(profile, field))) return false;
  if (typeof profile.profile_id !== 'string' || profile.profile_id.trim() === '') return false;
  if (typeof profile.version !== 'string' || profile.version.trim() === '') return false;
  if (typeof profile.created_date !== 'string' || !/^\d{4}-\d{2}-\d{2}/.test(profile.created_date)) return false;
  if (typeof profile.is_real_person !== 'boolean') return false;

  const hexaco = profile.trait_layer?.hexaco_scores;
  if (!isPlainObject(hexaco)) return false;
  if (!HEXACO_DIMENSIONS.every((dimension) => Number.isFinite(Number(hexaco[dimension])))) return false;

  const attachment = profile.attachment_layer?.attachment_dimensions;
  if (!isPlainObject(attachment)) return false;
  if (!Number.isFinite(Number(attachment.anxiety)) || !Number.isFinite(Number(attachment.avoidance))) return false;

  if (!Array.isArray(profile.situational_signatures?.if_then_patterns)) return false;
  if (!isPlainObject(profile.multimodal_signature)) return false;
  if (!isPlainObject(profile.ai_interaction_design)) return false;
  if (!isPlainObject(profile.narrative_layer)) return false;
  if (!isPlainObject(profile.metadata)) return false;

  return true;
}
