import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const readJson = (file) => JSON.parse(fs.readFileSync(path.join(root, file), 'utf8'));
const exists = (file) => fs.existsSync(path.join(root, file));

const requiredProfileFields = [
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
  'metadata',
];

const jsFiles = exists('js')
  ? fs.readdirSync(path.join(root, 'js'))
      .filter((file) => file.endsWith('.js'))
      .map((file) => `js/${file}`)
  : [];

const checks = [];

function check(name, pass, detail) {
  checks.push({ name, pass, detail });
}

function flattenQuestions() {
  return [
    ...readJson('data/hexaco-questions.json'),
    ...readJson('data/ecr-questions.json'),
    ...readJson('data/motivation-questions.json'),
    ...readJson('data/values-questions.json'),
    ...readJson('data/situational-questions.json'),
  ];
}

function maxLineLength(file) {
  return fs.readFileSync(path.join(root, file), 'utf8')
    .split(/\r?\n/u)
    .reduce((max, line) => Math.max(max, line.length), 0);
}

const questions = flattenQuestions();
const questionIds = questions.map((question) => question.id);
const uniqueQuestionIds = new Set(questionIds);
const situationalCount = questions.filter((q) => {
  return q.type === 'situational' || q.type === 'narrative_likert';
}).length;

check('total_questions_eq_93', questions.length === 93, String(questions.length));
check('question_ids_unique', uniqueQuestionIds.size === questionIds.length, String(uniqueQuestionIds.size));
check('situational_or_narrative_ge_28', situationalCount >= 28, String(situationalCount));

const missingUserFields = questions.filter((q) => !q.subtitle || !q._why_for_user);
check('all_questions_have_user_context', missingUserFields.length === 0, `${missingUserFields.length} missing`);

if (exists('data/chapters.json')) {
  const chapters = readJson('data/chapters.json');
  const chapterQuestionTotal = chapters.reduce((sum, chapter) => sum + chapter.question_ids.length, 0);
  const chapterIds = chapters.flatMap((chapter) => chapter.question_ids);
  const uniqueChapterIds = new Set(chapterIds);
  const unknownChapterIds = chapterIds.filter((id) => !uniqueQuestionIds.has(id));
  const omittedQuestionIds = questionIds.filter((id) => !uniqueChapterIds.has(id));
  const invalidChapterSizes = chapters.filter((chapter) => {
    return chapter.question_ids.length < 11 || chapter.question_ids.length > 15;
  });

  check('chapters_eq_7', chapters.length === 7, String(chapters.length));
  check('chapter_question_total_eq_93', chapterQuestionTotal === 93, String(chapterQuestionTotal));
  check('chapter_question_ids_unique', uniqueChapterIds.size === chapterIds.length, String(uniqueChapterIds.size));
  check('chapter_question_ids_known', unknownChapterIds.length === 0, unknownChapterIds.join(', ') || 'ok');
  check('chapter_question_ids_complete', omittedQuestionIds.length === 0, omittedQuestionIds.join(', ') || 'ok');
  check('chapter_sizes_11_to_15', invalidChapterSizes.length === 0, `${invalidChapterSizes.length} invalid`);
} else {
  check('chapters_eq_7', false, 'data/chapters.json missing');
  check('chapter_question_total_eq_93', false, 'data/chapters.json missing');
  check('chapter_sizes_11_to_15', false, 'data/chapters.json missing');
}

const sourceFiles = [
  'persona-assessment.html',
  'index.html',
  'README.md',
  'prompts/analysis-prompt.txt',
  'prompts/generation-prompt.txt',
  'examples/sample-profile.json',
  ...jsFiles,
].filter(exists);

const allSource = sourceFiles
  .map((file) => fs.readFileSync(path.join(root, file), 'utf8'))
  .join('\n');

check('no_hardcoded_values_chart', !allSource.includes('data:[1,.9,.4,.3]'), 'grep literal');
check('llm_client_exists', exists('js/llm-client.js'), 'js/llm-client.js');
check('batch_client_exists', exists('js/batch.js'), 'js/batch.js');
check('llm_client_uses_fetch', /\bfetch\s*\(/u.test(readFile('js/llm-client.js')), 'fetch(');
check('llm_client_supports_minimax', /MiniMax-M2\.7/u.test(readFile('js/llm-client.js')), 'MiniMax-M2.7');
check('analysis_uses_call_llm', /callLLM\s*\(/u.test(readFile('js/export.js')), 'export.js callLLM');
check('batch_uses_call_llm', /callLLM\s*\(/u.test(readFile('js/batch.js')), 'batch.js callLLM');
check('modular_js_file_count_ge_6', jsFiles.length >= 6, String(jsFiles.length));

const overlong = sourceFiles
  .map((file) => ({ file, max: maxLineLength(file) }))
  .filter((item) => item.max > 200);

check('source_lines_lte_200', overlong.length === 0, JSON.stringify(overlong));

if (exists('examples/sample-profile.json')) {
  const profileRaw = fs.readFileSync(path.join(root, 'examples/sample-profile.json'), 'utf8');
  const profile = JSON.parse(profileRaw);
  const missing = requiredProfileFields.filter((field) => !(field in profile));
  const patterns = profile.situational_signatures?.if_then_patterns ?? [];

  check('sample_profile_ge_8kb', Buffer.byteLength(profileRaw) >= 8192, `${Buffer.byteLength(profileRaw)} bytes`);
  check('sample_profile_required_fields', missing.length === 0, missing.join(', ') || 'ok');
  check('sample_profile_patterns_ge_5', patterns.length >= 5, String(patterns.length));
}

const readmeLines = exists('README.md')
  ? fs.readFileSync(path.join(root, 'README.md'), 'utf8').split(/\r?\n/u).length
  : 0;

check('readme_ge_80_lines', readmeLines >= 80, String(readmeLines));

const failed = checks.filter((item) => !item.pass);

for (const item of checks) {
  const mark = item.pass ? 'PASS' : 'FAIL';
  console.log(`${mark} ${item.name}: ${item.detail}`);
}

if (failed.length > 0) {
  process.exitCode = 1;
}

function readFile(file) {
  return exists(file) ? fs.readFileSync(path.join(root, file), 'utf8') : '';
}
