import {
  SCALE_META,
  activeAssessmentChapters,
  alignCurrentScale,
  answeredCount,
  chapterQuestions,
  makeOptionOrderSeed,
  resetState,
  saveState,
  setAnswer,
  state,
  totalQuestionCount
} from './state.js';
import { scoreAssessment } from './scoring.js';
import { renderHexChart, renderValuesChart } from './charts.js';
import {
  analyzeProfileWithLLM,
  buildProfile,
  downloadJson,
  exportScreenshot,
  generateProfilesWithLLM
} from './export.js';

const shell = 'rounded-3xl border border-white/10 bg-white/10 p-6 shadow-2xl backdrop-blur';
const soft = 'rounded-2xl border border-white/10 bg-slate-900/70 p-4';
const primary = 'rounded-xl bg-cyan-400 px-4 py-2 font-semibold text-slate-950 hover:bg-cyan-300';
const ghost = 'rounded-xl border border-white/15 px-4 py-2 text-slate-100 hover:bg-white/10';

export function renderApp() {
  const app = document.getElementById('app');
  if (!app) return;
  app.innerHTML = frame();
  const main = app.querySelector('[data-main]');
  main.innerHTML = viewHtml();
  bindGlobal(app);
  bindView(main);
}

function frame() {
  return html`
    <header class="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <p class="text-xs uppercase tracking-[0.35em] text-cyan-300">LnMem Persona Lab</p>
        <h1 class="mt-2 text-3xl font-black text-white sm:text-5xl">人格画像测评</h1>
      </div>
      <div class="flex gap-2">
        <button data-action="home" class="${ghost}">首页</button>
        <button data-action="reset" class="${ghost}">重置</button>
        <button data-action="debug" class="${ghost}">Debug</button>
      </div>
    </header>
    <main data-main></main>
  `;
}

function viewHtml() {
  if (state.view === 'assessment') return assessmentHtml();
  if (state.view === 'chapter-summary') return chapterSummaryHtml();
  if (state.view === 'result') return resultHtml();
  if (state.view === 'generate') return generateHtml();
  return homeHtml();
}

function homeHtml() {
  const chapters = state.chapters.map(chapterCard).join('');
  const hasAnswers = answeredCount() > 0;
  const hasResult = Boolean(state.result);
  return html`
    <section class="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
      <div class="${shell}">
        <p class="text-sm text-slate-300">7 章结构化测评，本地保存进度，结果可导出 JSON 和截图。</p>
        <div class="mt-6 grid gap-3 sm:grid-cols-2">
          <label class="grid gap-2 text-sm text-slate-300">
            <span>LLM Provider</span>
            <select id="provider" class="rounded-xl border border-white/10 bg-slate-950 p-3 text-white">
              ${option('openai', state.provider)}
              ${option('anthropic', state.provider)}
              ${option('deepseek', state.provider)}
              ${option('minimax', state.provider)}
            </select>
          </label>
          <label class="grid gap-2 text-sm text-slate-300">
            <span>API Key</span>
            <input id="apiKey" type="password" value="${escapeAttr(state.apiKey)}"
              class="rounded-xl border border-white/10 bg-slate-950 p-3 text-white" placeholder="仅当前页面内存，不持久化">
          </label>
        </div>
        <div class="mt-5 flex flex-wrap gap-3 text-sm">
          ${modeRadio('full', '完整版：覆盖全部章节')}
          ${modeRadio('fast', '快速版：跳过价值观章节')}
        </div>
        <div class="mt-6 flex flex-wrap gap-3">
          <button data-action="start" class="${primary}">开始新测评</button>
          <button data-action="save-config" class="${ghost}">保存 LLM 设置</button>
          ${hasResult ? `<button data-action="go-result" class="${ghost}">返回测评结果</button>` : ''}
          ${hasAnswers && !hasResult ? `<button data-action="go-assessment" class="${ghost}">继续未完成测评</button>` : ''}
        </div>
      </div>
      <div class="grid gap-3">${chapters}</div>
    </section>
  `;
}

function chapterCard(chapter, index) {
  return html`
    <article class="${soft}">
      <div class="flex items-start gap-3">
        <span class="grid h-9 w-9 place-items-center rounded-full bg-cyan-300 text-sm font-black text-slate-950">
          ${index + 1}
        </span>
        <div>
          <h2 class="font-bold text-white">${escapeHtml(chapter.title || chapter.id)}</h2>
          <p class="mt-1 text-sm text-slate-300">${escapeHtml(chapter.subtitle || '')}</p>
        </div>
      </div>
    </article>
  `;
}

function assessmentHtml() {
  alignCurrentScale();
  const chapters = activeAssessmentChapters();
  const chapter = chapters[state.currentChapter];
  const questions = chapterQuestions(chapter);
  const question = questions[state.currentQ];
  if (!question) return finishChapter();

  state.currentScale = question.__scale || state.currentScale;
  const meta = SCALE_META[state.currentScale] || { name: state.currentScale, likert: 5 };
  const done = answeredCount();
  const total = totalQuestionCount();
  const progress = total ? Math.round((done / total) * 100) : 0;
  const currentChapterIndex = state.currentChapter + 1;

  return html`
    <section class="${shell}">
      <div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p class="text-sm text-cyan-300">${escapeHtml(chapter.title || meta.name)}</p>
          <h2 class="mt-1 text-2xl font-black text-white">${escapeHtml(meta.name)}</h2>
          <p class="mt-1 text-sm text-slate-300">${escapeHtml(meta.theory || '')} · ${escapeHtml(meta.time || '')}</p>
        </div>
        <p class="rounded-full bg-white/10 px-3 py-1 text-sm text-slate-200">
          ${currentChapterIndex}/${chapters.length} · ${state.currentQ + 1}/${questions.length}
        </p>
      </div>
      <div class="mt-5 h-3 rounded-full bg-slate-800">
        <div class="h-3 rounded-full bg-cyan-300" style="width:${progress}%"></div>
      </div>
      <article class="mt-6 rounded-3xl bg-slate-950/80 p-5">
        ${questionTitle(question)}
        ${answerControl(question, meta)}
        <div class="mt-6 flex flex-wrap gap-3">
          <button data-action="prev-question" class="${ghost}">上一题</button>
          <button data-action="next-question" class="${primary}">下一题</button>
        </div>
      </article>
    </section>
  `;
}

function questionTitle(question) {
  const title = question.scenario || question.narrative || question.text || question.title || question.id;
  const prompt = question.question && question.question !== title
    ? `<p class="mt-3 text-lg font-semibold text-white">${escapeHtml(question.question)}</p>`
    : '';
  const subtitle = question.subtitle || question.facet || question.group || '请按第一反应作答。';
  const why = question._why_for_user || question.why_for_user || question.source || '用于提升画像稳定性。';
  return html`
    <p class="text-xs uppercase tracking-[0.25em] text-cyan-300">${escapeHtml(question.id || '')}</p>
    <h3 class="mt-3 text-xl font-bold text-white">${escapeHtml(title)}</h3>
    ${prompt}
    <p class="mt-2 text-sm text-slate-300">${escapeHtml(subtitle)}</p>
    <p class="mt-2 rounded-2xl bg-cyan-950/50 p-3 text-sm text-cyan-100">为什么问：${escapeHtml(why)}</p>
  `;
}

function answerControl(question, meta) {
  if (Array.isArray(question.options)) return optionControl(question);
  if (question.type === 'narrative_likert' || question.narrative_prompt) {
    return narrativeLikertControl(question, meta);
  }
  return likertControl(question, meta);
}

function optionControl(question) {
  const current = readAnswerKey(state.answers[question.id]);
  const options = stableShuffle(question.options, `${question.id}:options`);
  const rows = options.map((option) => html`
    <label class="block rounded-2xl border border-white/10 bg-white/5 p-4 hover:bg-white/10">
      <input type="radio" name="answer" value="${escapeAttr(option.key)}" ${checked(current === option.key)}>
      <span class="ml-2 font-semibold text-white">${escapeHtml(option.key)}.</span>
      <span class="ml-1 text-slate-100">${escapeHtml(option.text)}</span>
      ${option.signals ? `<p class="ml-6 mt-1 text-xs text-cyan-200">${escapeHtml(option.signals)}</p>` : ''}
    </label>
  `).join('');
  return `<div class="mt-5 grid gap-3">${rows}</div>`;
}

function narrativeLikertControl(question, meta) {
  const answer = normalizeAnswerObject(state.answers[question.id]);
  return html`
    ${likertControl(question, meta, answer.likert)}
    <textarea id="narrative" class="mt-4 h-28 w-full rounded-2xl border border-white/10 bg-slate-950 p-3 text-white"
      placeholder="${escapeAttr(question.narrative_prompt || '可补充一句你的真实想法')}"
    >${escapeHtml(answer.narrative || '')}</textarea>
  `;
}

function likertControl(question, meta, override) {
  const max = question.likert || question.scale_max || meta.likert || 5;
  const current = override ?? state.answers[question.id];
  const choices = stableShuffle(likertChoices(question, meta, max), `${question.id}:likert:${max}`);
  const rows = choices.map((choice) => {
    return html`
      <label class="rounded-2xl border border-white/10 bg-white/5 p-3 text-left hover:bg-white/10">
        <input type="radio" name="answer" value="${choice.value}" ${checked(Number(current) === choice.value)}>
        <span class="ml-1 text-sm text-slate-100">${escapeHtml(choice.label)}</span>
      </label>
    `;
  }).join('');
  return `<div class="mt-5 grid gap-2 sm:grid-cols-${Math.min(max, 7)}">${rows}</div>`;
}

function finishChapter() {
  state.result = scoreAssessment(state.questions, state.answers);
  state.view = 'chapter-summary';
  saveState();
  return chapterSummaryHtml();
}

function chapterSummaryHtml() {
  const chapter = activeAssessmentChapters()[state.currentChapter] || {};
  const questions = chapterQuestions(chapter);
  const meta = SCALE_META[state.currentScale] || { name: state.currentScale };
  const answered = questions.filter((q) => state.answers[q.id]).length;
  return html`
    <section class="${shell}">
      <p class="text-sm text-cyan-300">章节小结</p>
      <h2 class="mt-2 text-3xl font-black text-white">${escapeHtml(chapter.title || meta.name)}</h2>
      <p class="mt-2 text-slate-300">已完成 ${answered} 题，结果已进入本地评分模型。</p>
      <div class="mt-5 grid gap-3 sm:grid-cols-3">
        ${summaryMetric('当前量表', meta.name)}
        ${summaryMetric('已答总题数', String(answeredCount()))}
        ${summaryMetric('本地保存', '已保存')}
      </div>
      <button data-action="next-scale" class="${primary} mt-6">继续下一章</button>
    </section>
  `;
}

function resultHtml() {
  state.result ||= scoreAssessment(state.questions, state.answers);
  saveState();
  return html`
    <section id="report" class="${shell}">
      <p class="text-sm text-cyan-300">第 7 章：画像与生成</p>
      <h2 class="mt-2 text-3xl font-black text-white">测评结果</h2>
      <div class="mt-6 grid gap-5 lg:grid-cols-2">
        <div class="${soft}"><canvas id="hexChart"></canvas></div>
        <div class="${soft}"><canvas id="valuesChart"></canvas></div>
      </div>
      ${resultSummaryHtml()}
      <div class="mt-6 flex flex-wrap gap-3">
        <button data-action="export-json" class="${primary}">导出 JSON</button>
        <button data-action="analyze-llm" class="${ghost}">调用 LLM 分析</button>
        <button data-action="export-shot" class="${ghost}">导出截图</button>
        <button data-action="generate" class="${primary}">批量生成</button>
      </div>
      <pre id="analysisStatus" class="mt-4 max-h-72 overflow-auto rounded-2xl bg-slate-950 p-4 text-xs text-cyan-100"></pre>
    </section>
  `;
}

function resultSummaryHtml() {
  const result = state.result || {};
  const values = result.motivation?.values || [];
  const signals = result.situational?.totals || [];
  return html`
    <div class="mt-6 grid gap-4 lg:grid-cols-3">
      ${summaryMetric('依恋类型', result.attachment?.style || '未计算')}
      ${summaryMetric('焦虑 / 回避', formatPair(result.attachment?.anxiety, result.attachment?.avoidance))}
      ${summaryMetric('自我效能', formatScore(result.motivation?.self_efficacy))}
    </div>
    <div class="mt-4 grid gap-4 lg:grid-cols-2">
      ${listBlock('价值观排序', values.map((item) => `${item.value}: ${formatScore(item.score)}`))}
      ${listBlock('情境信号汇总', signals.map((item) => `${item.dimension}: ${item.score.toFixed(2)}`))}
    </div>
  `;
}

function generateHtml() {
  return html`
    <section class="${shell}">
      <p class="text-sm text-cyan-300">LLM Generate</p>
      <h2 class="mt-2 text-3xl font-black text-white">批量生成虚拟画像</h2>
      <p class="mt-2 text-sm text-slate-300">这里只动态调用 LLM 模块，不写 mock 结果。</p>
      <label class="mt-5 grid gap-2 text-sm text-slate-300">
        <span>生成数量：<b id="countLabel">5</b></span>
        <input id="count" type="range" min="1" max="50" value="5" class="w-full">
      </label>
      <textarea id="prompt" class="mt-4 h-28 w-full rounded-2xl border border-white/10 bg-slate-950 p-3 text-white"
        placeholder="例如：生成不同年龄、职业、关系模式的人群样本"></textarea>
      <button data-action="run-generate" class="${primary} mt-4">调用 LLM 生成</button>
      <pre id="output" class="mt-4 max-h-96 overflow-auto rounded-2xl bg-slate-950 p-4 text-xs text-cyan-100"></pre>
    </section>
  `;
}

function bindGlobal(root) {
  root.querySelector('[data-action="home"]')?.addEventListener('click', () => {
    state.view = 'home';
    saveState();
    renderApp();
  });
  root.querySelector('[data-action="reset"]')?.addEventListener('click', () => {
    resetState();
    renderApp();
  });
  root.querySelector('[data-action="debug"]')?.addEventListener('click', () => {
    state.debug = !state.debug;
    saveState();
    renderApp();
  });
  if (state.debug) root.insertAdjacentHTML('beforeend', debugHtml());
}

function bindView(root) {
  bindHome(root);
  bindAssessment(root);
  bindSummary(root);
  bindResult(root);
  bindGenerate(root);
}

function bindHome(root) {
  const applyHomeConfig = () => {
    state.provider = root.querySelector('#provider')?.value || 'openai';
    state.apiKey = normalizeUserApiKey(state.provider, root.querySelector('#apiKey')?.value || '');
    state.mode = root.querySelector('input[name="mode"]:checked')?.value || 'full';
    saveState();
  };

  root.querySelector('[data-action="save-config"]')?.addEventListener('click', () => {
    applyHomeConfig();
    alert('LLM 设置已保存到当前会话。');
  });

  root.querySelector('[data-action="go-result"]')?.addEventListener('click', () => {
    applyHomeConfig();
    state.result ||= scoreAssessment(state.questions, state.answers);
    state.view = 'result';
    saveState();
    renderApp();
  });

  root.querySelector('[data-action="go-assessment"]')?.addEventListener('click', () => {
    applyHomeConfig();
    state.view = 'assessment';
    alignCurrentScale();
    saveState();
    renderApp();
  });

  root.querySelector('[data-action="start"]')?.addEventListener('click', () => {
    applyHomeConfig();
    state.answers = {};
    state.optionOrderSeed = makeOptionOrderSeed();
    state.result = null;
    state.generated = [];
    state.currentChapter = 0;
    state.currentQ = 0;
    const first = chapterQuestions(activeAssessmentChapters()[0])[0];
    state.currentScale = first?.__scale || 'hexaco';
    state.view = 'assessment';
    saveState();
    renderApp();
  });
}

function bindAssessment(root) {
  root.querySelector('[data-action="prev-question"]')?.addEventListener('click', () => {
    state.currentQ = Math.max(0, state.currentQ - 1);
    saveState();
    renderApp();
  });
  root.querySelector('[data-action="next-question"]')?.addEventListener('click', () => {
    const chapter = activeAssessmentChapters()[state.currentChapter];
    const question = chapterQuestions(chapter)[state.currentQ];
    const answer = readCurrentAnswer(root, question);
    if (answer == null || answer === '') return alert('请选择一个选项');
    setAnswer(question.id, answer);
    state.currentQ += 1;
    saveState();
    renderApp();
  });
}

function bindSummary(root) {
  root.querySelector('[data-action="next-scale"]')?.addEventListener('click', () => {
    const chapters = activeAssessmentChapters();
    const nextChapter = state.currentChapter + 1;
    if (!chapters[nextChapter]) {
      state.result = scoreAssessment(state.questions, state.answers);
      state.view = 'result';
    } else {
      state.currentChapter = nextChapter;
      state.currentQ = 0;
      state.view = 'assessment';
      const first = chapterQuestions(chapters[nextChapter])[0];
      state.currentScale = first?.__scale || state.currentScale;
    }
    saveState();
    renderApp();
  });
}

function bindResult(root) {
  if (state.view !== 'result') return;
  renderHexChart(root.querySelector('#hexChart'), state.result.hex || {});
  renderValuesChart(root.querySelector('#valuesChart'), state.result.motivation?.values || []);
  const output = root.querySelector('#analysisStatus');
  const analyzeBtn = root.querySelector('[data-action="analyze-llm"]');
  let analyzing = false;

  root.querySelector('[data-action="export-json"]')?.addEventListener('click', () => {
    try {
      downloadJson('profile.json', buildProfile(state));
      if (output) output.textContent = 'JSON 导出已触发。若浏览器拦截下载，请检查下载权限。';
    } catch (error) {
      if (output) output.textContent = `JSON 导出失败：${error.message}`;
    }
  });
  analyzeBtn?.addEventListener('click', async () => {
    if (analyzing) return;
    analyzing = true;
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '分析中...';
    const controller = new AbortController();
    const hardTimeout = setTimeout(() => controller.abort(new Error('analysis hard timeout')), 130000);
    output.textContent = '调用 LLM 分析中...';
    try {
      state.fullProfile = await analyzeProfileWithLLM(state, {
        signal: controller.signal,
        onStatus: (message) => {
          output.textContent = message;
        }
      });
      saveState();
      output.textContent = JSON.stringify(state.fullProfile, null, 2);
    } catch (error) {
      output.textContent = normalizeAnalysisError(error);
    } finally {
      clearTimeout(hardTimeout);
      analyzing = false;
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = '调用 LLM 分析';
    }
  });
  root.querySelector('[data-action="export-shot"]')?.addEventListener('click', async () => {
    try {
      await exportScreenshot(root.querySelector('#report'));
      if (output) output.textContent = '截图导出已触发。';
    } catch (error) {
      if (output) output.textContent = `截图导出失败：${error.message}`;
    }
  });
  root.querySelector('[data-action="generate"]')?.addEventListener('click', () => {
    state.view = 'generate';
    saveState();
    renderApp();
  });
}

function bindGenerate(root) {
  const count = root.querySelector('#count');
  const label = root.querySelector('#countLabel');
  count?.addEventListener('input', () => {
    label.textContent = count.value;
  });
  root.querySelector('[data-action="run-generate"]')?.addEventListener('click', async () => {
    const output = root.querySelector('#output');
    output.textContent = '调用中...';
    try {
      const data = await generateProfilesWithLLM({
        state,
        prompt: root.querySelector('#prompt')?.value || '',
        count: Number(count?.value || 5)
      });
      state.generated = Array.isArray(data) ? data : data?.profiles || [];
      saveState();
      output.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
      output.textContent = error.message;
    }
  });
}

function readCurrentAnswer(root, question) {
  if (!question) return null;
  const selected = root.querySelector('input[name="answer"]:checked');
  if (Array.isArray(question.options)) return selected?.value || '';
  if (question.type === 'narrative_likert' || question.narrative_prompt) {
    if (!selected) return null;
    return {
      likert: Number(selected.value),
      narrative: root.querySelector('#narrative')?.value || ''
    };
  }
  return selected ? Number(selected.value) : null;
}

function summaryMetric(label, value) {
  return html`
    <div class="rounded-2xl border border-white/10 bg-slate-950/70 p-4">
      <p class="text-xs text-slate-400">${escapeHtml(label)}</p>
      <p class="mt-1 text-lg font-bold text-white">${escapeHtml(value)}</p>
    </div>
  `;
}

function listBlock(title, rows) {
  const items = rows.length ? rows : ['暂无数据'];
  return html`
    <div class="${soft}">
      <h3 class="font-bold text-white">${escapeHtml(title)}</h3>
      <ul class="mt-3 space-y-2 text-sm text-slate-200">
        ${items.map((row) => `<li>${escapeHtml(row)}</li>`).join('')}
      </ul>
    </div>
  `;
}

function modeRadio(value, label) {
  return html`
    <label class="rounded-full border border-white/10 px-4 py-2 hover:bg-white/10">
      <input type="radio" name="mode" value="${value}" ${checked(state.mode === value)}>
      <span class="ml-2">${escapeHtml(label)}</span>
    </label>
  `;
}

function option(value, current) {
  return `<option value="${value}" ${selected(value === current)}>${value}</option>`;
}

function likertChoices(question, meta, max) {
  if (Array.isArray(question.anchors) && question.anchors.length === max) {
    return question.anchors.map((label, index) => ({ value: index + 1, label }));
  }
  return likertLabels(question, meta, max).map((label, index) => ({ value: index + 1, label }));
}

function likertLabels(question, meta, max) {
  if (max === 5) return ['非常不同意', '不同意', '中立', '同意', '非常同意'];
  if (max === 7 && question.__scale === 'values') {
    return ['完全不像我', '很不像我', '有点不像我', '说不准', '有点像我', '比较像我', '非常像我'];
  }
  if (max === 7 && (question.group === 'domain' || question.scale === 'gse_domain')) {
    return ['完全没把握', '把握很低', '有点没底', '一半一半', '有一些把握', '比较有把握', '非常有把握'];
  }
  if (max === 7 && meta?.id === 'motivation') {
    return ['完全不同意', '比较不同意', '有点不同意', '说不准', '有点同意', '比较同意', '非常同意'];
  }
  return Array.from({ length: max }, (_, index) => `程度 ${index + 1}`);
}

function stableShuffle(items, salt) {
  return items
    .map((item, index) => ({ item, rank: seededHash(`${state.optionOrderSeed}:${salt}:${index}`) }))
    .sort((left, right) => left.rank - right.rank)
    .map((entry) => entry.item);
}

function seededHash(input) {
  let hash = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function readAnswerKey(answer) {
  if (typeof answer === 'string') return answer;
  return answer?.key || answer?.option || answer?.value || '';
}

function normalizeAnswerObject(answer) {
  if (!answer || typeof answer !== 'object') return { likert: Number(answer) || 0, narrative: '' };
  return answer;
}

function formatPair(left, right) {
  return `${formatScore(left)} / ${formatScore(right)}`;
}

function formatScore(value) {
  return Number.isFinite(value) ? value.toFixed(2) : '0.00';
}

function normalizeAnalysisError(error) {
  const message = String(error?.message || error || '未知错误');
  if (/401|login fail|api secret key|authorization/i.test(message)) {
    return 'API Key 校验失败：请检查 key 是否完整、无多余字符，并确认 provider 选择正确。';
  }
  if (/did not contain text content|no text/i.test(message)) {
    return '模型返回了思考片段但未给出正文结果，已触发重试；若仍失败请再点一次分析。';
  }
  if (/timed out|timeout/i.test(message)) {
    return 'LLM 请求超时：请确认代理是否走直连，或稍后重试。';
  }
  if (/aborted|abort/i.test(message)) {
    return 'LLM 请求已中止：请重试一次。';
  }
  return message;
}

function normalizeUserApiKey(provider, key) {
  const normalizedProvider = String(provider || '').toLowerCase();
  let value = String(key || '').trim();
  if (normalizedProvider === 'minimax' && /^sk-/i.test(value)) {
    value = value.replace(/[。．.,，;；:：!！?？"'`）)\]】》〉>]+$/u, '');
  }
  return value;
}

function debugHtml() {
  return html`
    <pre class="fixed bottom-4 right-4 z-50 max-h-80 w-96 overflow-auto rounded-2xl bg-black p-3 text-xs text-lime-300">
${escapeHtml(JSON.stringify({ ...state, questions: 'loaded' }, null, 2))}</pre>
  `;
}

function checked(value) {
  return value ? 'checked' : '';
}

function selected(value) {
  return value ? 'selected' : '';
}

function html(strings, ...values) {
  return strings.reduce((out, chunk, index) => out + chunk + (values[index] ?? ''), '');
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll('`', '&#096;');
}
