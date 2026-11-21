const byId = id => document.getElementById(id);
const element = (tag, text, className) => {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
};
const fmt = (value, digits = 2) => typeof value === 'number' ? value.toFixed(digits) : value ?? '—';
const base = '/api/benchmark/runs';
let groups = [], activeId = '', report = null, pollTimer, starting = false, running = false;

async function json(url, options) {
  const response = await fetch(url, {cache: 'no-store', ...options});
  const body = await response.json();
  if (!response.ok) throw Object.assign(new Error(body.error || '无法读取评测结果。'), {runId: body.id});
  return body;
}

const showError = error => { byId('batch-error').textContent = error ? error.message : ''; };
const selected = id => [...byId(id).querySelectorAll('input:checked')].map(input => input.value);
const groupName = id => groups.find(group => group.id === id)?.label ?? id;
const checkedGroups = () => selected('batch-groups');
const checkedModels = () => selected('batch-models');

function addChoice(panel, item, title, subtitle, enabled) {
  const label = element('label');
  const check = element('input');
  check.type = 'checkbox';
  check.value = item.id;
  check.disabled = !enabled;
  check.checked = enabled;
  check.onchange = updateControls;
  label.append(check, document.createTextNode(title));
  if (subtitle) label.append(element('small', subtitle));
  byId(panel).append(label);
}

function updateControls() {
  const datasets = checkedGroups();
  const modelCount = checkedModels().length;
  const count = groups.filter(item => datasets.includes(item.id)).reduce((sum, item) => sum + item.count, 0);
  byId('batch-start').disabled = byId('batch-preview').disabled = starting || running || !count || !modelCount;
  byId('batch-selection').textContent = count && modelCount
    ? `${count.toLocaleString()} 张样本 × ${modelCount} 个方法 · ${(count * modelCount).toLocaleString()} 次去噪`
    : '请选择测试组和方法。';
}

function buildTable(headings) {
  const table = element('table');
  const header = table.createTHead().insertRow();
  headings.forEach(name => {
    const th = element('th', name); th.scope = 'col'; header.append(th);
  });
  return {table, body: table.createTBody()};
}

function drawSummary() {
  byId('batch-summary').replaceChildren(benchmarkTables(report));
  byId('batch-summary-csv').href = `${base}/${activeId}/summary.csv`;
  byId('batch-images-csv').href = `${base}/${activeId}/images.csv`;
  const rowsWithModels = report.rows.filter(row => row.model !== 'input');
  const done = rowsWithModels.reduce((sum, row) => sum + row.done, 0);
  const total = rowsWithModels.reduce((sum, row) => sum + row.total, 0);
  const state = {
    running: '评测进行中', complete: '评测完成', complete_with_errors: '评测完成，有失败项',
    interrupted: '评测已中断', failed: '评测失败',
  }[report.state];
  byId('batch-state').textContent = `${report.preview ? '预览 · 每组 1 张 · ' : ''}${state} · 已处理 ${done}/${total} 项${report.error ? ` · ${report.error}` : ''}`;
  byId('batch-progress').max = total;
  byId('batch-progress').value = done;
}

async function refreshReport(id) {
  try {
    const data = await json(`${base}/${id}`);
    if (id !== activeId) return;
    const first = !report;
    report = data;
    running = report.state === 'running';
    updateControls();
    drawSummary();
    if (first) {
      byId('batch-report').hidden = false;
      if (![...byId('batch-runs').options].some(option => option.value === id)) {
        byId('batch-runs').prepend(runOption(report));
      }
      byId('batch-runs').value = id;
    }

    if (first) byId('cost-group').replaceChildren(...report.datasets.map(id => new Option(groupName(id), id)));
    drawCosts();
    if (running) pollTimer = setTimeout(() => refreshReport(id), 2000);
  } catch (error) {
    if (id !== activeId) return;
    showError(error);
    if (running) pollTimer = setTimeout(() => refreshReport(id), 4000);
  }
}

function runOption(run) {
  const date = new Date(run.created).toLocaleString('zh-CN', {hour12: false});
  return new Option(`${date} · ${run.preview ? '预览' : '模型评测'} · ${run.datasets.length} 组`, run.id);
}

function openRun(id) {
  clearTimeout(pollTimer);
  activeId = id;
  report = null;
  byId('batch-report').hidden = true;
  history.replaceState(null, '', `?run=${encodeURIComponent(id)}`);
  return refreshReport(id);
}

byId('batch-form').onsubmit = async event => {
  event.preventDefault();
  if (starting || running) return;
  showError(null);
  starting = true; updateControls();
  try {
    const result = await json(base, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({datasets: checkedGroups(), models: checkedModels(), preview: event.submitter?.id === 'batch-preview'}),
    });
    await openRun(result.id);
  } catch (error) {
    showError(error);
    if (error.runId) await openRun(error.runId);
  } finally {
    starting = false; updateControls();
  }
};
byId('batch-runs').onchange = event => { showError(null); openRun(event.target.value); };
async function initialize() {
  const [datasets, models, recent] = await Promise.all([json('/api/benchmark/datasets'), json('/api/models'), json(base)]);
  loadEpochCharts(models).catch(error => { byId('epoch-note').textContent = error.message; });
  groups = datasets;
  groups.forEach(group => addChoice('batch-groups', group, group.label, group.error || `${group.count} 张测试图`, group.count > 0));
  const {table, body} = buildTable(['参评', '模型 / 方法', '架构', '权重文件 (MiB)', '状态']);
  const families = {dncnn: '卷积网络', restormer: 'Transformer', p2n: 'U-Net · 预训练推理', bm3d: '传统非学习方法'};
  for (const model of models) {
    const row = body.insertRow(), checkbox = element('input');
    checkbox.type = 'checkbox'; checkbox.value = model.id; checkbox.checked = model.available;
    checkbox.disabled = !model.available; checkbox.setAttribute('aria-label', model.name);
    checkbox.onchange = updateControls; row.insertCell().append(checkbox);
    for (const value of [model.name, families[model.method] || model.method,
      Number.isFinite(model.weights_bytes) ? fmt(model.weights_bytes / 1048576) : '—',
      model.available ? '可用' : '缺少权重']) row.insertCell().textContent = value;
  }
  byId('batch-models').append(table);
  byId('batch-runs').replaceChildren(...recent.map(runOption));
  updateControls();
  const requested = new URLSearchParams(location.search).get('run');
  const id = requested || recent.find(run => run.state === 'running')?.id || recent[0]?.id;
  if (id) await openRun(id);
}

byId('cost-group').onchange = drawCosts;
initialize().catch(showError);
