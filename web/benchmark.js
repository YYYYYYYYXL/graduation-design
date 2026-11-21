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
let page = 1, sampleRequest = 0;
const examples = new Map();

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
  return done;
}

async function refreshReport(id, previousDone = -1) {
  try {
    const data = await json(`${base}/${id}`);
    if (id !== activeId) return;
    const first = !report;
    report = data;
    running = report.state === 'running';
    updateControls();
    const done = drawSummary();
    if (first) {
      byId('batch-report').hidden = byId('batch-samples').hidden = false;
      const filter = byId('batch-filter');
      filter.replaceChildren(new Option('所有测试组', ''));
      report.datasets.forEach(group => filter.add(new Option(groupName(group), group)));
      byId('batch-search').value = '';
      if (![...byId('batch-runs').options].some(option => option.value === id)) {
        byId('batch-runs').prepend(runOption(report));
      }
      byId('batch-runs').value = id;
    }

    if (first || done !== previousDone) {
      loadSamples().catch(showError);
      await Promise.all(report.datasets.map(dataset => loadExample(dataset)));
      if (id !== activeId) return;
    }
    if (running) pollTimer = setTimeout(() => refreshReport(id, done), 2000);
  } catch (error) {
    if (id !== activeId) return;
    showError(error);
    if (running) pollTimer = setTimeout(() => refreshReport(id, previousDone), 4000);
  }
}

function runOption(run) {
  const date = new Date(run.created).toLocaleString('zh-CN', {hour12: false});
  return new Option(`${date} · ${run.preview ? '预览' : '批量评测'} · ${run.datasets.length} 组`, run.id);
}

function openRun(id) {
  clearTimeout(pollTimer);
  activeId = id;
  report = null; page = 1;
  byId('batch-report').hidden = byId('batch-samples').hidden = true;
  examples.clear(); byId('batch-crops').replaceChildren(); byId('batch-png').disabled = true;
  history.replaceState(null, '', `?run=${encodeURIComponent(id)}`);
  return refreshReport(id);
}

async function loadSamples() {
  const id = activeId, version = ++sampleRequest;
  const params = new URLSearchParams({dataset: byId('batch-filter').value, search: byId('batch-search').value, page});
  const data = await json(`${base}/${id}/samples?${params}`);
  if (id !== activeId || version !== sampleRequest) return;
  const {table, body} = buildTable(['输入文件', '测试组', '输入 PSNR (dB)', '输入 SSIM', '结果']);
  data.items.forEach(sample => {
    const row = body.insertRow(), button = element('button', sample.filename, 'batch-filename');
    button.type = 'button';
    button.onclick = () => loadExample(sample.dataset, sample.id, true);
    row.insertCell().append(button);
    [groupName(sample.dataset), fmt(sample.psnr), fmt(sample.ssim, 4),
      sample.error ? '参考评分失败' : `${sample.ready}/${report.models.length} 个方法`]
      .forEach(value => { row.insertCell().textContent = value; });
  });
  byId('batch-sample-list').replaceChildren(data.total ? table : element('p', '没有匹配的样本。', 'muted'));
  page = data.page;
  byId('batch-prev').disabled = page <= 1;
  byId('batch-next').disabled = page * data.page_size >= data.total;
  byId('batch-page').textContent = `第 ${page} / ${Math.max(1, Math.ceil(data.total / data.page_size))} 页 · ${data.total} 张`;
}

async function loadExample(dataset, sampleId, focus = false) {
  const run = activeId;
  let slot = examples.get(dataset);
  if (!slot) {
    const container = element('div');
    byId('batch-crops').append(container);
    slot = {container, version: 0, sampleId: null, signature: '', plate: null};
    examples.set(dataset, slot);
  }
  if (sampleId !== undefined) slot.sampleId = sampleId;
  const version = ++slot.version;
  const current = () => run === activeId && version === slot.version && examples.get(dataset) === slot;
  try {
    if (slot.sampleId === null) {
      const samples = await json(base + '/' + run + '/samples?dataset=' + encodeURIComponent(dataset));
      if (!current() || !samples.items.length) return;
      slot.sampleId = samples.items[0].id;
    }
    const detail = await json(base + '/' + run + '/samples/' + slot.sampleId);
    if (!current()) return;
    const title = detail.label + ' · ' + detail.sample.filename;
    const psnr = result => result?.psnr == null ? '' : fmt(result.psnr) + ' dB';
    const views = [
      {label: '带噪全图', url: detail.input_url, overview: true},
      {label: '带噪输入', url: detail.input_url, score: psnr(detail.sample)},
      {label: '参考图', url: detail.reference_url},
      ...detail.models.map(model => ({label: model.name, url: model.url,
        score: psnr(model.result), modelPsnr: model.result?.psnr, status: model.result?.error || '待评测'}))
    ];
    const signature = JSON.stringify([title, views]);
    if (signature !== slot.signature) {
      slot.plate = paperRow(title, views, {x: byId('batch-x'), y: byId('batch-y'), size: byId('batch-size')});
      slot.container.replaceChildren(slot.plate.row);
      slot.signature = signature;
      byId('batch-png').disabled = false;
    }
    if (focus) {
      slot.container.tabIndex = -1;
      slot.container.focus();
    }
  } catch (error) {
    if (current()) showError(error);
  }
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
byId('batch-prev').onclick = () => { page--; loadSamples().catch(showError); };
byId('batch-next').onclick = () => { page++; loadSamples().catch(showError); };
byId('batch-filter').onchange = byId('batch-search').oninput = () => {
  page = 1; loadSamples().catch(showError);
};
byId('batch-png').onclick = () => exportPlate(byId('batch-crops'), '多模型去噪结果');
for (const id of ['batch-x', 'batch-y', 'batch-size']) {
  byId(id).oninput = () => examples.forEach(slot => slot.plate?.redraw());
}
async function initialize() {
  const [datasets, models, recent] = await Promise.all([json('/api/benchmark/datasets'), json('/api/models'), json(base)]);
  groups = datasets;
  groups.forEach(group => addChoice('batch-groups', group, group.label, group.error || `${group.count} 张测试图`, group.count > 0));
  models.forEach(model => addChoice('batch-models', model, model.name,
    model.available ? '' : '缺少权重', model.available));
  byId('batch-runs').replaceChildren(...recent.map(runOption));
  updateControls();
  const requested = new URLSearchParams(location.search).get('run');
  const id = requested || recent.find(run => run.state === 'running')?.id || recent[0]?.id;
  if (id) await openRun(id);
}

initialize().catch(showError);
