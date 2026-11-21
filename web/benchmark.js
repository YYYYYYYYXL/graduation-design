const byId = id => document.getElementById(id);
const element = (tag, text, className) => {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
};
const fmt = (value, digits = 2) => typeof value === 'number' ? value.toFixed(digits) : value ?? '—';
const signed = (value, digits) => typeof value === 'number' && value > 0
  ? '+' + fmt(value, digits) : fmt(value, digits);
const base = '/api/benchmark/runs';
let groups = [], activeId = '', report = null, pollTimer, starting = false, running = false;
let page = 1, sampleRequest = 0, detailRequest = 0, currentDetail = null, cropImages = [];

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
  const {table, body} = buildTable(['测试组 · 共同计分', '方法', 'PSNR ↑ (dB)', 'SSIM ↑', 'ΔPSNR (dB)', '耗时 (s/张)', '设备', '失败数']);
  const precision = {psnr: 2, ssim: 4};
  for (const dataset of report.datasets) {
    const rows = report.rows.filter(row => row.dataset === dataset);
    const best = {};
    for (const [key, digits] of Object.entries(precision)) {
      const measured = rows.filter(row => row.model !== 'input').map(row => row[key] === '∞'
        ? Infinity : Number.isFinite(row[key]) ? Number(row[key].toFixed(digits)) : -Infinity);
      best[key] = Math.max(...measured);
    }
    rows.forEach((row, index) => {
      const tr = body.insertRow(), input = row.model === 'input';
      if (!index) {
        const group = element('th', groupName(dataset)); group.rowSpan = rows.length;
        group.append(element('small', row.compared + '/' + row.total + ' 张')); tr.append(group);
      }
      const name = element('th', row.name, 'batch-model-name'); name.scope = 'row'; tr.append(name);
      if (input) tr.className = 'baseline';
      if (row.model === 'p2n_div2k') name.append(element('small', '预训练权重直接推理'));
      for (const [key, digits] of Object.entries(precision)) {
        const value = fmt(row[key], digits);
        const num = value === '∞' ? Infinity : Number(value);
        const cell = tr.insertCell();
        if (!input && num === best[key]) cell.append(element('strong', value));
        else cell.textContent = value;
      }
      [signed(row.delta_psnr, 2), fmt(row.seconds, 3), row.device?.toUpperCase() || '—',
        input ? '—' : row.failed].forEach(value => tr.insertCell().textContent = value);
    });
  }
  byId('batch-summary').replaceChildren(table);
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
    updateBenchmarkCharts(report);
    if (first || done !== previousDone) {
      loadSamples().catch(showError);
      if (currentDetail) loadDetail(currentDetail).catch(showError);
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
  closeDetail();
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
    button.onclick = () => {
      currentDetail = sample.id;
      loadDetail(sample.id, true).catch(showError);
    };
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

function closeDetail() {
  ++detailRequest;
  currentDetail = null;
  cropImages = [];
  byId('batch-views').replaceChildren();
  byId('batch-crops').replaceChildren();
  byId('batch-detail').hidden = true;
  byId('batch-crop').open = true; byId('batch-mode').value = 'detail';
}

function updateCrops() {
  const source = cropImages.find(item => item.image.naturalWidth)?.image;
  if (!source || !byId('batch-crop').open) return;
  const region = cropRegion(source.naturalWidth, source.naturalHeight, byId('batch-size').value,
    byId('batch-x').value, byId('batch-y').value);
  byId('batch-crop-position').textContent = `左上角 (${region.left}, ${region.top}) · ${region.size} × ${region.size} 像素`;
  const target = cropImages.find(item => item.isReference)?.image;
  const mode = byId('batch-mode');
  mode.options[1].disabled = !target?.naturalWidth;
  if (!target?.naturalWidth) mode.value = 'detail';
  byId('batch-legend').textContent = plateModes[mode.value];
  byId('batch-legend').classList.toggle('error-scale', mode.value === 'error');
  cropImages.forEach(({image, canvas}) => {
    if (image.naturalWidth === source.naturalWidth && image.naturalHeight === source.naturalHeight) {
      drawCrop(canvas, image, region, mode.value === 'error' ? target : null);
    }
  });
}

async function loadDetail(sampleId, focus = false) {
  const id = activeId, version = ++detailRequest;
  const detail = await json(`${base}/${id}/samples/${sampleId}`);
  if (id !== activeId || version !== detailRequest) return;
  const gallery = byId('batch-views'), crops = byId('batch-crops');
  gallery.replaceChildren(); crops.replaceChildren(); cropImages = [];
  byId('batch-detail-title').textContent = `${detail.label} · ${detail.sample.filename}`;
  byId('batch-detail').hidden = false;
  const views = [
    {name: '带噪输入', url: detail.input_url, result: detail.sample},
    {name: '干净参考图', url: detail.reference_url},
    ...detail.models,
  ];
  for (const {name, url, result} of views) {
    const figure = element('figure');
    figure.append(element('figcaption', name));
    gallery.append(figure);
    if (!url) {
      figure.append(element('p', result?.error || '等待去噪结果', result?.error ? 'error' : 'muted'));
      continue;
    }
    const link = element('a'), image = element('img');
    image.alt = name;
    link.href = url; link.target = '_blank'; link.rel = 'noopener';
    link.append(image); figure.append(link);
    if (result) figure.append(element('small', result.error || `${fmt(result.psnr)} dB / ${fmt(result.ssim, 4)}`));
    const crop = element('figure'), canvas = element('canvas');
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', `${name}的局部放大`);
    crop.dataset.label = name;
    crop.dataset.scores = result ? 'PSNR ' + fmt(result.psnr) + ' dB · SSIM ' + fmt(result.ssim, 4) : '参考图（GT）';
    canvas.hidden = true;
    crop.append(element('figcaption', name), canvas, element('small', crop.dataset.scores)); crops.append(crop);
    cropImages.push({image, canvas, isReference: url === detail.reference_url});
    image.onload = updateCrops;
    image.onerror = () => figure.append(element('p', '图片文件无法读取。', 'error'));
    image.src = url;
  }
  byId('batch-crop').hidden = cropImages.length < 2;
  if (focus) byId('batch-detail-title').focus();
  updateCrops();
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
byId('batch-close-detail').onclick = () => { closeDetail(); byId('batch-search').focus(); };
byId('batch-crop').ontoggle = updateCrops;
byId('batch-png').onclick = () => exportPlate(byId('batch-crops'), byId('batch-detail-title').textContent,
  byId('batch-crop-position').textContent, byId('batch-legend').textContent);
['batch-x', 'batch-y', 'batch-size', 'batch-mode'].forEach(id => { byId(id).oninput = updateCrops; });

async function initialize() {
  const [datasets, models, recent] = await Promise.all([json('/api/benchmark/datasets'), json('/api/models'), json(base)]);
  groups = datasets;
  groups.forEach(group => addChoice('batch-groups', group, group.label, group.error || `${group.count} 张测试图`, group.count > 0));
  models.forEach(model => addChoice('batch-models', model, model.name,
    model.available ? (model.method === 'p2n' ? '预训练权重直接推理' : '') : '缺少权重', model.available));
  byId('batch-runs').replaceChildren(...recent.map(runOption));
  updateControls();
  const requested = new URLSearchParams(location.search).get('run');
  const id = requested || recent.find(run => run.state === 'running')?.id || recent[0]?.id;
  if (id) await openRun(id);
}

initialize().catch(showError);
