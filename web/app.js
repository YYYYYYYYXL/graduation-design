const $ = id => document.getElementById(id);
let models = [], tab = 'normal', results = [];
const methods = {dncnn_div2k: 'DnCNN', restormer_sidd_batch16: 'Restormer', p2n_div2k: 'P2N（预训练推理）', bm3d: 'BM3D'};
const node = (tag, text, className) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (className) el.className = className; return el; };
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '请求失败');
  return data;
}
function table(headers, rows) {
  const el = node('table'), head = node('thead'), body = node('tbody'), tr = node('tr');
  headers.forEach(h => tr.append(node('th', h))); head.append(tr);
  rows.forEach(row => { const line = node('tr'); row.forEach(v => line.append(node('td', v ?? '—'))); body.append(line); });
  el.append(head, body); return el;
}
function selectTab(next) {
  tab = next;
  document.querySelectorAll('[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === next));
  $('workspace').hidden = !['normal', 'experiment'].includes(next);
  $('progress-panel').hidden = next !== 'progress';
  $('experiment-options').hidden = next !== 'experiment';
  $('experiment-options').disabled = next !== 'experiment';
  $('form-title').textContent = next === 'experiment' ? '实验对比' : '图片去噪';
  $('model-label').textContent = next === 'experiment' ? '模型' : '去噪方法';
  $('export').hidden = next !== 'experiment';
  const previous = $('model').value;
  $('model').replaceChildren();
  const choices = next === 'experiment' ? models.filter(m => m.available) : Object.keys(methods).map(id => models.find(m => m.id === id && m.available)).filter(Boolean);
  choices.forEach(m => { const option = node('option', next === 'experiment' ? m.name : methods[m.id]); option.value = m.id; $('model').append(option); });
  if (choices.some(m => m.id === previous)) $('model').value = previous;
  $('results').replaceChildren();
  (next === 'normal' ? results.slice(-1) : results).forEach(addResult);
}
document.querySelectorAll('[data-tab]').forEach(b => b.onclick = () => selectTab(b.dataset.tab));
$('compare').onchange = () => $('compare-models').hidden = !$('compare').checked;
$('synthetic').onchange = () => { $('reference').disabled = $('synthetic').checked; if ($('synthetic').checked) $('reference').value = ''; };
function format(value, digits = 3) { return value === null || value === undefined ? '未测量' : typeof value === 'number' ? value.toFixed(digits) : value; }
function addResult(result) {
  const experiment = tab === 'experiment';
  const article = node('article', undefined, 'result');
  if (experiment) article.append(node('h3', result.model_name));
  const images = node('div', undefined, 'images');
  [['输入图', result.input_url], ['去噪结果', result.output_url], ['参考图 GT', result.reference_url]].forEach(([label, url]) => {
    if (!url || (!experiment && label === '参考图 GT')) return;
    const figure = node('figure'), link = node('a'), img = node('img');
    link.href = url; link.target = '_blank'; link.rel = 'noopener'; img.src = url; img.alt = label; img.loading = 'lazy';
    link.append(img); figure.append(node('figcaption', label), link); images.append(figure);
  });
  const values = node('div', undefined, 'metrics');
  [['PSNR (dB)', format(result.metrics.psnr)], ['SSIM', format(result.metrics.ssim, 4)], ['耗时 (s)', format(result.seconds)]].forEach(([label, value]) => {
    const item = node('span', label); item.append(node('strong', value)); values.append(item);
  });
  const links = node('div', undefined, 'links');
  const downloads = experiment ? [['下载图片', result.output_url], ['导出指标', result.csv_url]] : [['下载图片', result.output_url]];
  downloads.forEach(([label, url]) => { const a = node('a', label); a.href = url; a.download = ''; links.append(a); });
  article.append(images);
  if (experiment) {
    article.append(values);
    if (result.tiled) article.append(node('p', '该图片采用分块处理，指标可能与整图处理不同。', 'muted'));
  }
  article.append(links);
  $('results').append(article);
}
$('denoise-form').onsubmit = async event => {
  event.preventDefault();
  if (!$('image').files.length) return;
  const experiment = tab === 'experiment';
  const ids = experiment && $('compare').checked ? [...document.querySelectorAll('.compare-item:checked')].map(c => c.value) : [$('model').value];
  if (!ids.length) { $('status').textContent = '请至少选择一个模型'; return; }
  const form = new FormData(); form.set('image', $('image').files[0]); form.set('device', 'auto'); form.set('sigma', experiment ? $('sigma').value : '');
  form.set('synthetic', String(experiment && $('synthetic').checked)); form.set('seed', experiment ? $('seed').value : '42');
  if (experiment && !$('synthetic').checked && $('reference').files.length) form.set('reference', $('reference').files[0]);
  const controls = [...document.querySelectorAll('form input, form select, form button, nav button')];
  const disabled = controls.map(c => c.disabled); controls.forEach(c => c.disabled = true);
  results = []; $('results').replaceChildren(); $('result-panel').hidden = false; $('export').disabled = true;
  $('status').className = ''; const errors = [];
  try {
    for (const [index, id] of ids.entries()) {
      const label = models.find(m => m.id === id).name;
      $('status').textContent = experiment ? `正在处理 ${index + 1}/${ids.length}：${label}` : '正在去噪，请稍候…';
      form.set('model', id);
      try { const result = await api('/api/denoise', {method: 'POST', body: form}); results.push(result); addResult(result); }
      catch (error) { errors.push(`${label}：${error.message}`); $('results').append(node('p', errors.at(-1), 'error')); }
    }
    $('status').textContent = errors.length ? `成功 ${results.length} 项，失败 ${errors.length} 项，原因见下方。` : experiment ? `已完成 ${results.length} 个模型的对比。` : '去噪完成。';
    $('status').className = errors.length ? 'error' : '';
  } finally { controls.forEach((c, i) => c.disabled = disabled[i]); $('export').disabled = !results.length; }
};
$('export').onclick = () => {
  const rows = [['model', 'psnr', 'ssim', 'seconds', 'load_seconds', 'device', 'sigma', 'seed', 'tiled', 'run_id'], ...results.map(r => [r.model, r.metrics.psnr, r.metrics.ssim, r.seconds, r.load_seconds, r.device, r.sigma, r.seed, r.tiled, r.id])];
  const csv = '\ufeff' + rows.map(row => row.map(v => `"${String(v ?? '').replaceAll('"', '""')}"`).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
  const a = node('a'); a.href = url; a.download = 'denoising_comparison.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
async function init() {
  models = await api('/api/models');
  models.filter(model => model.available).forEach(model => {
    const label = node('label'), check = node('input');
    check.type = 'checkbox'; check.className = 'compare-item'; check.value = model.id;
    check.checked = Object.hasOwn(methods, model.id);
    label.append(check, document.createTextNode(model.name)); $('compare-models').append(label);
  });
  selectTab(tab);
  $('model').disabled = false; $('submit').disabled = !$('model').options.length;
  try {
    const progress = await api('/api/progress');
    $('progress').append(table(['模块','序号','功能点','原表进度','当前状态','原表日期','实际说明'], progress.tasks.map(t => [t.module,t.id,t.name,`${t.original_progress}%`,t.status,t.original_date,t.note])));
  } catch (error) { $('progress').append(node('p', error.message, 'error')); }
}
init().catch(error => { $('status').className = 'error'; $('status').textContent = `加载失败：${error.message}`; });
