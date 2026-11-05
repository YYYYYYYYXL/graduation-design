const $ = id => document.getElementById(id);
let models = [], tab = 'normal', results = [];
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
  $('models-panel').hidden = next !== 'models'; $('progress-panel').hidden = next !== 'progress';
  $('experiment-options').hidden = next !== 'experiment';
  $('form-title').textContent = next === 'experiment' ? '实验对比' : '图片去噪';
}
document.querySelectorAll('[data-tab]').forEach(b => b.onclick = () => selectTab(b.dataset.tab));
$('model').onchange = () => $('model-note').textContent = models.find(m => m.id === $('model').value)?.note || '';
$('synthetic').onchange = () => { $('reference').disabled = $('synthetic').checked; if ($('synthetic').checked) $('reference').value = ''; };
function format(value, digits = 3) { return value === null || value === undefined ? '未测量' : typeof value === 'number' ? value.toFixed(digits) : value; }
function addResult(result) {
  const article = node('article', undefined, 'result'); article.append(node('h3', result.model_name));
  const images = node('div', undefined, 'images');
  [['输入图', result.input_url], ['去噪结果', result.output_url], ['参考图 GT', result.reference_url]].forEach(([label, url]) => {
    if (!url) return;
    const figure = node('figure'), link = node('a'), img = node('img');
    link.href = url; link.target = '_blank'; link.rel = 'noopener'; img.src = url; img.alt = label; img.loading = 'lazy';
    link.append(img); figure.append(node('figcaption', label + ' · 点击查看原尺寸'), link); images.append(figure);
  });
  const values = node('div', undefined, 'metrics');
  [['PSNR (dB)', format(result.metrics.psnr)], ['SSIM', format(result.metrics.ssim, 4)], ['推理耗时 (s)', format(result.seconds)], ['加载耗时 (s)', format(result.load_seconds)], ['输入 PSNR (dB)', format(result.input_metrics.psnr)]].forEach(([label, value]) => {
    const item = node('span', label); item.append(node('strong', value)); values.append(item);
  });
  const links = node('div', undefined, 'links');
  [['下载结果', result.output_url], ['指标 CSV', result.csv_url], ['完整记录 JSON', result.json_url]].forEach(([label, url]) => { const a = node('a', label); a.href = url; a.download = ''; links.append(a); });
  article.append(images, values, node('p', `${result.width} × ${result.height} · ${result.device} · ${result.tiled ? '分块推理（与整图结果可能不同）' : '单块推理'}。${result.note}`, 'muted'), links);
  $('results').append(article);
}
$('denoise-form').onsubmit = async event => {
  event.preventDefault();
  if (!$('image').files.length) return;
  const experiment = tab === 'experiment';
  const ids = experiment && $('compare').checked ? [...document.querySelectorAll('.compare-item:checked')].map(c => c.value) : [$('model').value];
  if (!ids.length) { $('status').textContent = '请至少选择一个模型'; return; }
  const form = new FormData(); form.set('image', $('image').files[0]); form.set('device', $('device').value); form.set('sigma', $('sigma').value);
  form.set('synthetic', String(experiment && $('synthetic').checked)); form.set('seed', $('seed').value);
  if (experiment && !$('synthetic').checked && $('reference').files.length) form.set('reference', $('reference').files[0]);
  const controls = [...document.querySelectorAll('form input, form select, form button, nav button')];
  const disabled = controls.map(c => c.disabled); controls.forEach(c => c.disabled = true);
  results = []; $('results').replaceChildren(); $('empty').hidden = true; $('export').disabled = true;
  $('status').className = ''; const errors = [];
  try {
    for (const [index, id] of ids.entries()) {
      const label = models.find(m => m.id === id).name;
      $('status').textContent = `正在处理 ${index + 1}/${ids.length}：${label}。大图或 CPU 处理可能需要较长时间，请保持页面打开。`;
      form.set('model', id);
      try { const result = await api('/api/denoise', {method: 'POST', body: form}); results.push(result); addResult(result); }
      catch (error) { errors.push(`${label}：${error.message}`); $('results').append(node('p', errors.at(-1), 'error')); }
    }
    $('status').textContent = `处理结束：成功 ${results.length}，失败 ${errors.length}。${errors.length ? '失败原因见结果区域。' : ''}`;
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
  models.forEach(model => {
    const option = node('option', model.name + (model.available ? '' : ' · 权重缺失')); option.value = model.id; option.disabled = !model.available; $('model').append(option);
    const label = node('label'), check = node('input'); check.type = 'checkbox'; check.className = 'compare-item'; check.value = model.id; check.disabled = !model.available;
    check.checked = model.available && ['bm3d','dncnn_official','restormer_real','p2n_div2k'].includes(model.id); label.append(check, document.createTextNode(model.name)); $('compare-models').append(label);
    const card = node('article', undefined, 'model-card'); card.append(node('h3', model.name), node('p', model.available ? '本地文件已就绪' : '权重缺失'), node('small', model.path || '无需权重'), node('small', model.source), node('p', model.note, 'muted'));
    const button = node('button', '查看日志'); button.onclick = async () => {
      try { const data = await api(`/api/history/${model.id}`); $('history-note').textContent = `${model.name}：${data.note}`; $('history').replaceChildren(); if (data.rows.length) { const keys = Object.keys(data.rows[0]); $('history').append(table(keys, data.rows.map(r => keys.map(k => r[k])))); } }
      catch (error) { $('history-note').textContent = error.message; }
    }; card.append(button); $('model-cards').append(card);
  });
  $('model').value = models.find(m => m.id === 'dncnn_official' && m.available)?.id || models.find(m => m.available)?.id;
  $('model').onchange();
  const progress = await api('/api/progress');
  $('progress').append(table(['模块','序号','功能点','原表进度','当前状态','原表日期','实际说明'], progress.tasks.map(t => [t.module,t.id,t.name,`${t.original_progress}%`,t.status,t.original_date,t.note])));
}
init().catch(error => { $('status').className = 'error'; $('status').textContent = `加载失败：${error.message}`; });
