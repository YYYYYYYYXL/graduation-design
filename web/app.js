const $ = id => document.getElementById(id);
let models = [], tab = 'normal', results = [], defaultModel = '', previewUrl = '', validImage = false;
const methods = {dncnn_div2k: 'DnCNN', restormer_sidd_batch16: 'Restormer', p2n_div2k: 'P2N（预训练推理）', bm3d: 'BM3D'};
const node = (tag, text, className) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (className) el.className = className; return el; };
async function api(url, options) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { throw new Error('服务暂时没有响应，请稍后重试。'); }
  if (!response.ok) throw new Error(data.error || '请求失败');
  return data;
}
function selectTab(next) {
  if (tab !== next) {
    results = [];
    $('status').textContent = '';
    $('export').disabled = true;
  }
  tab = next;
  document.querySelectorAll('[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === next));
  $('workspace').hidden = !['normal', 'experiment'].includes(next);
  $('model-field').hidden = next !== 'experiment';
  $('intro').hidden = next === 'experiment';
  $('experiment-options').hidden = next !== 'experiment';
  $('experiment-options').disabled = next !== 'experiment';
  $('form-title').textContent = next === 'experiment' ? '实验对比' : '图片去噪';
  $('model-label').textContent = next === 'experiment' ? '模型' : '去噪方法';
  $('export').hidden = next !== 'experiment';
  const previous = $('model').value;
  $('model').replaceChildren();
  const choices = next === 'experiment' ? models.filter(m => m.available) : models.filter(m => m.id === defaultModel && m.available);
  choices.forEach(m => { const option = node('option', m.name); option.value = m.id; $('model').append(option); });
  if (choices.some(m => m.id === previous)) $('model').value = previous;
  $('results').replaceChildren();
  results.forEach(addResult);
  $('result-panel').hidden = !results.length;
  updateSubmit();
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
  [['PSNR (dB)', format(result.metrics.psnr)], ['SSIM', format(result.metrics.ssim, 4)], ['处理用时 (s)', format(result.total_seconds)]].forEach(([label, value]) => {
    const item = node('span', label); item.append(node('strong', value)); values.append(item);
  });
  const links = node('div', undefined, 'links');
  const downloads = experiment ? [['下载图片', result.output_url], ['导出指标', result.csv_url]] : [['下载图片', result.output_url]];
  downloads.forEach(([label, url]) => { const a = node('a', label); a.href = url + (url.endsWith('.png') ? '?download=1' : ''); a.download = ''; links.append(a); });
  article.append(images);
  if (experiment) {
    article.append(values);
    article.append(node('p', '处理用时包含读写图片和模型加载；首次使用通常更慢。', 'muted'));
    if (result.tiled) article.append(node('p', '该图片采用分块处理，指标可能与整图处理不同。', 'muted'));
  }
  article.append(links);
  $('results').append(article);
}
$('denoise-form').onsubmit = async event => {
  event.preventDefault();
  if (!validImage) return;
  const experiment = tab === 'experiment';
  const ids = experiment && $('compare').checked ? [...document.querySelectorAll('.compare-item:checked')].map(c => c.value) : [$('model').value];
  if (!ids.length) { $('status').textContent = '请至少选择一个模型'; return; }
  if (experiment && $('synthetic').checked && !$('sigma').value.trim()) { $('status').textContent = '添加噪声时请填写噪声强度。'; return; }
  if (experiment && $('image').files[0].size + ($('reference').files[0]?.size || 0) > 25 * 1024 * 1024) { $('status').textContent = '图片合计不能超过 25 MB。'; return; }
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
  const rows = [['model', 'psnr', 'ssim', 'seconds', 'load_seconds', 'total_seconds', 'device', 'sigma', 'seed', 'tiled', 'run_id'], ...results.map(r => [r.model, r.metrics.psnr, r.metrics.ssim, r.seconds, r.load_seconds, r.total_seconds, r.device, r.sigma, r.seed, r.tiled, r.id])];
  const csv = '\ufeff' + rows.map(row => row.map(v => `"${String(v ?? '').replaceAll('"', '""')}"`).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}));
  const a = node('a'); a.href = url; a.download = 'denoising_comparison.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
async function init() {
  const [available, settings] = await Promise.all([api('/api/models'), api('/api/settings')]);
  models = available; defaultModel = settings.default_model;
  models.filter(model => model.available).forEach(model => {
    const label = node('label'), check = node('input');
    check.type = 'checkbox'; check.className = 'compare-item'; check.value = model.id;
    check.checked = Object.hasOwn(methods, model.id);
    label.append(check, document.createTextNode(model.name)); $('compare-models').append(label);
  });
  selectTab(tab);
  $('model').disabled = false; updateSubmit();
}
init().catch(error => { $('status').className = 'error'; $('status').textContent = `加载失败：${error.message}`; });

function updateSubmit() {
  $('submit').disabled = !validImage || !$('model').options.length;
}
$('image').onchange = async () => {
  validImage = false; updateSubmit();
  $('selection-error').textContent = ''; $('status').textContent = '';
  results = []; $('results').replaceChildren(); $('result-panel').hidden = true; $('export').disabled = true;
  $('preview').hidden = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const file = $('image').files[0];
  if (!file) return;
  if (file.size > 25 * 1024 * 1024) { $('selection-error').textContent = '图片不能超过 25 MB。'; return; }
  const url = URL.createObjectURL(file); previewUrl = url;
  const img = new Image(); img.src = url;
  try {
    await img.decode();
    if (previewUrl !== url) return;
    if (Math.min(img.naturalWidth, img.naturalHeight) < 11 || img.naturalWidth * img.naturalHeight > 4000000) throw new Error('图片每边至少 11 像素，总像素不超过 400 万。');
    $('preview-image').src = url;
    $('file-info').textContent = `${file.name} · ${img.naturalWidth} × ${img.naturalHeight}`;
    $('preview').hidden = false; validImage = true;
  } catch (error) {
    if (previewUrl === url) $('selection-error').textContent = error.message.startsWith('图片') ? error.message : '无法读取图片，请选择 PNG、JPG、WebP 或 BMP 文件。';
  }
  updateSubmit();
};
