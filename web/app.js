const $ = id => document.getElementById(id);
let models = [], tab = 'normal', results = [], defaultModel = '', previewUrl = '', validImage = false;
const methods = {dncnn_div2k: 'DnCNN · 自训 DIV2K', restormer_real: 'Restormer · 官方真实噪声', p2n_div2k: 'P2N · 预训练推理', bm3d: 'BM3D'};
const node = (tag, text, className) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (className) el.className = className; return el; };
async function api(url, options) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { throw new Error('服务暂时没有响应，请稍后重试。'); }
  if (!response.ok) throw new Error(data.error || '请求失败');
  return data;
}
function clearResults() {
  results = [];
  $('results').replaceChildren(); $('shared-input').replaceChildren(); $('comparison-summary').replaceChildren();
  $('shared-input').hidden = true; $('comparison-summary').hidden = true;
  $('result-panel').hidden = true; $('export').disabled = true;
  $('status').textContent = '';
}
function selectTab(next) {
  if (tab !== next) clearResults();
  tab = next;
  const experiment = tab === 'experiment';
  document.querySelectorAll('[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === next));
  for (const id of ['experiment-input', 'experiment-options']) {
    $(id).hidden = !experiment; $(id).disabled = !experiment;
  }
  $('intro').textContent = experiment ? '同一张图片，看看不同方法去噪后的区别。' : '减少照片里的颗粒噪点。上传照片后直接开始处理。';
  $('form-title').textContent = experiment ? '实验对比' : '图片去噪';
  $('result-title').textContent = experiment ? '对比结果' : '去噪结果';
  $('export').hidden = !experiment;
  $('results').className = experiment ? 'result-grid' : '';
  updateMode();
}
function updateMode() {
  const clean = tab === 'experiment' && $('synthetic').checked;
  $('upload-label').textContent = tab === 'normal' ? '上传图片' : clean ? '上传清晰照片' : '上传带噪照片';
  $('noise-options').hidden = !clean; $('noise-level').disabled = !clean;
  $('reference-options').hidden = clean; $('reference').disabled = clean;
  $('sigma-field').hidden = clean; $('sigma').disabled = clean;
  $('seed-field').hidden = !clean; $('seed').disabled = !clean;
  updateSubmit();
}
document.querySelectorAll('[data-tab]').forEach(b => b.onclick = () => selectTab(b.dataset.tab));
document.querySelectorAll('[name="input-mode"]').forEach(radio => radio.onchange = () => { clearResults(); updateMode(); });
$('clear-reference').onclick = () => { $('reference').value = ''; };
function imageFigure(label, url) {
  const figure = node('figure'), link = node('a'), img = node('img');
  link.href = url; link.target = '_blank'; link.rel = 'noopener';
  img.src = url; img.alt = label; img.loading = 'lazy';
  link.append(img); figure.append(node('figcaption', label), link); return figure;
}
function showSummary() {
  const first = results[0], hasReference = first.reference_provided;
  const summary = $('comparison-summary'); summary.replaceChildren(); summary.hidden = false;
  summary.append(node('p', hasReference ? 'PSNR、SSIM 越高，通常越接近参考图；仍需结合纹理和细节判断。' : '没有干净参考图，本次只比较图片效果和用时，不计算 PSNR、SSIM。', 'muted'));
  const wrapper = node('div', undefined, 'table-scroll'), table = node('table');
  const caption = node('caption', '本次对比汇总'); table.append(caption);
  const head = node('thead'), heading = node('tr');
  const labels = ['方法', ...(hasReference ? ['PSNR (dB)', 'SSIM'] : []), '处理用时 (s)'];
  labels.forEach(label => { const th = node('th', label); th.scope = 'col'; heading.append(th); });
  head.append(heading); table.append(head);
  const body = node('tbody');
  results.forEach(r => {
    const row = node('tr');
    [r.model_name, ...(hasReference ? [format(r.metrics.psnr), format(r.metrics.ssim, 4)] : []), format(r.total_seconds)].forEach(v => row.append(node('td', v)));
    body.append(row);
  });
  table.append(body); wrapper.append(table); summary.append(wrapper);
  summary.append(node('p', '用时包含图片读写和模型加载，首次使用通常更慢。点击图片可查看原图。' + (results.some(r => r.tiled) ? '部分结果采用分块处理，指标可能与整图不同。' : ''), 'muted'));
}
function format(value, digits = 3) { return value === null || value === undefined ? '未测量' : typeof value === 'number' ? value.toFixed(digits) : value; }
function addResult(result) {
  const experiment = tab === 'experiment';
  const article = node('article', undefined, 'result');
  if (experiment) {
    article.append(node('h3', result.model_name));
    if (results.length === 1) {
      $('shared-input').hidden = false;
      $('shared-input').append(imageFigure('共同输入 · 带噪照片', result.input_url));
      if (result.reference_url) $('shared-input').append(imageFigure('干净参考图', result.reference_url));
    }
    showSummary();
  }
  const images = node('div', undefined, 'images');
  if (!experiment) images.append(imageFigure('输入图', result.input_url));
  images.append(imageFigure('去噪结果', result.output_url)); article.append(images);
  const links = node('div', undefined, 'links');
  const a = node('a', '下载图片'); a.href = result.output_url + '?download=1'; a.download = ''; links.append(a);
  article.append(links); $('results').append(article);
}
$('denoise-form').onsubmit = async event => {
  event.preventDefault();
  if (!validImage) return;
  const experiment = tab === 'experiment';
  const ids = experiment ? [...document.querySelectorAll('.compare-item:checked')].map(c => c.value) : [defaultModel];
  const clean = experiment && $('synthetic').checked;
  if (!ids.length) { $('status').textContent = '请至少选择一个模型'; return; }
  if (experiment && $('image').files[0].size + (!clean ? $('reference').files[0]?.size || 0 : 0) > 25 * 1024 * 1024) { $('status').textContent = '图片合计不能超过 25 MB。'; return; }
  const form = new FormData(); form.set('image', $('image').files[0]); form.set('device', 'auto'); form.set('sigma', experiment ? (clean ? $('noise-level').value : $('sigma').value) : '');
  form.set('synthetic', String(clean)); form.set('seed', clean ? $('seed').value : '42');
  if (experiment && !clean && $('reference').files.length) form.set('reference', $('reference').files[0]);
  const controls = [...document.querySelectorAll('form input, form select, form button, nav button')];
  const disabled = controls.map(c => c.disabled); controls.forEach(c => c.disabled = true);
  clearResults(); $('result-panel').hidden = false;
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
  } finally { controls.forEach((c, i) => c.disabled = disabled[i]); $('export').disabled = !results.length; updateSubmit(); }
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
    label.append(check, document.createTextNode(methods[model.id] || model.name));
    $(Object.hasOwn(methods, model.id) ? 'compare-models' : 'more-models').append(label);
    check.onchange = updateSubmit;
  });
  selectTab(tab);
  updateSubmit();
}
init().catch(error => { $('status').className = 'error'; $('status').textContent = `加载失败：${error.message}`; });

function updateSubmit() {
  const count = document.querySelectorAll('.compare-item:checked').length;
  $('submit').disabled = !validImage || (tab === 'experiment' ? !count : !models.some(m => m.id === defaultModel && m.available));
  $('submit').textContent = tab === 'experiment' ? (count ? `开始对比（${count} 种）` : '请至少选择一种方法') : '开始去噪';
}
$('image').onchange = async () => {
  validImage = false; updateSubmit();
  $('selection-error').textContent = ''; $('status').textContent = '';
  clearResults();
  $('preview').hidden = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = '';
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
