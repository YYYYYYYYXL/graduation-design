const $ = id => document.getElementById(id);
let models = [], tab = 'normal', results = [], defaultModel = '', previewUrl = '', validImage = false;
let detailImages = [];
let processing = false;
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
  renderSingleImageCharts(results);
  $('result-panel').hidden = true; $('export').disabled = true;
  $('status').textContent = '';
  detailImages = []; $('detail-grid').replaceChildren(); $('detail-panel').hidden = true;
  $('detail-panel').open = true; $('detail-mode').value = 'detail';
}
function selectTab(next) {
  if (tab !== next) clearResults();
  tab = next;
  const experiment = tab === 'experiment';
  document.querySelectorAll('[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === next));
  for (const id of ['experiment-input', 'experiment-options']) {
    $(id).hidden = !experiment; $(id).disabled = !experiment;
  }
  $('form-title').textContent = experiment ? '实验对比' : '图片去噪';
  $('result-title').textContent = experiment ? '对比结果' : '去噪结果';
  $('export').hidden = !experiment;
  $('full-views').open = !experiment;
  $('results').className = experiment ? 'result-grid' : '';
  updateMode();
}
function updateMode() {
  const clean = tab === 'experiment' && $('synthetic').checked;
  $('upload-label').textContent = tab === 'normal' ? '上传图片' : clean ? '上传清晰照片' : '上传带噪照片';
  $('noise-options').hidden = !clean; $('noise-level').disabled = !clean;
  $('reference-options').hidden = clean; $('reference').disabled = clean;
  updateSubmit();
}
document.querySelectorAll('[data-tab]').forEach(b => b.onclick = () => selectTab(b.dataset.tab));
document.querySelectorAll('[name="input-mode"]').forEach(radio => radio.onchange = () => { clearResults(); updateMode(); });
$('clear-reference').onclick = () => { $('reference').value = ''; clearResults(); };
$('reference').onchange = clearResults;
$('clear-image').onclick = () => {
  if (processing) return;
  $('image').value = ''; validImage = false;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = ''; $('preview-image').removeAttribute('src');
  $('preview').hidden = true; $('clear-image').hidden = true;
  $('file-info').textContent = ''; $('selection-error').textContent = '';
  clearResults(); updateSubmit();
};
function removeButton(label, action) {
  const button = node('button', '移除图片', 'remove-image');
  button.type = 'button'; button.disabled = processing;
  button.setAttribute('aria-label', `移除${label}`);
  button.onclick = () => { if (!processing) action(); };
  return button;
}
function removeImageViews(url) {
  document.querySelectorAll('figure[data-image-url]').forEach(figure => {
    if (figure.dataset.imageUrl === url) figure.remove();
  });
  detailImages = detailImages.filter(item => item.url !== url);
  drawDetails();
  $('detail-panel').hidden = !detailImages.length;
  $('shared-input').hidden = !$('shared-input').children.length;
}
function removeResult(result) {
  removeImageViews(result.output_url);
  document.querySelectorAll('article[data-result-id]').forEach(article => {
    if (article.dataset.resultId === result.id) article.remove();
  });
  results = results.filter(item => item.id !== result.id);
  if (!results.length) clearResults();
  else {
    renderSingleImageCharts(results);
    if (tab === 'experiment') showSummary();
  }
  $('status').textContent = results.length ? `当前保留 ${results.length} 项结果。` : '已移除结果，可重新开始去噪。';
  $('export').disabled = !results.length;
}
function imageFigure(label, url, onRemove = () => removeImageViews(url)) {
  const figure = node('figure'), link = node('a'), img = node('img');
  figure.dataset.imageUrl = url;
  link.href = url; link.target = '_blank'; link.rel = 'noopener';
  img.src = url; img.alt = label; img.loading = 'lazy';
  const caption = node('figcaption', undefined, 'image-caption');
  caption.append(node('span', label), removeButton(label, onRemove));
  link.append(img); figure.append(caption, link); return figure;
}
function showSummary() {
  const first = results[0];
  const summary = $('comparison-summary'); summary.replaceChildren(); summary.hidden = false;
  summary.append(node('p', `处理尺寸 ${first.width} × ${first.height} 像素 · ${first.synthetic ? `合成高斯噪声 σ=${first.requested_sigma} · 随机种子 ${first.seed}` : '带噪输入'}${first.resized ? ` · 原图 ${first.original_width} × ${first.original_height}` : ''}`, 'muted'));
}
function format(value, digits = 3) { return value === null || value === undefined ? '未测量' : typeof value === 'number' ? value.toFixed(digits) : value; }
function addResult(result) {
  const experiment = tab === 'experiment';
  renderSingleImageCharts(results);
  const article = node('article', undefined, 'result');
  article.dataset.resultId = result.id;
  if (result.resized) article.append(node('p', `原图 ${result.original_width} × ${result.original_height} →  ${result.width} × ${result.height} px`, 'muted'));
  if (experiment) {
    article.append(node('h3', result.model_name));
    if (results.length === 1) {
      $('shared-input').hidden = false;
      $('shared-input').append(imageFigure('共同输入 · 带噪照片', result.input_url));
      addDetailImage('带噪输入', result.input_url, result.input_metrics);
      if (result.reference_url) {
        $('shared-input').append(imageFigure('干净参考图', result.reference_url));
        addDetailImage('干净参考图', result.reference_url, null, true);
      }
    }
    addDetailImage(result.model_name, result.output_url, result.metrics, false, () => removeResult(result));
    showSummary();
  }
  const images = node('div', undefined, 'images');
  if (!experiment) images.append(imageFigure('输入图', result.input_url));
  images.append(imageFigure('去噪结果', result.output_url, () => removeResult(result))); article.append(images);
  const links = node('div', undefined, 'links');
  const a = node('a', '下载图片'); a.href = result.output_url + '?download=1'; a.download = ''; links.append(a);
  if (experiment) {
    const record = node('a', '完整参数 JSON'); record.href = result.json_url; links.append(record);
    const info = node('p', `权重文件 ${format(result.weights_bytes / 1024 ** 2, 2)} MiB · ${result.tiled ? '分块推理' : '整图推理'}${result.model === 'bm3d' ? ` · 实际噪声强度 σ=${format(result.sigma, 2)}` : ''}`, 'muted');
    article.append(info);
  }
  article.append(links); $('results').append(article);
}
function addDetailImage(label, url, metrics, isReference = false, onRemove = () => removeImageViews(url)) {
  const figure = node('figure'), canvas = node('canvas'), img = new Image();
  figure.dataset.imageUrl = url;
  canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `${label}的局部放大`);
  const caption = node('figcaption', undefined, 'image-caption');
  caption.append(node('span', label), removeButton(label, onRemove));
  const scores = !metrics ? '参考图（GT）' : results[0].reference_provided
    ? 'PSNR ' + format(metrics.psnr, 2) + ' dB · SSIM ' + format(metrics.ssim, 4)
    : 'BRISQUE ' + format(metrics.brisque, 2);
  figure.dataset.label = label; figure.dataset.scores = scores;
  canvas.hidden = true;
  figure.append(caption, canvas, node('small', scores)); $('detail-grid').append(figure);
  const item = {img, canvas, url, isReference, ready: false}; detailImages.push(item);
  img.onload = () => { item.ready = true; if ($('detail-panel').open) drawDetails(); };
  img.onerror = () => figure.append(node('p', '图片加载失败，可点击上方图片重新查看。', 'error'));
  img.src = url; $('detail-panel').hidden = false; drawDetails();
}
function drawDetails() {
  if (!results.length) return;
  const {width, height} = results[0];
  const crop = cropRegion(width, height, $('detail-size').value, $('detail-x').value, $('detail-y').value);
  const {left, top, size} = crop;
  $('detail-location').textContent = `ROI (${left}, ${top}) · ${size} × ${size} px`;
  const target = detailImages.find(item => item.isReference);
  const mode = $('detail-mode');
  mode.options[1].disabled = !target?.ready;
  if (!target?.ready) mode.value = 'detail';
  $('detail-legend').hidden = mode.value !== 'error';
  $('detail-legend').textContent = plateModes[mode.value];
  $('detail-legend').classList.toggle('error-scale', mode.value === 'error');
  $('detail-png').disabled = !detailImages.some(item => item.ready);
  detailImages.forEach(({img, canvas, ready}) => {
    if (ready) drawCrop(canvas, img, crop, mode.value === 'error' ? target.img : null);
  });
}
$('detail-png').onclick = () => exportPlate($('detail-grid'),
  $('image').files[0].name + (results[0].synthetic ? ' · σ=' + results[0].requested_sigma + ' · seed=' + results[0].seed : ''),
  $('detail-location').textContent, $('detail-legend').textContent);
['detail-x', 'detail-y', 'detail-size', 'detail-mode'].forEach(id => $(id).oninput = drawDetails);
$('detail-panel').ontoggle = () => { if ($('detail-panel').open) drawDetails(); };
$('denoise-form').onsubmit = async event => {
  event.preventDefault();
  if (!validImage || processing) return;
  const experiment = tab === 'experiment';
  const ids = experiment ? [...document.querySelectorAll('.compare-item:checked')].map(c => c.value) : [defaultModel];
  const clean = experiment && $('synthetic').checked;
  if (!ids.length) { $('status').textContent = '请至少选择一个模型'; return; }
  if (experiment && $('image').files[0].size + (!clean ? $('reference').files[0]?.size || 0 : 0) > 50 * 1024 * 1024) { $('status').textContent = '图片合计不能超过 50 MB。'; return; }
  const form = new FormData(); form.set('image', $('image').files[0]); form.set('device', 'auto'); form.set('sigma', clean ? $('noise-level').value : '');
  // 单图去噪也复用已有的无参考评分，供去噪前后图表使用。
  form.set('experiment', 'true');
  form.set('synthetic', String(clean)); form.set('seed', '42');
  if (experiment && !clean && $('reference').files.length) form.set('reference', $('reference').files[0]);
  processing = true;
  const controls = [...document.querySelectorAll('form input, form select, form button, nav button')];
  const disabled = controls.map(c => c.disabled); controls.forEach(c => c.disabled = true);
  clearResults(); $('result-panel').hidden = false;
  $('status').className = ''; const errors = [];
  try {
    for (const [index, id] of ids.entries()) {
      const label = models.find(m => m.id === id).name;
      $('status').textContent = experiment ? `正在处理 ${index + 1}/${ids.length}：${label}` : '正在去噪，请稍候…';
      form.set('model', id);
      try {
        const result = await api('/api/denoise', {method: 'POST', body: form});
        if (results.length && (result.input_sha256 !== results[0].input_sha256 || result.reference_sha256 !== results[0].reference_sha256 || result.width !== results[0].width || result.height !== results[0].height)) throw new Error('输入或参考图与本组不一致，不能合并比较，请重新开始。');
        results.push(result); addResult(result);
      }
      catch (error) { $('full-views').open = true; errors.push(`${label}：${error.message}`); $('results').append(node('p', errors.at(-1), 'error')); }
    }
    $('status').textContent = errors.length ? `成功 ${results.length} 项，失败 ${errors.length} 项，原因见下方。` : experiment ? `已完成 ${results.length} 项对比。` : '去噪完成。';
    $('status').className = errors.length ? 'error' : '';
  } finally {
    processing = false;
    controls.forEach((c, i) => c.disabled = disabled[i]);
    document.querySelectorAll('.remove-image').forEach(button => button.disabled = false);
    $('export').disabled = !results.length; updateSubmit();
  }
};
$('export').onclick = () => {
  const fields = ['model', 'model_name', 'metric_mode', 'psnr', 'ssim', 'brisque', 'estimated_noise', 'quality_notes', 'seconds', 'load_seconds', 'total_seconds', 'parameters', 'weights_bytes', 'device', 'sigma', 'requested_sigma', 'synthetic', 'seed', 'tiled', 'id', 'created_at', 'original_width', 'original_height', 'width', 'height', 'resized', 'input_sha256', 'reference_sha256'];
  const first = results[0];
  const baseline = {...first, ...first.input_metrics, model: 'input', model_name: '去噪前', seconds: null, load_seconds: null, total_seconds: null, parameters: null, weights_bytes: null, device: null, sigma: null, tiled: null};
  const rows = [fields, ...[baseline, ...results.map(r => ({...r, ...r.metrics}))].map(r => fields.map(key => Array.isArray(r[key]) ? r[key].join('；') : r[key]))];
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
    check.checked = true;
    label.append(check, document.createTextNode(model.name));
    $('compare-models').append(label);
    check.onchange = updateSubmit;
  });
  selectTab(tab);
  updateSubmit();
}
init().catch(error => { $('status').className = 'error'; $('status').textContent = `加载失败：${error.message}`; });

function updateSubmit() {
  const count = document.querySelectorAll('.compare-item:checked').length;
  $('submit').disabled = processing || !validImage || (tab === 'experiment' ? !count : !models.some(m => m.id === defaultModel && m.available));
  $('submit').textContent = tab === 'experiment' ? (count ? `开始对比（${count} 项）` : '请至少选择一种方法') : '开始去噪';
}
$('image').onchange = async () => {
  validImage = false; updateSubmit();
  $('selection-error').textContent = ''; $('status').textContent = '';
  clearResults();
  $('preview').hidden = true; $('preview-image').hidden = false;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = '';
  const file = $('image').files[0];
  $('clear-image').hidden = !file;
  if (!file) return;
  if (file.size > 50 * 1024 * 1024) { $('selection-error').textContent = '图片不能超过 50 MB。'; return; }
  const url = URL.createObjectURL(file); previewUrl = url;
  const img = new Image(); img.src = url;
  try {
    await img.decode();
    if (previewUrl !== url) return;
    if (Math.min(img.naturalWidth, img.naturalHeight) < 11 || img.naturalWidth * img.naturalHeight > 100000000) throw new Error('图片每边至少 11 像素，原图总像素不超过 1 亿。');
    $('preview-image').src = url;
    $('file-info').textContent = `${file.name} · ${img.naturalWidth} × ${img.naturalHeight}`;
    $('preview').hidden = false; validImage = true;
  } catch (error) {
    if (previewUrl !== url) return;
    if (/\.hei[cf]$/i.test(file.name) || /^image\/hei[cf]/i.test(file.type)) {
      $('preview-image').hidden = true; $('preview').hidden = false;
      $('file-info').textContent = `${file.name} · 预览不可用`;
      validImage = true;
    } else $('selection-error').textContent = error.message.startsWith('图片') ? error.message : '无法读取图片，请选择 JPG、PNG、WebP、BMP 或 HEIC/HEIF 文件。';
  }
  updateSubmit();
};
