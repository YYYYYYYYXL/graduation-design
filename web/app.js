const $ = id => document.getElementById(id);
let defaultModel = '', available = false, processing = false, validImage = false, previewUrl = '';
const node = (tag, text, className) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (className) el.className = className; return el; };
async function api(url, options) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { throw new Error('服务暂时没有响应，请稍后重试。'); }
  if (!response.ok) throw new Error(data.error || '请求失败');
  return data;
}
function clearResults() {
  $('results').replaceChildren(); $('single-score-panels').replaceChildren(); $('result-panel').hidden = true; $('status').textContent = '';
}
function updateSubmit() { $('submit').disabled = processing || !validImage || !available; }
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
function imageFigure(label, url, onRemove) {
  const figure = node('figure'), link = node('a'), img = node('img');
  figure.dataset.imageUrl = url;
  link.href = url; link.target = '_blank'; link.rel = 'noopener';
  img.src = url; img.alt = label; img.loading = 'lazy';
  const caption = node('figcaption', undefined, 'image-caption');
  caption.append(node('span', label), removeButton(label, onRemove || (() => figure.remove())));
  link.append(img); figure.append(caption, link); return figure;
}
function displayResult(result) {
  const images = node('div', undefined, 'images');
  images.append(imageFigure('输入图', result.input_url), imageFigure('去噪结果', result.output_url, clearResults));
  const download = node('a', '下载去噪结果');
  download.href = result.output_url + '?download=1'; download.download = '';
  $('results').append(node('h3', result.model_name), images, download);
  const score = value => Number.isFinite(value) ? value.toFixed(2) : '未测量';
  $('single-score-panels').textContent = 'BRISQUE ↓　输入 ' + score(result.input_metrics.brisque) + '　输出 ' + score(result.metrics.brisque);
  for (const note of result.metrics.quality_notes || []) $('single-score-panels').append(node('p', note, 'muted'));
  $('result-panel').hidden = false;
}
$('denoise-form').onsubmit = async event => {
  event.preventDefault();
  if (!validImage || !available || processing) return;
  const form = new FormData();
  form.set('image', $('image').files[0]); form.set('model', defaultModel); form.set('experiment', 'true');
  clearResults(); processing = true; updateSubmit();
  $('image').disabled = $('clear-image').disabled = true;
  $('status').className = ''; $('status').textContent = '正在去噪…';
  try {
    displayResult(await api('/api/denoise', {method: 'POST', body: form}));
    $('status').textContent = '去噪完成。';
  } catch (error) {
    $('status').className = 'error'; $('status').textContent = error.message;
  } finally {
    processing = false; $('image').disabled = $('clear-image').disabled = false;
    document.querySelectorAll('.remove-image').forEach(button => button.disabled = false);
    updateSubmit();
  }
};
async function init() {
  const [models, settings] = await Promise.all([api('/api/models'), api('/api/settings')]);
  defaultModel = settings.default_model;
  available = models.some(model => model.id === defaultModel && model.available);
  if (!available) $('selection-error').textContent = '默认模型权重不可用。';
  updateSubmit();
}
init().catch(error => { $('status').className = 'error'; $('status').textContent = error.message; });
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
