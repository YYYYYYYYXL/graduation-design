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
  $('result-panel').hidden = true; $('export').disabled = true;
  $('status').textContent = '';
  detailImages = []; $('detail-grid').replaceChildren(); $('detail-panel').hidden = true;
  $('detail-panel').open = false;
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
  updateMetricGuide();
  updateSubmit();
}
function updateMetricGuide() {
  const paired = $('synthetic').checked || $('reference').files.length > 0;
  $('metric-guide').textContent = paired
    ? '本次看还原准确度：PSNR、SSIM 越高通常越接近干净图，同时比较用时和局部细节。'
    : '本次看无参考质量：BRISQUE 越低通常越好。噪声估计仅作辅助，变糊也可能让噪声数值变低；请结合放大图判断。';
}
document.querySelectorAll('[data-tab]').forEach(b => b.onclick = () => selectTab(b.dataset.tab));
document.querySelectorAll('[name="input-mode"]').forEach(radio => radio.onchange = () => { clearResults(); updateMode(); });
$('clear-reference').onclick = () => { $('reference').value = ''; clearResults(); updateMetricGuide(); };
$('reference').onchange = () => { clearResults(); updateMetricGuide(); };
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
  else if (tab === 'experiment') showSummary();
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
  const first = results[0], hasReference = first.reference_provided;
  const summary = $('comparison-summary'); summary.replaceChildren(); summary.hidden = false;
  summary.append(node('p', hasReference ? 'PSNR、SSIM ↑：越高通常越接近参考图。先与“去噪前”比较，再看各模型的结果。' : 'BRISQUE ↓：0～100，越低通常表示感知质量越好，不是准确率。噪声估计是辅助数值，不能单凭它判断谁更好。', 'metric-guide'));
  summary.append(node('p', `本次处理 ${first.width} × ${first.height} 像素 · ${first.synthetic ? `合成高斯噪声 σ=${first.requested_sigma} · 随机种子 ${first.seed}` : '上传的带噪照片，未额外加噪'}${first.resized ? ` · 原图 ${first.original_width} × ${first.original_height} 已缩小，指标针对缩小后的图片` : ''}`, 'muted'));
  const wrapper = node('div', undefined, 'table-scroll'), table = node('table');
  const caption = node('caption', '本次对比汇总'); table.append(caption);
  const head = node('thead'), heading = node('tr');
  const labels = ['模型 / 方法', ...(hasReference ? ['PSNR ↑ (dB)', 'SSIM ↑'] : ['BRISQUE ↓', '噪声估计（辅助）']), '去噪 (s)', '加载 (s)', '总处理 (s)', '参数量 (M)', '设备'];
  labels.forEach(label => { const th = node('th', label); th.scope = 'col'; heading.append(th); });
  head.append(heading); table.append(head);
  const body = node('tbody');
  const scores = value => hasReference ? [format(value.psnr), format(value.ssim, 4)] : [format(value.brisque, 2), format(value.estimated_noise, 2)];
  const baseline = node('tr', undefined, 'baseline');
  ['去噪前 · 原始输入', ...scores(first.input_metrics), '—', '—', '—', '—', '—'].forEach(v => baseline.append(node('td', v)));
  body.append(baseline);
  results.forEach(r => {
    const row = node('tr');
    [r.model_name, ...scores(r.metrics), format(r.seconds), format(r.load_seconds), format(r.total_seconds), r.parameters === null ? '不适用' : format(r.parameters / 1e6), r.device.toUpperCase()].forEach(v => row.append(node('td', v)));
    body.append(row);
  });
  table.append(body); wrapper.append(table); summary.append(wrapper);
  const help = node('details'); help.append(node('summary', '怎么看这些指标和用时？'));
  help.append(node('p', '总处理时间包含读写、加载、去噪和评分，不含上传传输；加载受缓存影响。BM3D 去噪用时包含噪声估计，神经网络包含分块和设备传输；CPU 与 GPU 速度不能直接当作算法优劣。参数量单位 M 是百万个参数，表示网络规模，不代表质量或内存占用。' + (results.some(r => r.tiled) ? '部分结果使用分块推理。' : ''), 'muted'));
  help.append(node('p', '这张图的结果不代表所有照片。可以比较同一网络的 DIV2K 与 SIDD 两个版本，但差异也受训练设置影响；也可以比较五个模型与 BM3D。P2N 为现有网络直接推理。', 'muted'));
  summary.append(help);
  const notes = [...first.input_metrics.quality_notes.map(note => `去噪前：${note}`), ...results.flatMap(r => r.metrics.quality_notes.map(note => `${r.model_name}：${note}`))];
  notes.forEach(note => summary.append(node('p', note, 'muted')));
}
function format(value, digits = 3) { return value === null || value === undefined ? '未测量' : typeof value === 'number' ? value.toFixed(digits) : value; }
function addResult(result) {
  const experiment = tab === 'experiment';
  const article = node('article', undefined, 'result');
  article.dataset.resultId = result.id;
  if (result.resized) article.append(node('p', `原图 ${result.original_width} × ${result.original_height}，已等比例缩小至 ${result.width} × ${result.height}；下载图片为此尺寸。`, 'muted'));
  if (experiment) {
    article.append(node('h3', result.model_name));
    if (results.length === 1) {
      $('shared-input').hidden = false;
      $('shared-input').append(imageFigure('共同输入 · 带噪照片', result.input_url));
      addDetailImage('去噪前', result.input_url);
      if (result.reference_url) {
        $('shared-input').append(imageFigure('干净参考图', result.reference_url));
        addDetailImage('干净参考图', result.reference_url);
      }
    }
    addDetailImage(result.model_name, result.output_url, () => removeResult(result));
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
function addDetailImage(label, url, onRemove = () => removeImageViews(url)) {
  const figure = node('figure'), canvas = node('canvas'), img = new Image();
  figure.dataset.imageUrl = url;
  canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', `${label}的局部放大`);
  const caption = node('figcaption', undefined, 'image-caption');
  caption.append(node('span', label), removeButton(label, onRemove));
  figure.append(caption, canvas); $('detail-grid').append(figure);
  const item = {img, canvas, url, ready: false}; detailImages.push(item);
  img.onload = () => { item.ready = true; if ($('detail-panel').open) drawDetails(); };
  img.onerror = () => figure.append(node('p', '图片加载失败，可点击上方图片重新查看。', 'error'));
  img.src = url; $('detail-panel').hidden = false;
}
function drawDetails() {
  if (!results.length) return;
  const {width, height} = results[0];
  const size = Math.min(Number($('detail-size').value), width, height);
  const left = Math.round((width - size) * Number($('detail-x').value) / 100);
  const top = Math.round((height - size) * Number($('detail-y').value) / 100);
  $('detail-location').textContent = `区域左上角 (${left}, ${top})，取 ${size} × ${size} 个原始像素，同倍数放大。`;
  detailImages.forEach(({img, canvas, ready}) => {
    if (!ready) return;
    canvas.width = 320; canvas.height = 320;
    const context = canvas.getContext('2d'); context.imageSmoothingEnabled = false;
    context.drawImage(img, left, top, size, size, 0, 0, 320, 320);
  });
}
['detail-x', 'detail-y', 'detail-size'].forEach(id => $(id).oninput = drawDetails);
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
  form.set('experiment', String(experiment));
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
      catch (error) { errors.push(`${label}：${error.message}`); $('results').append(node('p', errors.at(-1), 'error')); }
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
    if (model.id === 'p2n_div2k') label.append(node('small', '使用现有网络直接推理'));
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
    $('file-info').textContent = `${file.name} · ${img.naturalWidth} × ${img.naturalHeight}${img.naturalWidth * img.naturalHeight > 4000000 ? " · 大图将自动缩小后去噪" : ""}`;
    $('preview').hidden = false; validImage = true;
  } catch (error) {
    if (previewUrl !== url) return;
    if (/\.hei[cf]$/i.test(file.name) || /^image\/hei[cf]/i.test(file.type)) {
      $('preview-image').hidden = true; $('preview').hidden = false;
      $('file-info').textContent = `${file.name} · 浏览器无法预览此格式，可以直接开始处理；系统会读取照片并自动调整大图尺寸。`;
      validImage = true;
    } else $('selection-error').textContent = error.message.startsWith('图片') ? error.message : '无法读取图片，请选择 JPG、PNG、WebP、BMP 或 HEIC/HEIF 文件。';
  }
  updateSubmit();
};
