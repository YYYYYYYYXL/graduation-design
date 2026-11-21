// Restormer (CVPR 2022), Fig. 6: scene, noisy patch, reference patch, method patches.
function cropRegion(width, height, requested, horizontal, vertical) {
  const size = Math.min(Number(requested), width, height);
  return {size, left: Math.round((width - size) * horizontal / 100),
    top: Math.round((height - size) * vertical / 100)};
}

function drawPatch(canvas, image, {left, top, size}, overview) {
  const side = 320;
  canvas.width = canvas.height = side;
  const ctx = canvas.getContext('2d');
  if (overview) {
    const scale = side / Math.max(image.naturalWidth, image.naturalHeight);
    const w = image.naturalWidth * scale, h = image.naturalHeight * scale;
    const x = (side - w) / 2, y = (side - h) / 2;
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, side, side);
    ctx.drawImage(image, x, y, w, h);
    ctx.strokeStyle = '#dc2727'; ctx.lineWidth = 2;
    ctx.strokeRect(x + left * scale, y + top * scale, size * scale, size * scale);
  } else {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(image, left, top, size, size, 0, 0, side, side);
  }
  canvas.hidden = false;
}

const plateReady = new WeakMap();

function paperRow(title, views, controls) {
  const row = document.createElement('section'), heading = document.createElement('h4');
  const strip = document.createElement('div');
  heading.textContent = title; strip.className = 'paper-strip';
  row.className = 'paper-row'; row.append(heading, strip);
  const sources = new Map(), tiles = [];
  function redraw() {
    const source = tiles.find(tile => tile.img?.naturalWidth)?.img;
    if (!source) return;
    const {naturalWidth: width, naturalHeight: height} = source;
    const roi = cropRegion(width, height, controls.size.value, controls.x.value, controls.y.value);
    row.dataset.region = 'ROI (' + roi.left + ', ' + roi.top + ') · ' + roi.size + ' × ' + roi.size + ' px';
    for (const {img, canvas, status, overview} of tiles) {
      if (!img?.naturalWidth) continue;
      if (img.naturalWidth !== width || img.naturalHeight !== height) {
        canvas.hidden = true; status.textContent = '尺寸不一致'; continue;
      }
      status.textContent = ''; drawPatch(canvas, img, roi, overview);
    }
  }
  const scoreValue = value => value === '∞' ? Infinity :
    Number.isFinite(value) ? Number(value.toFixed(2)) : null;
  const best = Math.max(...views.map(view => scoreValue(view.modelPsnr)).filter(value => value !== null));
  for (const view of views) {
    const figure = document.createElement('figure'), canvas = document.createElement('canvas');
    const caption = document.createElement('figcaption'), score = document.createElement('span');
    const status = document.createElement('span'), link = document.createElement(view.url ? 'a' : 'div');
    if (scoreValue(view.modelPsnr) === best) figure.classList.add('best-score');
    figure.dataset.label = view.label; figure.dataset.score = view.score || '';
    if (view.url) { link.href = view.url; link.target = '_blank'; link.rel = 'noopener'; }
    canvas.hidden = true; canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', view.label);
    link.className = 'paper-image'; status.textContent = view.url ? '加载中' : view.status || '待评测';
    score.textContent = figure.dataset.score; caption.append(view.label, score);
    link.append(canvas, status); figure.append(link, caption); strip.append(figure);
    if (view.url && !sources.has(view.url)) {
      const img = new Image(); sources.set(view.url, img);
      img.onload = redraw;
      img.onerror = () => tiles.filter(tile => tile.img === img).forEach(tile => tile.status.textContent = '图像不可用');
      img.src = view.url;
    }
    tiles.push({img: sources.get(view.url), canvas, status, overview: view.overview});
  }
  plateReady.set(row, Promise.allSettled([...sources.values()].map(img => img.decode())).then(redraw));
  return {row, redraw};
}

async function exportPlate(host, title) {
  const rows = [...host.querySelectorAll('.paper-row')];
  if (!rows.length) return;
  await Promise.all(rows.map(row => plateReady.get(row)));
  const columns = Math.max(...rows.map(row => row.querySelectorAll('figure').length));
  const sheet = document.createElement('canvas'), col = 336, height = 422;
  sheet.width = columns * col + 24; sheet.height = 50 + rows.length * height;
  const ctx = sheet.getContext('2d');
  ctx.fillStyle = 'white'; ctx.fillRect(0, 0, sheet.width, sheet.height);
  ctx.fillStyle = '#192328'; ctx.font = 'bold 22px sans-serif'; ctx.fillText(title, 12, 31);
  rows.forEach((row, index) => {
    const y = 50 + index * height;
    ctx.textAlign = 'left'; ctx.font = '16px sans-serif';
    ctx.fillText([row.querySelector('h4').textContent, row.dataset.region].filter(Boolean).join(' · '), 12, y + 16);
    row.querySelectorAll('figure').forEach((figure, i) => {
      const x = 12 + i * col, canvas = figure.querySelector('canvas');
      ctx.textAlign = 'center';
      if (!canvas.hidden) ctx.drawImage(canvas, x, y + 28);
      else ctx.fillText(figure.querySelector('.paper-image span').textContent, x + 160, y + 188);
      ctx.font = 'bold 18px sans-serif'; ctx.fillText(figure.dataset.label, x + 160, y + 376, 320);
      ctx.font = figure.classList.contains('best-score') ? 'bold 17px sans-serif' : '17px sans-serif';
      ctx.fillText(figure.dataset.score, x + 160, y + 401, 320);
    });
  });
  sheet.toBlob(blob => {
    if (!blob) return;
    const link = document.createElement('a'); link.href = URL.createObjectURL(blob);
    link.download = 'denoising-comparison.png'; link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }, 'image/png');
}
