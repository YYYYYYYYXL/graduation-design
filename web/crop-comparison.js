function cropRegion(width, height, requested, horizontal, vertical) {
  const size = Math.min(Number(requested), width, height);
  return {size, left: Math.round((width - size) * horizontal / 100),
    top: Math.round((height - size) * vertical / 100)};
}

// Every column uses the same coordinate system, zoom and error scale.
function drawCrop(canvas, image, {left, top, size}, reference) {
  const w = 320, overview = 176, gap = 10;
  canvas.width = w; canvas.height = overview + gap + w;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#f2f3f2'; ctx.fillRect(0, 0, w, overview);
  const scale = Math.min(w / image.naturalWidth, overview / image.naturalHeight);
  const x = (w - image.naturalWidth * scale) / 2, y = (overview - image.naturalHeight * scale) / 2;
  ctx.drawImage(image, x, y, image.naturalWidth * scale, image.naturalHeight * scale);
  for (const [lineWidth, color] of [[4, '#fff'], [2, '#ad321b']]) {
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth;
    ctx.strokeRect(x + left * scale, y + top * scale, size * scale, size * scale);
  }
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(image, left, top, size, size, 0, overview + gap, w, w);
  if (reference) {
    const scratch = document.createElement('canvas');
    scratch.width = scratch.height = size;
    const local = scratch.getContext('2d', {willReadFrequently: true});
    local.drawImage(reference, left, top, size, size, 0, 0, size, size);
    const target = local.getImageData(0, 0, size, size).data;
    local.drawImage(image, left, top, size, size, 0, 0, size, size);
    const output = local.getImageData(0, 0, size, size);
    for (let i = 0; i < target.length; i += 4) {
      const error = (Math.abs(output.data[i] - target[i]) + Math.abs(output.data[i+1] - target[i+1])
        + Math.abs(output.data[i+2] - target[i+2])) / 3;
      output.data[i] = output.data[i+1] = output.data[i+2] = Math.min(255, error * 255 / 32);
    }
    local.putImageData(output, 0, 0);
    ctx.drawImage(scratch, 0, overview + gap, w, w);
  }
  canvas.hidden = false;
}

function exportPlate(grid, title, region, mode) {
  const figures = [...grid.querySelectorAll('figure')].filter(f => f.querySelector('canvas:not([hidden])'));
  if (!figures.length) return;
  const columns = Math.min(4, figures.length), col = 348, row = 582;
  const sheet = document.createElement('canvas');
  sheet.width = Math.max(700, columns * col + 32);
  sheet.height = 126 + Math.ceil(figures.length / columns) * row;
  const ctx = sheet.getContext('2d');
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, sheet.width, sheet.height);
  ctx.fillStyle = '#1a3038'; ctx.font = 'bold 22px sans-serif';
  ctx.fillText(title, 22, 32, sheet.width - 44);
  ctx.font = '16px sans-serif'; ctx.fillText(region, 22, 60, sheet.width - 44);
  ctx.fillText(mode, 22, 87, sheet.width - 44);
  figures.forEach((f, index) => {
    const x = 22 + (index % columns) * col, y = 120 + Math.floor(index / columns) * row;
    ctx.font = 'bold 18px sans-serif'; ctx.fillText(f.dataset.label, x, y, 320);
    ctx.drawImage(f.querySelector('canvas'), x, y + 14);
    ctx.font = '15px sans-serif'; ctx.fillText(f.dataset.scores || '', x, y + 546, 320);
  });
  sheet.toBlob(blob => {
    if (!blob) return;
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob); link.download = 'denoising-comparison.png'; link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }, 'image/png');
}

const plateModes = {
  detail: '纹理细节',
  error: 'RGB 平均绝对误差 · 0–32（8-bit）'
};
