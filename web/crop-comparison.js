function cropRegion(width, height, requested, horizontal, vertical) {
  const size = Math.min(Number(requested), width, height);
  return {
    size,
    left: Math.round((width - size) * Number(horizontal) / 100),
    top: Math.round((height - size) * Number(vertical) / 100),
  };
}

function drawCrop(canvas, image, {left, top, size}) {
  canvas.width = 320;
  canvas.height = 320;
  const context = canvas.getContext('2d');
  context.imageSmoothingEnabled = false;
  context.drawImage(image, left, top, size, size, 0, 0, 320, 320);
}
