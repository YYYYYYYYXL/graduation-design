function renderSingleImageCharts(results) {
  const host = $('single-score-panels');
  host.replaceChildren();
  $('single-score-charts').hidden = !results.length;
  if (!results.length) return;
  const first = results[0];
  const columns = [{model_name: '带噪输入', metrics: first.input_metrics}, ...results];
  const metrics = first.reference_provided
    ? [['psnr', 'PSNR ↑ (dB)', 2], ['ssim', 'SSIM ↑', 3]]
    : [['brisque', 'BRISQUE ↓', 2]];
  const table = node('table'), head = table.createTHead().insertRow(), body = table.createTBody();
  for (const label of ['指标', ...columns.map(item => item.model_name)]) {
    const th = node('th', label); th.scope = 'col'; head.append(th);
  }
  for (const [key, label, digits] of metrics) {
    const row = body.insertRow(), name = node('th', label); name.scope = 'row'; row.append(name);
    const numbers = columns.map(item => {
      const value = item.metrics[key];
      return value === '∞' ? Infinity : Number.isFinite(value) ? Number(value.toFixed(digits)) : null;
    });
    const measured = numbers.slice(1).filter(value => value !== null);
    const best = key === 'brisque' ? Math.min(...measured) : Math.max(...measured);
    columns.forEach((item, i) => {
      const cell = row.insertCell(), value = format(item.metrics[key], digits);
      if (i && numbers[i] === best) cell.append(node('strong', value));
      else cell.textContent = value;
    });
  }
  host.append(table);
  if (!first.reference_provided) {
    for (const row of columns) for (const note of row.metrics.quality_notes || []) {
      host.append(node('p', row.model_name + ' · ' + note, 'muted'));
    }
  }
}
