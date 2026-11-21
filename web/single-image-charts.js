function renderSingleImageCharts(results) {
  const host = $('single-score-panels');
  host.replaceChildren();
  $('single-score-charts').hidden = !results.length;
  if (!results.length) return;
  const first = results[0], paired = first.reference_provided;
  const metrics = paired ? [['psnr', 'PSNR ↑ (dB)', 2], ['ssim', 'SSIM ↑', 4]] : [['brisque', 'BRISQUE ↓', 2]];
  const table = node('table'), head = table.createTHead().insertRow(), body = table.createTBody();
  for (const label of ['方法', ...metrics.flatMap(([, label]) => [label, 'Δ 对输入']), '耗时 (s)', '设备']) {
    const th = node('th', label); th.scope = 'col'; head.append(th);
  }
  const rows = [{model_name: '带噪输入', metrics: first.input_metrics}, ...results];
  for (const [index, result] of rows.entries()) {
    const tr = body.insertRow(), name = node('th', result.model_name); name.scope = 'row'; tr.append(name);
    if (!index) tr.className = 'baseline';
    for (const [key, , digits] of metrics) {
      const score = result.metrics[key], baseline = first.input_metrics[key];
      tr.insertCell().textContent = format(score, digits);
      const delta = score - baseline;
      tr.insertCell().textContent = index && Number.isFinite(score) && Number.isFinite(baseline)
        ? (delta > 0 ? '+' : '') + delta.toFixed(digits) : '—';
    }
    tr.insertCell().textContent = index ? format(result.seconds) : '—';
    tr.insertCell().textContent = index ? result.device.toUpperCase() : '—';
  }
  host.append(table);
  host.append(node('p', paired ? 'Δ 为该方法减去带噪输入的整图分数，正值表示改善。'
    : 'Δ 为该方法减去带噪输入的 BRISQUE 分数，负值表示该无参考指标改善。', 'muted'));
  if (!paired) {
    for (const row of rows) for (const note of row.metrics.quality_notes || []) {
      host.append(node('p', row.model_name + ' · ' + note, 'muted'));
    }
  }
}
