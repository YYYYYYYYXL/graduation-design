// Restormer, CVPR 2022, Tables 5 and 6. Report only the run's measured methods and test groups.
function benchmarkTables(report) {
  const fragment = document.createDocumentFragment();
  const gaussian = report.datasets.filter(id => id.startsWith('div2k_'));
  const real = report.datasets.filter(id => !gaussian.includes(id));
  const rows = new Map(report.rows.map(row => [row.dataset + '/' + row.model, row]));
  const get = (dataset, model) => rows.get(dataset + '/' + model);
  function datasetLabel(id) {
    const row = get(id, 'input');
    return groupName(id) + (row ? ' · ' + row.compared + '/' + row.total + ' 张' : '');
  }
  function metricCells(dataset, metric, digits) {
    const scores = report.models.map(model => get(dataset, model.id)?.[metric]);
    const values = scores.map(value => value === '∞' ? Infinity :
      Number.isFinite(value) ? Number(value.toFixed(digits)) : null);
    const best = Math.max(...values.filter(value => value !== null));
    return scores.map((score, i) => {
      const cell = element('td'), label = fmt(score, digits);
      if (values[i] === best) cell.append(element('strong', label));
      else cell.textContent = label;
      return cell;
    });
  }
  if (gaussian.length) {
    const {table, body} = buildTable(['方法', ...gaussian.map(datasetLabel)]);
    table.createCaption().textContent = '高斯彩色图像去噪 · PSNR ↑ (dB)';
    const columns = gaussian.map(dataset => metricCells(dataset, 'psnr', 2));
    report.models.forEach((model, index) => {
      const row = body.insertRow(), name = element('th', model.name);
      name.scope = 'row'; row.append(name, ...columns.map(cells => cells[index]));
    });
    fragment.append(table);
  }
  if (real.length) {
    const {table, body} = buildTable(['测试组', '指标', ...report.models.map(m => m.name)]);
    table.createCaption().textContent = '真实图像去噪';
    for (const dataset of real) {
      const psnr = body.insertRow(), group = element('th', datasetLabel(dataset));
      group.rowSpan = 2;
      psnr.append(group, element('th', 'PSNR ↑ (dB)'), ...metricCells(dataset, 'psnr', 2));
      body.insertRow().append(element('th', 'SSIM ↑'), ...metricCells(dataset, 'ssim', 3));
    }
    fragment.append(table);
  }
  return fragment;
}
