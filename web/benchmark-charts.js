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

const performanceCharts = new Map();
const symbols = ['circle', 'rect', 'triangle', 'diamond', 'roundRect', 'pin'];
let plotReport;

function performanceFigure(id, title, xAxis, series, downloadName) {
  const host = byId(id);
  let chart = performanceCharts.get(id);
  const measured = series.filter(line => line.data.some(point => Number.isFinite(point[1])));
  if (!measured.length || !window.echarts) {
    chart?.dispose(); performanceCharts.delete(id);
    host.textContent = measured.length ? '图形组件不可用' : '暂无有限测量值';
    return;
  }
  if (!chart) {
    host.replaceChildren();
    chart = echarts.init(host, null, {renderer: 'svg'});
    performanceCharts.set(id, chart);
  }
  const narrow = host.clientWidth < 560;
  chart.setOption({
    animation: false, backgroundColor: '#fff',
    color: ['#195c80', '#ad4d17', '#007754', '#834582', '#555d17', '#4c5369'],
    title: {text: title, left: 6, textStyle: {fontSize: narrow ? 13 : 16}},
    grid: {left: 65, right: 28, top: 78, bottom: 92},
    legend: {type: 'scroll', bottom: 8, left: 12, right: 12},
    tooltip: {trigger: 'item', renderMode: 'richText', confine: true,
      formatter: p => p.seriesName + '\n' + xAxis.name + '  ' + p.value[0] + '\n平均 PSNR  ' + p.value[1].toFixed(2) + ' dB'},
    toolbox: {right: 4, feature: {saveAsImage: {type: 'svg', name: downloadName, title: '导出 SVG'}}},
    xAxis: {type: 'value', nameLocation: 'middle', nameGap: 32, ...xAxis},
    yAxis: {type: 'value', name: '平均 PSNR (dB)', scale: true},
    series: measured
  }, true);
  chart.resize();
}

// Noise2Void (CVPR 2019), Fig. 7 left: average PSNR versus added Gaussian noise.
function noiseFigure(data) {
  const levels = data.datasets.filter(id => /^div2k_\d+$/.test(id))
    .map(id => ({id, sigma: Number(id.split('_')[1])})).sort((a, b) => a.sigma - b.sigma);
  byId('noise-panel').hidden = !levels.length;
  if (!levels.length) return;
  const rows = new Map(data.rows.map(row => [row.dataset + '/' + row.model, row]));
  const series = data.models.map((model, i) => ({
    name: model.name, type: 'line', symbol: symbols[i % symbols.length], symbolSize: 8,
    showAllSymbol: true, connectNulls: false,
    data: levels.map(level => {
      const row = rows.get(level.id + '/' + model.id);
      return [level.sigma, row?.compared > 0 && Number.isFinite(row.psnr) ? row.psnr : null];
    })
  }));
  performanceFigure('noise-curve', (data.preview ? '预览 · ' : '') + 'DIV2K · 高斯噪声',
    {name: '噪声标准差 σ', min: 0, max: levels.at(-1).sigma + 5, interval: 5}, series, data.id + '-noise-psnr');
}

// IDEA-Net (WACV Workshops 2022), Fig. 5: PSNR versus denoising time.
// Our stored time is per image; the original paper times a full image set.
function costFigure() {
  if (!plotReport) return;
  const dataset = byId('cost-group').value;
  const rows = plotReport.rows.filter(row => row.dataset === dataset && row.model !== 'input');
  const series = rows.filter(row => row.compared > 0 && Number.isFinite(row.seconds) && Number.isFinite(row.psnr))
    .map(row => ({
      name: row.name + ' · ' + (row.device || '设备未知'), type: 'scatter',
      symbol: symbols[plotReport.models.findIndex(m => m.id === row.model) % symbols.length],
      symbolSize: 13, data: [[row.seconds, row.psnr]]
    }));
  performanceFigure('cost-plot', (plotReport.preview ? '预览 · ' : '') + groupName(dataset) + ' · n=' + (rows[0]?.compared || 0),
    {name: '平均去噪耗时 (s/张)', min: 0}, series, plotReport.id + '-' + dataset + '-time-psnr');
  const {table, body} = buildTable(['模型', '参数量 (M)', '权重文件 (MiB)', '设备', '平均耗时 (s/张)', '计分 / 总样本', '失败数']);
  for (const row of rows) {
    const tr = body.insertRow();
    for (const value of [row.name, Number.isFinite(row.parameters) ? fmt(row.parameters / 1e6, 3) : '—',
      Number.isFinite(row.weights_bytes) ? fmt(row.weights_bytes / 1048576) : '—', row.device || '—', fmt(row.seconds, 3), row.compared + ' / ' + row.total, row.failed]) {
      tr.insertCell().textContent = value;
    }
  }
  byId('model-costs').replaceChildren(table);
}

function drawPerformancePlots(data) {
  const select = byId('cost-group');
  if (plotReport?.id !== data.id) select.replaceChildren(...data.datasets.map(id => new Option(groupName(id), id)));
  plotReport = data;
  byId('model-plots').hidden = false;
  noiseFigure(data); costFigure();
}
