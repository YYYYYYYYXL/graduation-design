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

function drawCosts() {
  const rows = report.rows.filter(row => row.dataset === byId('cost-group').value && row.model !== 'input');
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

const epochMetrics = [
  {label: '验证 PSNR (dB)', keys: ['val_psnr', 'val_psnr_mean']},
  {label: '验证 SSIM', keys: ['val_ssim']},
  {label: '训练 MSE', keys: ['train_mse']},
  {label: '训练 L1', keys: ['train_l1']},
  {label: '每轮耗时 (s)', keys: ['seconds']},
  {label: '学习率', keys: ['lr']}
];

function logNumber(value) {
  if (value == null || String(value).trim() === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

async function loadEpochCharts(models) {
  const host = byId('epoch-chart'), picker = byId('epoch-metric'), note = byId('epoch-note');
  const histories = await Promise.all(models.map(async model => {
    if (model.method === 'bm3d') return {model, note: '无训练轮次'};
    try {
      const log = await json('/api/history/' + encodeURIComponent(model.id));
      const rows = log.rows.map(row => ({...row, epoch: logNumber(row.epoch)}))
        .filter(row => Number.isInteger(row.epoch) && row.epoch >= 0)
        .sort((a, b) => a.epoch - b.epoch);
      return rows.length ? {model, rows} : {model, note: '无逐轮数据'};
    } catch {
      return {model, note: '日志读取失败'};
    }
  }));
  const metrics = epochMetrics.filter(metric => histories.some(h =>
    h.rows?.some(row => metric.keys.some(key => logNumber(row[key]) !== null))));
  const unavailable = histories.filter(h => h.note).map(h => h.model.name + '：' + h.note);
  if (!metrics.length || !window.echarts) {
    host.hidden = true;
    note.textContent = [metrics.length ? '图表组件不可用' : '暂无训练曲线', ...unavailable].join('。');
    return;
  }
  const chart = echarts.init(host, null, {renderer: 'svg'});
  picker.replaceChildren(...metrics.map(metric => new Option(metric.label, metric.label)));
  picker.disabled = false;
  function draw() {
    const metric = metrics[picker.selectedIndex];
    const notes = [...unavailable], series = [];
    const colors = ['#195c80', '#ad4d17', '#007754', '#834582', '#676619', '#4c5369'];
    histories.forEach(({model, rows}, index) => {
      if (!rows) return;
      const data = rows.map(row => [row.epoch, metric.keys.map(key => logNumber(row[key]))
        .find(value => value !== null) ?? null]);
      if (!data.some(point => point[1] !== null)) {
        notes.push(model.name + '：未记录该指标');
        return;
      }
      series.push({
        name: model.name, type: 'line', data,
        showSymbol: data.length === 1, itemStyle: {color: colors[index % colors.length]},
        lineStyle: {width: 2.5}
      });
    });
    chart.setOption({
      animation: false, backgroundColor: '#fff',
      legend: {type: 'scroll', left: 8, right: 8, bottom: 4},
      grid: {left: 16, right: 25, top: 48, bottom: 90, containLabel: true},
      tooltip: {trigger: 'axis', renderMode: 'richText', confine: true},
      xAxis: {type: 'value', name: 'Epoch', nameLocation: 'middle', nameGap: 32, minInterval: 1},
      yAxis: {type: 'value', name: metric.label, scale: true, nameTextStyle: {align: 'left'}},
      toolbox: {right: 8, feature: {saveAsImage: {type: 'svg', title: '导出曲线图'}}},
      series
    }, true);
    note.textContent = ['各曲线使用对应模型的训练日志，验证集随模型而异', ...notes].join('。') + '。';
  }
  picker.onchange = draw;
  draw();
  new ResizeObserver(() => chart.resize()).observe(host);
}
