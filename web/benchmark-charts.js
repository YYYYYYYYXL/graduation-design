// Read the existing per-image CSV once for the selected, finished run.
function parseBenchmarkCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  text = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (!quoted && c === ',') { row.push(field); field = ''; }
    else if (!quoted && (c === '\n' || c === '\r')) {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); rows.push(row); row = []; field = '';
    } else field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const header = rows.shift() || [];
  return rows.map(values => Object.fromEntries(header.map((key, i) => [key, values[i]])));
}

function pairedDifferences(records, dataset, models, method, comparator, metric) {
  const samples = new Map();
  for (const row of records) {
    if (row.dataset !== dataset || row.error) continue;
    if (!samples.has(row.filename)) samples.set(row.filename, new Map());
    samples.get(row.filename).set(row.model, row);
  }
  const values = [];
  for (const sample of samples.values()) {
    if (!models.every(model => sample.has(model.id))) continue;
    const row = sample.get(method);
    const a = row[metric];
    const b = comparator === 'input' ? row['input_' + metric] : sample.get(comparator)[metric];
    if (a?.trim() && b?.trim() && Number.isFinite(+a) && Number.isFinite(+b)) values.push(+a - +b);
  }
  return values.sort((a, b) => a - b);
}

function quantile(sorted, p) {
  const pos = (sorted.length - 1) * p;
  const lo = Math.floor(pos);
  return sorted[lo] + (pos - lo) * (sorted[Math.ceil(pos)] - sorted[lo]);
}

let chartRun, imageRecords, distributionChart;
const field = key => document.getElementById('batch-chart-' + key);

function comparisonOptions(select, options, preferred) {
  select.replaceChildren(...options.map(({id, name}) => new Option(name, id)));
  select.value = options.some(item => item.id === preferred) ? preferred : options[0].id;
}

async function updateBenchmarkCharts(data) {
  if (chartRun !== data.id) {
    chartRun = data.id;
    imageRecords = null;
    comparisonOptions(field('group'), data.datasets.map(id => ({id, name: groupName(id)})));
    comparisonOptions(field('method'), data.models);
    updateComparators();
  }
  if (data.state === 'running') {
    distributionMessage('评测结束后显示逐图配对分布。当前共同样本均值见上表。');
    return;
  }
  if (imageRecords) { drawDistribution(); return; }
  distributionMessage('正在读取逐图测量值…');
  try {
    const response = await fetch(base + '/' + encodeURIComponent(data.id) + '/images.csv');
    if (!response.ok) throw new Error();
    const records = parseBenchmarkCsv(await response.text());
    if (activeId !== data.id || report !== data) return;
    imageRecords = records;
    drawDistribution();
  } catch {
    if (activeId === data.id) distributionMessage('逐图数据读取失败，请重新选择本次评测或刷新页面。');
  }
}

function updateComparators() {
  const options = [{id: 'input', name: '带噪输入'}, ...report.models.filter(m => m.id !== field('method').value)];
  comparisonOptions(field('base'), options, field('base').value);
}

function distributionMessage(message) {
  if (distributionChart) { distributionChart.dispose(); distributionChart = null; }
  field('content').hidden = true;
  field('context').textContent = message;
  field('stats').replaceChildren();
  field('export').disabled = true;
}

function drawDistribution() {
  if (!imageRecords || !report) return;
  const dataset = field('group').value, metric = field('metric').value;
  const method = field('method').value, comparator = field('base').value;
  const values = pairedDifferences(imageRecords, dataset, report.models, method, comparator, metric);
  const baseline = report.rows.find(row => row.dataset === dataset && row.model === 'input');
  const n = values.length, unit = metric === 'psnr' ? ' dB' : '', digits = metric === 'psnr' ? 2 : 4;
  const pair = field('method').selectedOptions[0].text + ' − ' + field('base').selectedOptions[0].text;
  const context = groupName(dataset) + ' · ' + pair + ' · 有效 ' + n + '/' + baseline.total
    + ' 张（未共同成功 ' + (baseline.total - baseline.compared) + '，差值非有限或缺失 ' + (baseline.compared - n) + '）';
  distributionMessage(context + (n ? '' : ' · 尚无可比较的有限差值。'));
  if (!n) return;
  const higher = values.filter(v => v > 0).length, tied = values.filter(v => v === 0).length;
  const {table, body} = buildTable(['Δ 中位数', '第 25 / 75 百分位', '改善 / 持平 / 下降（张）']);
  const row = body.insertRow();
  [signed(quantile(values, .5), digits) + unit,
    [quantile(values, .25), quantile(values, .75)].map(v => signed(v, digits)).join(' / ') + unit,
    [higher, tied, n - higher - tied].join(' / ')].forEach(v => row.insertCell().textContent = v);
  field('stats').append(table);
  if (n < 2 || !window.echarts) {
    field('context').textContent += n < 2 ? ' · 单例仅报告观测值。' : ' · 图形组件不可用。';
    return;
  }
  const points = [];
  const margin = Math.max((values[n-1] - values[0]) * .05, 10 ** -digits);
  const low = Math.min(0, values[0]) - margin, high = Math.max(0, values[n-1]) + margin;
  points.push([low, 0]);
  for (let i = 0; i < n; i++) {
    if (i + 1 === n || values[i] !== values[i + 1]) points.push([values[i], (i + 1) * 100 / n]);
  }
  points.push([high, 100]);
  const host = field('content');
  host.hidden = false;
  distributionChart = window.echarts.init(host, null, {renderer: 'svg'});
  distributionChart.setOption({
    animation: false, backgroundColor: '#fff',
    title: {text: groupName(dataset) + ' · Δ' + metric.toUpperCase(), subtext: pair + '   n = ' + n,
      textStyle: {fontSize: 15}, left: 12},
    grid: {left: 56, right: 26, top: 90, bottom: 55},
    xAxis: {type: 'value', min: low, max: high, name: '逐图 Δ' + metric.toUpperCase() + unit,
      nameLocation: 'middle', nameGap: 32, axisLabel: {hideOverlap: true, formatter: v => Number(v.toFixed(digits))}},
    yAxis: {type: 'value', min: 0, max: 100, name: '累计样本比例 (%)'},
    series: [{type: 'line', step: 'end', showSymbol: false, data: points, lineStyle: {color: '#16576c', width: 2.5},
      markLine: {silent: true, symbol: 'none', label: {show: false}, data: [{xAxis: 0}]}}]
  });
  field('export').disabled = false;
}

field('method').onchange = () => { updateComparators(); drawDistribution(); };
for (const key of ['base', 'metric', 'group']) field(key).onchange = drawDistribution;
field('export').onclick = () => {
  if (!distributionChart) return;
  const link = document.createElement('a');
  link.href = distributionChart.getDataURL({type: 'svg'});
  link.download = [activeId, field('group').value, field('method').value, field('base').value, field('metric').value].join('_') + '.svg';
  link.click();
};
new ResizeObserver(() => distributionChart?.resize()).observe(field('content'));
