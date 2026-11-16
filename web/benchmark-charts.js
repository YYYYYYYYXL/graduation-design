const chartMetrics = {
  psnr: {label: 'PSNR', digits: 2, unit: ' dB', delta: 'delta_psnr', padding: 0.4},
  ssim: {label: 'SSIM', digits: 4, unit: '', delta: 'delta_ssim', padding: 0.004},
};
const noiseLevels = [
  {id: 'div2k_15', sigma: 15},
  {id: 'div2k_25', sigma: 25},
  {id: 'div2k_50', sigma: 50},
];
const preferredTrendModels = ['restormer_div2k_batch16', 'p2n_div2k'];
const chartColors = {
  restormer_div2k_batch16: '#146771',
  p2n_div2k: '#436897',
  dncnn_div2k: '#7a5598',
  dncnn_sidd: '#a86725',
  restormer_sidd_batch16: '#a64953',
  bm3d: '#6d7844',
  input: '#687782',
};
const chartHost = document.getElementById('batch-chart-content');
let benchmarkChart, chartReport, trendSelection;
let chartView = 'ranking';
let chartGroupsKey = '';

function updateBenchmarkCharts(data) {
  if (chartReport?.id !== data.id) trendSelection = undefined;
  chartReport = data;
  const picker = byId('batch-chart-group');
  const key = data.datasets.join(',');
  if (key !== chartGroupsKey) {
    const previous = picker.value;
    picker.replaceChildren(...data.datasets.map(id => new Option(groupName(id), id)));
    picker.value = data.datasets.includes(previous) ? previous : data.datasets[0];
    chartGroupsKey = key;
  }
  const canShowTrend = noiseLevels.filter(level => data.datasets.includes(level.id)).length > 1;
  if (!canShowTrend) chartView = 'ranking';
  document.querySelectorAll('[data-chart-view]').forEach(button => {
    const target = button.dataset.chartView;
    button.disabled = target === 'trend' && !canShowTrend;
    button.setAttribute('aria-pressed', String(chartView === target));
  });
  byId('batch-chart-group-field').hidden = chartView === 'trend';
  drawBenchmarkChart();
}

function drawBenchmarkChart() {
  if (!chartReport) return;
  const highlights = byId('batch-chart-highlights');
  const message = byId('batch-chart-message');
  highlights.replaceChildren();
  highlights.hidden = message.hidden = chartHost.hidden = true;
  const metricId = byId('batch-chart-metric').value;
  const metric = chartMetrics[metricId];
  if (!window.echarts) {
    showChartMessage('图表库未能加载。可以查看下方的汇总表。');
    return;
  }
  const view = chartView === 'ranking'
    ? rankingChart(metricId, metric, highlights)
    : trendChart(metricId, metric);
  if (!view) return;

  chartHost.hidden = false;
  if (!benchmarkChart) {
    benchmarkChart = window.echarts.init(chartHost, null, {renderer: 'svg'});
    benchmarkChart.on('legendselectchanged', event => { trendSelection = event.selected; });
  } else if (benchmarkChart.getWidth() !== chartHost.clientWidth) {
    benchmarkChart.resize();
  }
  const narrow = chartHost.clientWidth < 520;
  benchmarkChart.setOption(view(narrow), {notMerge: true});
}

function showChartMessage(text) {
  const message = byId('batch-chart-message');
  message.textContent = text;
  message.hidden = false;
}

function scoreText(value, metric) {
  if (!Number.isFinite(value)) return value === '∞' ? '∞' : '—';
  return `${fmt(value, metric.digits)}${metric.unit}`;
}

function chartHeading(metric) {
  return `平均 ${metric.label}${metric.unit ? '（dB）' : ''}`;
}

function chartHighlight(label, value, leading = false) {
  const card = element('div', undefined, leading ? 'batch-chart-stat batch-chart-stat-leading' : 'batch-chart-stat');
  card.append(element('span', label), element('strong', value));
  return card;
}

function chartOptions(description) {
  return {
    animation: false,
    aria: {enabled: true, label: {description}},
    textStyle: {fontFamily: 'Microsoft YaHei, system-ui, sans-serif', color: '#283d45'},
    tooltip: {confine: true, renderMode: 'richText', backgroundColor: '#20383e',
      borderWidth: 0, textStyle: {color: '#fff', fontSize: 12}},
  };
}

function rankingChart(metricId, metric, highlights) {
  const dataset = byId('batch-chart-group').value;
  const rows = chartReport.rows.filter(row => row.dataset === dataset);
  const baseline = rows.find(row => row.model === 'input');
  if (!baseline) return null;
  const preview = chartReport.preview ? '预览 · ' : '';
  byId('batch-chart-context').textContent = `${preview}${groupName(dataset)} · 共同计分 ${baseline.compared}/${baseline.total} 张`;
  if (!baseline.compared) {
    showChartMessage('等待所选方法完成同一批样本，随后显示排名。');
    return null;
  }

  const score = row => row[metricId] === '∞' ? Infinity : row[metricId];
  const ranked = rows.filter(row => row.model !== 'input' &&
    (Number.isFinite(score(row)) || score(row) === Infinity));
  ranked.sort((a, b) => score(a) === score(b) ? 0 : score(a) > score(b) ? -1 : 1);
  if (!ranked.length) {
    showChartMessage('当前没有可用的模型分数。');
    return null;
  }
  const [first, second] = ranked;
  const leadingScore = scoreText(first[metricId], metric);
  highlights.append(chartHighlight(`第 1 名 · ${first.name}`, leadingScore, true));
  if (second && Number.isFinite(score(first)) && Number.isFinite(score(second))) {
    const gap = score(first) - score(second);
    highlights.append(chartHighlight(gap ? `领先 ${second.name}` : `与 ${second.name} 并列`, scoreText(gap, metric)));
  }
  highlights.append(chartHighlight('去噪前', scoreText(baseline[metricId], metric)));
  highlights.hidden = false;

  const finite = ranked.filter(row => Number.isFinite(score(row)));
  if (!finite.length) {
    showChartMessage('当前模型分数均为 ∞，完整读数见汇总表。');
    return null;
  }
  const names = new Map(ranked.map(row => [row.model,
    row[metricId] === '∞' ? `${row.name}  ∞` : row.name]));
  const low = score(finite.at(-1)), high = score(finite[0]);
  const padding = Math.max(metric.padding, (high - low) * 0.085);

  return narrow => ({
    ...chartOptions(`${groupName(dataset)}的${chartHeading(metric)}排名。${first.name}排名第一，得分 ${leadingScore}。完整分数见下方汇总表。`),
    grid: {left: narrow ? 112 : 170, right: narrow ? 53 : 82, top: 11, bottom: 57},
    xAxis: {
      type: 'value', min: low - padding, max: high + padding, splitNumber: narrow ? 3 : 5,
      name: chartHeading(metric), nameLocation: 'middle', nameGap: 34,
      axisLabel: {color: '#526971', formatter: value => Number(value.toFixed(metricId === 'psnr' ? 1 : 3))},
      splitLine: {lineStyle: {color: '#e4edee', type: 'dashed'}},
    },
    yAxis: {
      type: 'category', inverse: true, data: ranked.map(row => row.model),
      axisLine: {show: false}, axisTick: {show: false},
      axisLabel: {formatter: id => names.get(id), width: narrow ? 100 : 150, overflow: 'truncate',
        color: '#304a53', fontSize: narrow ? 10 : 13},
      splitArea: {show: true, areaStyle: {color: ['#f5f9f9', '#ffffff']}},
    },
    tooltip: {
      ...chartOptions('').tooltip,
      trigger: 'item',
      formatter: ({data}) => {
        const row = data.row;
        const detail = [row.name, `${chartHeading(metric)} ${scoreText(row[metricId], metric)}`];
        if (Number.isFinite(row[metric.delta])) detail.push(`较去噪前 ${signed(row[metric.delta], metric.digits)}${metric.unit}`);
        if (Number.isFinite(row.seconds)) detail.push(`${fmt(row.seconds, 3)} 秒/张 · ${row.device.toUpperCase()}`);
        return detail.join('\n');
      },
    },
    series: [{
      type: 'scatter',
      symbolSize: 12,
      label: {show: true, position: 'right', distance: 7, color: '#233f46', fontWeight: 600,
        fontSize: narrow ? 10 : 12, formatter: ({data}) => scoreText(data.value[0], metric)},
      emphasis: {scale: 1.25},
      data: finite.map(row => ({
        value: [score(row), row.model], row,
        symbolSize: row === first ? 16 : 12,
        itemStyle: {color: row === first ? '#146771' : row === second ? '#436897' : '#839ba1', opacity: 1},
      })),
    }],
  });
}

function trendChart(metricId, metric) {
  const levels = noiseLevels.filter(level => chartReport.datasets.includes(level.id));
  const rows = new Map(chartReport.rows.map(row => [`${row.dataset}/${row.model}`, row]));
  const valueAt = (level, model) => rows.get(`${level.id}/${model}`)?.[metricId];
  const samples = levels.map(level => {
    const baseline = rows.get(`${level.id}/input`);
    return `σ=${level.sigma} ${baseline.compared}/${baseline.total} 张`;
  });
  byId('batch-chart-context').textContent = `${chartReport.preview ? '预览 · ' : ''}共同计分 · ${samples.join(' · ')}`;
  const models = [...chartReport.models, {id: 'input', name: '去噪前'}];
  const plottedLevels = levels.filter(level => models.some(model => Number.isFinite(valueAt(level, model.id))));
  if (plottedLevels.length < 2) {
    showChartMessage('至少两个噪声档位有可用分数时显示趋势。');
    return null;
  }
  const plotted = models.filter(model => levels.some(level => Number.isFinite(valueAt(level, model.id))));
  const preferred = plotted.filter(model => preferredTrendModels.includes(model.id));
  const otherModels = plotted.filter(model => !preferredTrendModels.includes(model.id));
  if (!trendSelection) {
    const defaults = preferred.length ? preferred : plotted.filter(model => model.id !== 'input').slice(0, 2);
    trendSelection = Object.fromEntries(plotted.map(model => [model.name, defaults.includes(model)]));
  }
  const hasInfinite = models.some(model => levels.some(level => valueAt(level, model.id) === '∞'));
  if (hasInfinite) byId('batch-chart-context').textContent += ' · ∞ 分数见汇总表';
  const baseOptions = chartOptions(`DIV2K 各噪声档位的${chartHeading(metric)}。点击图例切换模型，完整分数见下方汇总表。`);
  return narrow => ({
    ...baseOptions,
    color: plotted.map(model => chartColors[model.id] || '#677e8e'),
    grid: {left: narrow ? 45 : 60, right: narrow ? 22 : 36, top: 35, bottom: 91},
    legend: {type: 'scroll', bottom: 15, left: 12, right: 12, itemWidth: 19, itemHeight: 9,
      data: [...preferred, ...otherModels].map(model => model.name),
      selected: trendSelection, textStyle: {fontSize: 12, color: '#38545c'}},
    xAxis: {
      type: 'value', min: levels[0].sigma, max: levels.at(-1).sigma, interval: 5,
      name: '噪声强度 σ', nameLocation: 'middle', nameGap: 31, nameTextStyle: {color: '#526971'},
      axisLabel: {color: '#526971', formatter: value => levels.some(level => level.sigma === value) ? String(value) : ''},
      axisTick: {show: false}, splitLine: {show: false},
    },
    yAxis: {
      type: 'value', scale: true, name: chartHeading(metric), splitNumber: 4, nameTextStyle: {color: '#526971'},
      axisLabel: {color: '#526971'}, splitLine: {lineStyle: {color: '#e4edee', type: 'dashed'}},
    },
    tooltip: {
      ...baseOptions.tooltip, trigger: 'axis', axisPointer: {type: 'line', lineStyle: {color: '#8da6aa'}},
      formatter: points => {
        if (!points.length) return '';
        const lines = points.filter(point => Number.isFinite(point.value[1]))
          .map(point => `${point.seriesName}  ${scoreText(point.value[1], metric)}`);
        return [`σ=${points[0].axisValue}`, ...lines].join('\n');
      },
    },
    series: plotted.map(model => ({
      id: model.id, name: model.name, type: 'line', showSymbol: true, connectNulls: false,
      symbol: model.id === 'p2n_div2k' ? 'diamond' : 'circle', symbolSize: 8,
      lineStyle: {width: 2.5, type: model.id === 'input' ? 'dashed' : 'solid'},
      emphasis: {focus: 'series'},
      data: levels.map(level => {
        const score = valueAt(level, model.id);
        return [level.sigma, Number.isFinite(score) ? score : null];
      }),
    })),
  });
}

document.querySelectorAll('[data-chart-view]').forEach(button => {
  button.onclick = () => {
    chartView = button.dataset.chartView;
    updateBenchmarkCharts(chartReport);
  };
});
document.getElementById('batch-chart-group').onchange = drawBenchmarkChart;
document.getElementById('batch-chart-metric').onchange = drawBenchmarkChart;
new ResizeObserver(() => {
  if (benchmarkChart && !chartHost.hidden && chartHost.clientWidth &&
      chartHost.clientWidth !== benchmarkChart.getWidth()) drawBenchmarkChart();
}).observe(chartHost);
