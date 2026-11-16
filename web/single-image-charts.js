const singleImageMetrics = {
  psnr: {title: 'PSNR（dB）', hint: '与干净参考图比较 · 越高越好', digits: 2, unit: ' dB', min: 0, max: Infinity, padding: 0.4},
  ssim: {title: 'SSIM', hint: '与干净参考图比较 · 越接近 1 越好', digits: 4, unit: '', min: -1, max: 1, padding: 0.004},
  brisque: {title: 'BRISQUE', hint: '无参考质量评分 · 通常越低越好', digits: 2, unit: '', min: 0, max: 100, padding: 0.5},
};
const singleImagePlots = [];

function renderSingleImageCharts(results) {
  for (const {chart, observer} of singleImagePlots.splice(0)) {
    observer.disconnect();
    chart.dispose();
  }
  const section = $('single-score-charts');
  const panels = $('single-score-panels');
  panels.replaceChildren();
  section.hidden = !results.length;
  if (!results.length) return;

  const first = results[0];
  const entries = [
    {name: '去噪前', metrics: first.input_metrics, baseline: true},
    ...results.map(result => ({name: result.model_name, metrics: result.metrics, baseline: false})),
  ];
  const keys = first.reference_provided ? ['psnr', 'ssim'] : ['brisque'];
  const plots = [];
  for (const key of keys) {
    const metric = singleImageMetrics[key];
    const card = node('section', undefined, 'single-score-card');
    const title = node('h4', metric.title);
    title.id = `single-score-${key}`;
    card.setAttribute('aria-labelledby', title.id);
    card.append(title, node('p', metric.hint, 'muted'));

    const values = entries.map(entry => ({...entry, value: entry.metrics[key]}));
    const measured = values.filter(entry => Number.isFinite(entry.value));
    if (measured.length && window.echarts) {
      const plot = node('div', undefined, 'single-score-plot');
      plot.style.height = `${Math.max(174, 58 + measured.length * 34)}px`;
      card.append(plot);
      plots.push({plot, metric, measured});
    }
    const details = values.filter(entry => !window.echarts || !Number.isFinite(entry.value));
    if (details.length) {
      const list = node('ul', undefined, 'single-score-details');
      for (const entry of details) {
        const value = entry.value === '∞' ? '∞（与参考图完全一致）' : singleImageScore(entry.value, metric);
        list.append(node('li', `${entry.name} · ${value}`));
      }
      card.append(list);
      const notes = new Set(details.flatMap(entry => entry.metrics.quality_notes ?? []));
      for (const note of notes) card.append(node('p', note, 'muted'));
    }
    panels.append(card);
  }
  for (const {plot, metric, measured} of plots) {
    const chart = window.echarts.init(plot, null, {renderer: 'svg'});
    const draw = () => chart.setOption(singleImageChartOptions(metric, measured, plot.clientWidth < 365));
    draw();
    const observer = new ResizeObserver(() => {
      if (!plot.clientWidth || plot.clientWidth === chart.getWidth()) return;
      chart.resize();
      draw();
    });
    observer.observe(plot);
    singleImagePlots.push({chart, observer});
  }
}

function singleImageScore(value, metric) {
  return Number.isFinite(value) ? `${format(value, metric.digits)}${metric.unit}` : '未测量';
}

function singleImageChartOptions(metric, measured, narrow) {
  const values = measured.map(entry => entry.value);
  const low = Math.min(...values), high = Math.max(...values);
  const padding = Math.max(metric.padding, (high - low) * 0.085);
  const names = measured.map(entry => entry.name);
  return {
    animation: false,
    aria: {enabled: true, label: {description: [metric.title, ...measured.map(
      entry => `${entry.name} ${singleImageScore(entry.value, metric)}`
    )].join('。')}},
    textStyle: {fontFamily: 'Microsoft YaHei, system-ui, sans-serif', color: '#283d45'},
    grid: {left: narrow ? 96 : 150, right: narrow ? 63 : 78, top: 10, bottom: 32},
    xAxis: {
      type: 'value',
      min: Math.min(low, Math.max(metric.min, low - padding)),
      max: Math.max(high, Math.min(metric.max, high + padding)),
      splitNumber: narrow ? 3 : 5,
      axisLabel: {color: '#526971', formatter: value => Number(value.toFixed(metric.digits - 1))},
      splitLine: {lineStyle: {color: '#e4edee', type: 'dashed'}},
    },
    yAxis: {
      type: 'category', inverse: true, data: names.map((_, index) => String(index)),
      axisLine: {show: false}, axisTick: {show: false},
      axisLabel: {formatter: index => names[Number(index)], width: narrow ? 84 : 138,
        overflow: 'truncate', color: '#304a53', fontSize: narrow ? 10 : 12},
      splitArea: {show: true, areaStyle: {color: ['#f5f9f9', '#ffffff']}},
    },
    tooltip: {trigger: 'item', confine: true, renderMode: 'richText',
      backgroundColor: '#20383e', borderWidth: 0, textStyle: {color: '#fff', fontSize: 12},
      formatter: ({data}) => `${names[Number(data.value[1])]}\n${metric.title} ${singleImageScore(data.value[0], metric)}`},
    series: [{
      type: 'scatter', symbolSize: 12,
      label: {show: true, position: 'right', distance: 6, fontSize: narrow ? 10 : 12,
        color: '#233f46', fontWeight: 600, formatter: ({data}) => singleImageScore(data.value[0], metric)},
      data: measured.map((entry, index) => ({
        value: [entry.value, String(index)],
        itemStyle: {color: entry.baseline ? '#82979d' : '#146771', opacity: 1},
      })),
    }],
  };
}
