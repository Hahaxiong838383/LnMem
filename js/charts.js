const chartStore = new Map();

export function renderHexChart(canvas, hex) {
  if (!canvas || !window.Chart) return;
  replaceChart(canvas, {
    type: 'radar',
    data: {
      labels: Object.keys(hex),
      datasets: [{
        label: 'HEXACO',
        data: Object.values(hex),
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56, 189, 248, 0.2)'
      }]
    },
    options: chartOptions(0, 1)
  });
}

export function renderValuesChart(canvas, values) {
  if (!canvas || !window.Chart) return;
  const rows = Array.isArray(values) ? values : [];
  replaceChart(canvas, {
    type: 'bar',
    data: {
      labels: rows.map((item) => valueLabel(item.value)),
      datasets: [{
        label: '价值观排序',
        data: rows.map((item) => item.score),
        backgroundColor: rows.map((_, index) => valueColor(index))
      }]
    },
    options: chartOptions(0, 1)
  });
}

function replaceChart(canvas, config) {
  const old = chartStore.get(canvas.id);
  if (old) old.destroy();
  chartStore.set(canvas.id, new window.Chart(canvas, config));
}

function chartOptions(min, max) {
  return {
    responsive: true,
    scales: { r: { min, max }, y: { min, max } },
    plugins: { legend: { labels: { color: '#e2e8f0' } } }
  };
}

function valueColor(index) {
  const colors = ['#f97316', '#14b8a6', '#38bdf8', '#a3e635', '#facc15'];
  return colors[index % colors.length];
}

function valueLabel(value) {
  const labels = {
    self_direction: '自主',
    stimulation: '刺激',
    hedonism: '享乐',
    achievement: '成就',
    power: '权力',
    security: '安全',
    conformity: '遵从',
    tradition: '传统',
    benevolence: '仁善',
    universalism: '普世'
  };
  return labels[value] || value;
}
