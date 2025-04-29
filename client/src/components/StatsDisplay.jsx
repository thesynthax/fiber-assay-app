import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip
} from 'recharts';

const StatsDisplay = ({ stats }) => {
  // Convert ratios into histogram bins
  const binSize = 0.1;
  const binCount = Math.ceil(1 / binSize);
  const bins = Array.from({ length: binCount }, (_, i) => ({
    bin: `${(i * binSize).toFixed(1)}–${((i + 1) * binSize).toFixed(1)}`,
    count: 0
  }));

  stats.ratios.forEach((r) => {
    const idx = Math.min(Math.floor(r / binSize), binCount - 1);
    bins[idx].count++;
  });

  return (
    <div className="stats">
      <h3>Analysis Results</h3>
      <p>Lines Detected: {stats.lines_detected}</p>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={bins} margin={{ top: 5, right: 20, bottom: 25, left: 0 }}>
          <CartesianGrid stroke="#ccc" />
          <XAxis
            dataKey="bin"
            label={{ value: 'Red / (Red + Green)', position: 'insideBottom', offset: -10 }}
          />
          <YAxis label={{ value: 'Fiber Count', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Bar dataKey="count" fill="#82ca9d" />
        </BarChart>
      </ResponsiveContainer>

      <details>
        <summary>Individual Ratios</summary>
        <ul>
          {stats.ratios.map((r, i) => (
            <li key={i}>{r.toFixed(2)}</li>
          ))}
        </ul>
      </details>
    </div>
  );
};

export default StatsDisplay;
