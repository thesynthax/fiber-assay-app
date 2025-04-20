import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const StatsDisplay = ({ stats }) => {
  const data = stats.ratios.map((r, i) => ({ index: i + 1, ratio: Number(r.toFixed(2)) }));

  return (
    <div className="stats">
      <h3>Analysis Results</h3>
      <p>Lines Detected: {stats.lines_detected}</p>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid stroke="#ccc" />
          <XAxis dataKey="index" label={{ value: 'Fiber #', position: 'insideBottomRight', offset: -5 }} />
          <YAxis label={{ value: 'Red/(Red+Green)', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Line type="monotone" dataKey="ratio" stroke="#8884d8" dot={false} />
        </LineChart>
      </ResponsiveContainer>
      <details>
        <summary>Ratios</summary>
        <ul>
          {stats.ratios.map((r, i) => (
            <li key={i}>{r.toFixed(2)}</li>
          ))}
        </ul>
      </details>
    </div>
  )
}

export default StatsDisplay;
