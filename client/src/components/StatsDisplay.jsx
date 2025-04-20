import React from 'react';

const StatsDisplay = ({ stats }) => {

  return (
    <div className="stats">
      <h3>Analysis Results</h3>
      <p>Lines Detected: {stats.lines_detected}</p>
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
