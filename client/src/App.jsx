import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import ImagePreview from './components/ImagePreview';
import StatsDisplay from './components/StatsDisplay';
import './App.css';

const App = () => {
  const [file, setFile]               = useState(null);
  const [originalUrl, setOriginalUrl] = useState(null);
  const [params, setParams]           = useState({
    brightness: 0,
    contrast: 1.0,
    gamma: 1.0,
    noiseReduction: 1,
    artifactSize: 30,
    redBoost: 1.5,
    greenSuppress: 1.0,
    advanced: false,
    morphKernel: 3
  });
  const [previewUrl, setPreviewUrl] = useState(null);
  const [stats, setStats] = useState(null);

  const handleAnalyze = async () => {
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    form.append('brightness', params.brightness);
    form.append('contrast', params.contrast);
    form.append('gamma', params.gamma);
    form.append('noise_reduction', params.noiseReduction);
    form.append('artifact_size', params.artifactSize);
    form.append('red_factor', params.redBoost);
    form.append('green_suppression', params.greenSuppress);
    form.append('advanced', params.advanced);
    form.append('morph_kernel', params.morphKernel);

    try {
      const resp = await fetch('/upload', { method: 'POST', body: form });
      const json = await resp.json();
      const { processed_image, ratios, lines_detected } = json.data;
      setPreviewUrl(processed_image);
      setStats({ ratios, lines_detected });
    } catch (err) {
      console.error('Analyze error:', err);
    }
  };

  return (
    <div className="app-container">
      <h1>Fiber Assay Analysis</h1>
      <ImageUploader setFile={setFile} setOriginalUrl={setOriginalUrl} />
      {file && originalUrl && (
        <>
          <ImagePreview
            originalUrl={originalUrl}
            file={file}
            params={params}
            setParams={setParams}
            previewUrl={previewUrl}
            setPreviewUrl={setPreviewUrl}
          />
          <button onClick={handleAnalyze}>Analyze</button>
        </>
      )}
      {stats && <StatsDisplay stats={stats} />}
    </div>
  );
}

export default App;
