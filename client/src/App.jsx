import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import ImagePreview from './components/ImagePreview';
import StatsDisplay from './components/StatsDisplay';
import './App.css';

const App = () => {
  const [file, setFile] = useState(null)
  const [originalUrl, setOriginalUrl] = useState(null)
  const [params, setParams] = useState({ threshold: 45, blurRadius: 9, cleanLevel: 3, redFactor: 1.5 })
  const [previewUrl, setPreviewUrl] = useState(null)
  const [stats, setStats] = useState(null)

  const handleAnalyze = async () => {
    if (!file) return
    const formData = new FormData()
    formData.append('file', file)
    formData.append('threshold', params.threshold)
    formData.append('bilateral_d', params.blurRadius)
    formData.append('morph_k', params.cleanLevel)
    formData.append('red_factor', params.redFactor);

    try {
      const resp = await fetch('/upload', { method: 'POST', body: formData })
      const json = await resp.json()
      const { processed_image, ratios, lines_detected } = json.data
      setPreviewUrl(processed_image)
      setStats({ ratios, lines_detected })
    } catch (e) {
      console.error(e)
    }
  }

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
  )
}

export default App;
