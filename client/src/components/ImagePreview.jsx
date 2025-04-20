import React, { useEffect } from 'react';

const ImagePreview = ({ 
  originalUrl,
  file,
  params,
  setParams,
  previewUrl,
  setPreviewUrl,
  }) => {
  
  const { threshold, blurRadius, cleanLevel, redFactor } = params

  useEffect(() => {
    if (!file) return
    const timer = setTimeout(async () => {
      const form = new FormData()
      form.append('file', file)
      form.append('threshold', threshold)
      form.append('bilateral_d', blurRadius)
      form.append('morph_k', cleanLevel)
      form.append('red_factor', redFactor)

      try {
        const resp = await fetch('/preview', { method: 'POST', body: form })
        const blob = await resp.blob()
        setPreviewUrl(URL.createObjectURL(blob))
      } catch (e) {
        console.error(e)
      }
    }, 300)
    return () => clearTimeout(timer)
  }, [file, threshold, blurRadius, cleanLevel, redFactor, setPreviewUrl])

  return (
    <div className="preview">
      <div className="images">
        <div>
          <h3>Original</h3>
          <img src={originalUrl} alt="Original" />
        </div>
        <div>
          <h3>Preview</h3>
          {previewUrl && <img src={previewUrl} alt="Preview" />}
        </div>
      </div>
      <div className="controls">
        <label>
          Brightness Cutoff: {threshold}
          <input
            type="range"
            min="0"
            max="255"
            step="1"
            value={threshold}
            onChange={e => setParams(p => ({ ...p, threshold: Number(e.target.value) }))}
          />
        </label>
        <label>
          Blur Radius: {blurRadius}
          <input
            type="range"
            min="1"
            max="25"
            step="2"
            value={blurRadius}
            onChange={e => setParams(p => ({ ...p, blurRadius: Number(e.target.value) }))}
          />
        </label>
        <label>
          Clean-up Level: {cleanLevel}
          <input
            type="range"
            min="1"
            max="15"
            step="2"
            value={cleanLevel}
            onChange={e => setParams(p => ({ ...p, cleanLevel: Number(e.target.value) }))}
          />
        </label>
        <label>
          Red Enhancement: {redFactor.toFixed(1)}
          <input
            type="range" min="1" max="3" step="0.1"
            value={redFactor}
            onChange={e => setParams(p => ({ ...p, redFactor: Number(e.target.value) }))}
          />
        </label>
      </div>
    </div>
  )
}

export default ImagePreview;
