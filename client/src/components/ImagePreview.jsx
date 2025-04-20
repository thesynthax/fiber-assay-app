import React, { useEffect } from 'react';

const ImagePreview = ({ 
  originalUrl,
  file,
  params,
  setParams,
  previewUrl,
  setPreviewUrl,
  }) => {
  
  const { brightness, contrast, gamma, noiseReduction, artifactSize, redBoost, greenSuppress, advanced, morphKernel } = params

  useEffect(() => {
    if (!file) return
    const timeout = setTimeout(async () => {
      const form = new FormData();
      form.append('file', file);
      form.append('brightness', brightness);
      form.append('contrast', contrast);
      form.append('gamma', gamma);
      form.append('noise_reduction', noiseReduction);
      form.append('artifact_size', artifactSize);
      form.append('red_factor', redBoost);
      form.append('green_suppression', greenSuppress);
      form.append('advanced', advanced);
      form.append('morph_kernel', morphKernel);

      try {
        const resp = await fetch('/preview', { method: 'POST', body: form });
        const blob = await resp.blob();
        setPreviewUrl(URL.createObjectURL(blob));
      } catch (err) {
        console.error('Preview error:', err);
      }
    }, 300);
    return () => clearTimeout(timeout);
  }, [file, brightness, contrast, gamma, noiseReduction, artifactSize, redBoost, greenSuppress, advanced, morphKernel]);

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
          Brightness: {brightness}
          <input type="range" min="0" max="100" step="1" value={brightness} onChange={e => setParams(p => ({ ...p, brightness: Number(e.target.value) }))} />
        </label>
        
        <label>
          Noise Reduction: {noiseReduction}
          <input type="range" min="1" max="10" step="1" value={noiseReduction} onChange={e => setParams(p => ({ ...p, noiseReduction: Number(e.target.value) }))} />
        </label>
        <label>
          Minimum Fiber Size: {artifactSize}px
          <input type="range" min="1" max="100" step="1" value={artifactSize} onChange={e => setParams(p => ({ ...p, artifactSize: Number(e.target.value) }))} />
        </label>
        <label>
          Red Boost: {redBoost.toFixed(1)}×
          <input type="range" min="1.0" max="3.0" step="0.1" value={redBoost} onChange={e => setParams(p => ({ ...p, redBoost: Number(e.target.value) }))} />
        </label>
        <label>
          Green Suppression: {greenSuppress.toFixed(1)}×
          <input type="range" min="0.2" max="1.0" step="0.1" value={greenSuppress} onChange={e => setParams(p => ({ ...p, greenSuppress: Number(e.target.value) }))} />
        </label>
        <label>
          <input type="checkbox" checked={advanced} onChange={e => setParams(p => ({ ...p, advanced: e.target.checked }))} /> Advanced Morphology
        </label>
        {advanced && (
          <>
            <label>
              Morph Kernel: {morphKernel}
              <input type="range" min="1" max="15" step="2" value={morphKernel} onChange={e => setParams(p => ({ ...p, morphKernel: Number(e.target.value) }))} />
            </label>
            <label>
              Contrast: {contrast.toFixed(1)}×
              <input type="range" min="0.5" max="3.0" step="0.1" value={contrast} onChange={e => setParams(p => ({ ...p, contrast: Number(e.target.value) }))} />
            </label>
            <label>
              Gamma: {gamma.toFixed(1)}×
              <input type="range" min="0.2" max="5.0" step="0.1" value={gamma} onChange={e => setParams(p => ({ ...p, gamma: Number(e.target.value) }))} />
            </label>
          </>
        )}
      </div>
    </div>
  );
}

export default ImagePreview;
