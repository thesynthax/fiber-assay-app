import React from 'react';

const ImageUploader = ({ setFile, setOriginalUrl }) => {
  
  const handleChange = async e => {
    const f = e.target.files[0]
    if (!f) return
    setFile(f)
    // immediately convert TIFF → PNG
    const form = new FormData()
    form.append('file', f)
    try {
      const resp = await fetch('/convert', { method: 'POST', body: form })
      const blob = await resp.blob()
      setOriginalUrl(URL.createObjectURL(blob))
    } catch (err) {
      console.error('Convert error:', err)
    }
  }
  return (
    <div className="uploader">
      <input type="file" accept="image/tiff,image/*" onChange={handleChange} />
    </div>
  )
}

export default ImageUploader;
