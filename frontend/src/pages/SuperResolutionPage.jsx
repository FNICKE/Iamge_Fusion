
import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

export default function SuperResolutionPage() {
  const [images, setImages] = useState([])
  const [model,  setModel] = useState('edsr')
  const [loading, setLoading] = useState(false)
  const [result,  setResult] = useState(null)
  const [error,   setError]   = useState(null)

  const onDrop = useCallback(acceptedFiles => {
    if (acceptedFiles.length > 0) {
      setImages(prev => [...prev, ...acceptedFiles])
      setResult(null)
      setError(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.webp'] },
    multiple: true
  })

  const handleProcess = async () => {
    if (images.length === 0) return
    setLoading(true)
    setError(null)

    const formData = new FormData()
    // Always process the FIRST image from the list
    formData.append('image', images[0])
    formData.append('model', model)

    try {
      const resp = await fetch('/api/super-resolve', {
        method: 'POST',
        body: formData,
      })

      const data = await resp.json()
      if (!resp.ok) throw new Error(data.error || 'Failed to process image')
      
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <header className="max-w-3xl mx-auto text-center space-y-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-sm font-medium mb-2">
          <span>✨</span>
          <span>Enhance Image Quality</span>
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white">Super Resolution</h1>
        <p className="text-lg text-slate-400">
          Transform low-resolution images into crystal-clear high-definition versions 
          using state-of-the-art OpenCV Super Resolution models.
        </p>
      </header>

      <section className="grid lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
        {/* INPUT PANEL */}
        <div className="glass-panel p-8 space-y-8 h-fit">
          <div>
            <h3 className="text-xl font-bold text-white mb-6">1. Upload Low-Res Images</h3>
            <div 
              {...getRootProps()} 
              className={`relative min-h-[200px] rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer overflow-hidden flex flex-col items-center justify-center p-6 ${
                isDragActive ? 'border-indigo-500 bg-indigo-500/10' : 'border-white/10 hover:border-white/20 bg-white/5'
              }`}
            >
              <input {...getInputProps()} />
              {images.length > 0 ? (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 w-full">
                  {images.map((img, idx) => (
                    <div key={idx} className="relative aspect-square rounded-xl overflow-hidden border border-white/10 group/img">
                      <img 
                        src={URL.createObjectURL(img)} 
                        alt={`Input ${idx}`} 
                        className="w-full h-full object-cover" 
                      />
                      {idx === 0 && (
                        <div className="absolute top-2 left-2 bg-indigo-600 text-[10px] uppercase font-bold px-2 py-0.5 rounded-full text-white shadow-lg">
                          Processing
                        </div>
                      )}
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/img:opacity-100 transition-opacity flex items-center justify-center">
                        <button 
                          onClick={(e) => {
                            e.stopPropagation()
                            setImages(prev => prev.filter((_, i) => i !== idx))
                          }}
                          className="bg-red-500/80 hover:bg-red-500 p-2 rounded-lg text-white"
                        >
                          🗑️
                        </button>
                      </div>
                    </div>
                  ))}
                  <div className="aspect-square rounded-xl border border-dashed border-white/20 flex flex-col items-center justify-center text-slate-400 hover:text-white hover:bg-white/5 transition-colors">
                    <span className="text-2xl">+</span>
                    <span className="text-[10px]">Add More</span>
                  </div>
                </div>
              ) : (
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mx-auto text-3xl">🖼️</div>
                  <div className="space-y-1">
                    <p className="text-white font-medium">Drop your images here</p>
                    <p className="text-slate-400 text-sm">PNG, JPEG, WEBP or JPG</p>
                  </div>
                </div>
              )}
            </div>
            {images.length > 0 && (
              <p className="mt-3 text-xs text-slate-500 italic">
                * Note: Only the <strong>first image</strong> in the list will be processed.
              </p>
            )}
          </div>

          <div className="space-y-6">
            <h3 className="text-xl font-bold text-white">2. Select Enhancement Model</h3>
            <div className="grid grid-cols-2 gap-4">
              {[
                { id: 'edsr',   name: 'EDSR (x4)',   desc: 'Enhanced Deep Super-Resolution (4x upscale). Best balance.' },
                { id: 'lapsrn', name: 'LapSRN (x8)', desc: 'Laplacian Pyramid Super-Resolution (8x upscale). Maximum detail.' },
              ].map(m => (
                <button
                  key={m.id}
                  onClick={() => setModel(m.id)}
                  className={`p-4 rounded-xl border text-left transition-all duration-300 ${
                    model === m.id 
                      ? 'bg-indigo-600/20 border-indigo-500/50 ring-1 ring-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.25)]' 
                      : 'bg-white/5 border-white/10 hover:bg-white/10'
                  }`}
                >
                  <p className={`font-bold ${model === m.id ? 'text-indigo-300' : 'text-white'}`}>{m.name}</p>
                  <p className="text-xs text-slate-400 mt-1 line-clamp-2">{m.desc}</p>
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={handleProcess}
            disabled={images.length === 0 || loading}
            className={`w-full py-4 rounded-xl font-bold text-white transition-all duration-300 flex items-center justify-center gap-3 ${
              images.length === 0 || loading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-white/5'
                : 'bg-indigo-600 hover:bg-indigo-500 shadow-[0_10px_25px_rgba(79,70,229,0.4)] hover:shadow-[0_12px_30px_rgba(79,70,229,0.5)] transform hover:-translate-y-0.5'
            }`}
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Processing Image…</span>
              </>
            ) : (
              <>
                <span>🚀</span>
                <span>Upscale Image</span>
              </>
            )}
          </button>

          {error && (
            <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm flex items-start gap-3">
              <span className="mt-0.5 text-lg">⚠️</span>
              <p>{error}</p>
            </div>
          )}
        </div>

        {/* OUTPUT PANEL */}
        <div className="glass-panel p-8 min-h-[400px] flex flex-col">
          <h3 className="text-xl font-bold text-white mb-6">Enhanced Result</h3>
          
          {result ? (
            <div className="flex-1 flex flex-col animate-in fade-in duration-500">
              <div className="relative flex-1 rounded-2xl bg-black/40 border border-white/10 overflow-hidden mb-6 flex items-center justify-center group">
                <img 
                  src={`data:image/png;base64,${result.image_b64}`} 
                  alt="Enhanced Result" 
                  className="max-h-[500px] w-full object-contain p-2 transition-transform duration-500 group-hover:scale-[1.02]" 
                />
                <a
                  href={`data:image/png;base64,${result.image_b64}`}
                  download={`enhanced_${result.result_id}.png`}
                  className="absolute bottom-4 right-4 bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium shadow-lg transition-all opacity-0 group-hover:opacity-100 flex items-center gap-2 transform translate-y-2 group-hover:translate-y-0"
                >
                  <span>⬇️</span> Download PNG
                </a>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 rounded-xl bg-white/5 border border-white/5 text-center">
                  <p className="text-xs text-slate-500 mb-1">Model</p>
                  <p className="text-sm font-bold text-indigo-400">{result.method}</p>
                </div>
                <div className="p-4 rounded-xl bg-white/5 border border-white/5 text-center">
                  <p className="text-xs text-slate-500 mb-1">Time</p>
                  <p className="text-sm font-bold text-white">{result.time_seconds}s</p>
                </div>
                <div className="p-4 rounded-xl bg-white/5 border border-white/5 text-center">
                  <p className="text-xs text-slate-500 mb-1">Original Size</p>
                  <p className="text-sm font-bold text-slate-300">{result.original_size.width}×{result.original_size.height}</p>
                </div>
                <div className="p-4 rounded-xl bg-white/5 border border-white/5 text-center">
                  <p className="text-xs text-slate-500 mb-1">Enhanced Size</p>
                  <p className="text-sm font-bold text-emerald-400">{result.new_size.width}×{result.new_size.height}</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 border-2 border-dashed border-white/5 rounded-2xl flex flex-col items-center justify-center text-center p-8 space-y-4">
              <div className="text-5xl opacity-40">✨</div>
              <div>
                <p className="text-slate-400 font-medium italic">Upload an image and click 'Upscale'</p>
                <p className="text-slate-600 text-sm mt-2">The high-resolution result will appear here</p>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
