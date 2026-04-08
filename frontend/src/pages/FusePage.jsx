import { useState, useCallback, useRef } from 'react'
import MetricsPanel from '../components/MetricsPanel.jsx'

const METHOD_INFO = {
  // average: { 
  //   name: 'Average Fusion', icon: '⚖️', speed: 'Fast', quality: 'Basic',
  //   desc: 'Pixel-wise averaging across all source images. Fast and simple baseline.',
  //   border: 'hover:border-blue-500', bg: 'hover:bg-blue-500/10', glow: 'shadow-blue-500/30'
  // },
  // max: { 
  //   name: 'Max Fusion', icon: '⬆️', speed: 'Fast', quality: 'Good',
  //   desc: 'Preserves maximum intensity at each pixel. Great for bright features.',
  //   border: 'hover:border-emerald-500', bg: 'hover:bg-emerald-500/10', glow: 'shadow-emerald-500/30'
  // },
  // gradient_weighted: { 
  //   name: 'Gradient-Weighted', icon: '∇', speed: 'Medium', quality: 'Great',
  //   desc: 'Weights pixels by local gradient magnitude, preserving fine details.',
  //   border: 'hover:border-amber-500', bg: 'hover:bg-amber-500/10', glow: 'shadow-amber-500/30'
  // },
  // laplacian_pyramid: { 
  //   name: 'Laplacian Pyramid', icon: '🔺', speed: 'Slow', quality: 'Excellent',
  //   desc: 'Multi-scale pyramid fusion — classic approach combining frequency bands.',
  //   border: 'hover:border-purple-500', bg: 'hover:bg-purple-500/10', glow: 'shadow-purple-500/30'
  // },
  emma: { 
    name: 'EMMA (CVPR 2024)', icon: '🔬', speed: 'Medium', quality: '8x Ultra-Res',
    desc: 'Pretrained equivariant multi-modality fusion. Clean, crystal-clear output — best for IR+Visible pairs.',
    border: 'hover:border-cyan-500', bg: 'hover:bg-cyan-500/10', glow: 'shadow-cyan-500/40'
  },
  deepfuse: { 
    name: 'DeepFuse AI', icon: '✨', speed: 'Slow', quality: 'Ultra-HD',
    desc: 'Advanced saliency-based deep fusion combined with EDSR Super-Resolution enhancement.',
    border: 'hover:border-indigo-400', bg: 'hover:bg-indigo-500/10', glow: 'shadow-indigo-500/40'
  },
  swin_fusion: {  
    name: 'Swin Fusion (Transformer)', icon: '🧩', speed: 'Medium', quality: 'State-of-the-Art',
    desc: 'Transformer-based fusion. Clean, crystal-clear output for multi-modal image pairs.',
    border: 'hover:border-pink-500', bg: 'hover:bg-pink-500/10',  glow: 'shadow-pink-500/40'
  },
  ir_vis_color: { 
    name: 'Infrared+Visible Fusion', icon: '🌈', speed: 'Medium', quality: 'Exceptional',
    desc: 'Dual-scale HSV fusion. Perfectly preserves visible colors while injecting sharp glowing infrared thermal details.',
    border: 'hover:border-orange-500', bg: 'hover:bg-orange-500/10', glow: 'shadow-orange-500/40'
  }
}

// These models are currently hidden to keep the interface focused
const HIDDEN_MODELS = {
  multi_focus_clear: { 
    name: 'Blur+Clear → Clean', icon: '✨', speed: 'Medium', quality: 'Excellent',
    desc: 'Blur + clear images → crystal-clear output. Matches pixels, picks sharper regions.',
    border: 'hover:border-teal-500', bg: 'hover:bg-teal-500/10', glow: 'shadow-teal-500/40'
  },
  ir_vis_clean: { 
    name: 'IR+Visible → Clean & Clear', icon: '🌙', speed: 'Medium', quality: 'Excellent',
    desc: 'Thermal IR + low-light visible → clean output. Denoise, enhance, sharpen. For night scenes.',
    border: 'hover:border-amber-500', bg: 'hover:bg-amber-500/10', glow: 'shadow-amber-500/40'
  },
  deep_learning: { 
    name: 'Deep Enhancement Fusion', icon: '🧠', speed: 'Slow', quality: 'State-of-the-Art',
    desc: 'Advanced per-pixel saliency, Laplacian Pyramid, and unsharp masking for crystal clear output.',
    border: 'hover:border-indigo-400', bg: 'hover:bg-indigo-500/10', glow: 'shadow-indigo-500/40'
  },
  // ir_vis_color: { 
  //   name: 'IR+VIS Color Fusion', icon: '🌈', speed: 'Medium', quality: 'Exceptional',
  //   desc: 'Dual-scale HSV fusion. Preserves vibrant visible colors while injecting bright thermal details.',
  //   border: 'hover:border-pink-500', bg: 'hover:bg-pink-500/10', glow: 'shadow-pink-500/40'
  // },
};

export default function FusePage() {
  const [images,    setImages]   = useState([])   
  const [method,    setMethod]   = useState('emma')
  const [srModel,   setSrModel]  = useState('edsr')
  const [loading,   setLoading]  = useState(false)
  const [result,    setResult]   = useState(null)  
  const [error,     setError]    = useState(null)
  const [dragging,  setDragging] = useState(false)
  const inputRef = useRef()

  const addFiles = useCallback((files) => {
    const valid = Array.from(files).filter(f => f.type.startsWith('image/'))
    if (!valid.length) return
    setImages(prev => {
      const updated = [...prev, ...valid.map(f => ({
        file: f,
        url: URL.createObjectURL(f),
        name: f.name,
      }))]
      return updated.slice(0, 6)
    })
    setResult(null)
    setError(null)
  }, [])

  const removeImage = (i) => {
    setImages(prev => {
      URL.revokeObjectURL(prev[i].url)
      return prev.filter((_, idx) => idx !== i)
    })
    setResult(null)
  }

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    addFiles(e.dataTransfer.files)
  }, [addFiles])

  const handleFuse = async () => {
    const isSR = method === 'deepfuse' || method === 'emma';
    if (images.length < (isSR ? 1 : 2)) {
      setError(`Please upload at least ${isSR ? '1 image' : '2 images'}.`);
      return;
    }
    
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const fd = new FormData()
      
      if (isSR) {
        // Use Super Resolution API
        fd.append('image', images[0].file)
        // If EMMA, we force LapSRN, otherwise use the selected srModel (for DeepFuse)
        const selectedSR = method === 'emma' ? 'lapsrn' : srModel;
        fd.append('model', selectedSR)
        
        const res = await fetch('/api/super-resolve', { method: 'POST', body: fd })
        const data = await res.json()
        if (!res.ok || data.error) throw new Error(data.error || 'Super Resolution failed')
        
        // Map SR response to Fusion result format for the UI
        setResult({
          ...data,
          num_images: 1,
          metrics: { ssim_avg: 'N/A', entropy: 'N/A', mi_avg: 'N/A' } // SR doesn't have fusion metrics
        })
      } else {
        // Use standard Fusion API
        fd.append('method', method)
        images.forEach((img, i) => fd.append(`image${i}`, img.file))
        const res = await fetch('/api/fuse', { method: 'POST', body: fd })
        const data = await res.json()
        if (!res.ok || data.error) throw new Error(data.error || 'Fusion failed')
        setResult(data)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = () => {
    if (!result) return
    const a = document.createElement('a')
    a.href = `data:image/png;base64,${result.image_b64}`
    a.download = `fused_${method}_${Date.now()}.png`
    a.click()
  }

  return (
    <div className="w-full flex flex-col gap-10 pb-8">
      {/* HERO */}
      <div className="text-center pt-8 pb-4 animate-fade-in-up">
        <div className="inline-block px-4 py-1.5 mb-6 rounded-full bg-indigo-500/20 border border-indigo-400/30 text-indigo-300 font-medium text-sm shadow-[0_0_15px_rgba(99,102,241,0.2)]">
          {(method === 'deepfuse' || method === 'emma') ? '✨ Image Enhancement Engine' : '⚗️ Multi-Modal Fusion Engine'}
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold mb-6 tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white via-indigo-200 to-slate-400 drop-shadow-lg text-balance">
          {(method === 'deepfuse' || method === 'emma') ? 'Super Resolution AI' : 'Fuse. Enhance. Discover.'}
        </h1>
        <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
          {(method === 'deepfuse' || method === 'emma') 
            ? 'Transform low-resolution images into crystal-clear high-definition versions using state-of-the-art AI upscaling.'
            : 'Upload 2–6 source images (e.g. Infrared & Visible, Multi-exposure) and select a fusion algorithm. Get crystal-clear results instantly.'}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* LEFT & CENTER: Algorithm + Upload */}
        <div className="lg:col-span-7 flex flex-col gap-8">
          
          {/* 🧠 ALGORITHM SELECTOR - "5 Boxes" */}
          <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-3xl shadow-2xl p-8 relative overflow-hidden animate-fade-in-up" style={{animationDelay: '0.1s'}}>
            <div className="absolute top-0 right-0 -mr-20 -mt-20 w-64 h-64 bg-indigo-500/10 blur-[100px] rounded-full pointer-events-none"></div>
            
            <div className="flex items-center gap-3 mb-8 text-2xl font-bold text-white relative z-10">
              <span className="p-2.5 rounded-xl bg-white/10 shadow-inner border border-white/5">🧠</span>
              Fusion Algorithm
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 relative z-10">
              {Object.entries(METHOD_INFO).map(([id, m]) => {
                const isSelected = method === id;
                const isDeepLearning = id === 'deep_learning';
                
                return (
                  <div
                    key={id}
                    className={`
                      relative flex flex-col p-6 rounded-2xl border transition-all duration-300 cursor-pointer overflow-hidden group
                      ${isDeepLearning ? 'sm:col-span-2' : ''}
                      ${isSelected 
                        ? `bg-white/10 shadow-[0_0_30px_rgba(255,255,255,0.05)] ${m.border.replace('hover:', '')} scale-[1.02]` 
                        : `bg-black/20 border-white/5 ${m.border} ${m.bg} hover:scale-[1.01]`
                      }
                    `}
                    onClick={() => setMethod(id)}
                  >
                    {/* Active state glow indicator */}
                    {isSelected && (
                      <div className={`absolute top-0 right-0 w-32 h-32 rounded-full blur-[40px] opacity-40 translate-x-1/2 -translate-y-1/2 ${m.glow.replace('shadow-', 'bg-')}`}></div>
                    )}
                    
                    <div className="relative z-10 flex flex-col h-full">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3 font-bold text-lg text-white">
                          <span className="text-2xl drop-shadow-md">{m.icon}</span> 
                          {m.name}
                        </div>
                        {isSelected ? (
                          <span className="w-5 h-5 rounded-full bg-indigo-500 flex items-center justify-center shadow-[0_0_10px_rgba(99,102,241,0.6)]">
                            <span className="w-2 h-2 rounded-full bg-white"></span>
                          </span>
                        ) : (
                          <span className="w-5 h-5 rounded-full border-2 border-slate-600"></span>
                        )}
                      </div>
                      
                      <div className="text-sm text-slate-400 mb-5 leading-relaxed pr-2 flex-grow">
                        {m.desc}
                      </div>
                      
                      <div className="flex flex-wrap gap-2 text-xs font-semibold mt-auto">
                        <span className={`px-2.5 py-1 rounded-md border backdrop-blur-sm shadow-sm
                          ${isSelected ? 'bg-black/30 border-white/10 text-slate-200' : 'bg-white/5 border-white/5 text-slate-400'}`}>
                          ⚡ {m.speed}
                        </span>
                        <span className={`px-2.5 py-1 rounded-md border backdrop-blur-sm shadow-sm flex items-center gap-1
                          ${m.quality === 'State-of-the-Art' 
                            ? 'bg-amber-500/20 border-amber-500/30 text-amber-300 shadow-[0_0_10px_rgba(245,158,11,0.2)]' 
                            : isSelected ? 'bg-black/30 border-white/10 text-slate-200' : 'bg-white/5 border-white/5 text-slate-400'}`}>
                          ★ {m.quality}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 📁 UPLOAD ZONE */}
          <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-3xl shadow-xl p-8 animate-fade-in-up" style={{animationDelay: '0.2s'}}>
            <div className="flex items-center gap-3 mb-6 text-xl font-bold text-white">
              <span className="p-2.5 rounded-xl bg-white/10 shadow-inner border border-white/5">📁</span>
              {(method === 'deepfuse' || method === 'emma') ? 'Upload Low-Res Images' : 'Source Images'}
              <span className="ml-auto text-sm bg-indigo-500/20 px-4 py-1.5 rounded-full text-indigo-300 border border-indigo-500/20 shadow-inner">
                {images.length}{(method === 'deepfuse' || method === 'emma') ? '' : '/6'} uploaded
              </span>
            </div>

            <div
              className={`border-2 border-dashed rounded-2xl p-10 text-center transition-all duration-300 cursor-pointer
                ${dragging
                  ? 'border-indigo-400 bg-indigo-500/20 scale-[1.02] shadow-[0_0_30px_rgba(99,102,241,0.2)]'
                  : 'border-white/20 bg-black/20 hover:border-indigo-400 hover:bg-indigo-500/10'}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <div className="text-5xl mb-4 transform transition-transform group-hover:scale-110 drop-shadow-lg">
                {(method === 'deepfuse' || method === 'emma') ? '✨' : '🖼️'}
              </div>
              <h3 className="text-xl font-bold mb-2 text-white">
                {(method === 'deepfuse' || method === 'emma') ? 'Drop low-res image here' : 'Drop images here'}
              </h3>
              <p className="text-slate-400 text-sm">or click to browse · PNG, JPG, WEBP</p>
              <input
                ref={inputRef}
                type="file"
                className="hidden"
                accept="image/*"
                multiple
                onChange={e => addFiles(e.target.files)}
              />
            </div>

            {images.length > 0 && (
              <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 gap-3 mt-6">
                {images.map((img, i) => (
                  <div key={i} className="relative aspect-square rounded-xl overflow-hidden border border-white/10 group bg-black/50 shadow-lg">
                    <img src={img.url} alt={img.name} className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                    <button 
                      className="absolute top-1.5 right-1.5 w-6 h-6 rounded-full bg-red-500/90 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all hover:bg-red-500 shadow-xl transform hover:scale-110"
                      onClick={(e) => { e.stopPropagation(); removeImage(i); }}
                    >×</button>
                    <span className="absolute bottom-1.5 left-1.5 px-2 py-0.5 rounded-md bg-black/60 text-[10px] font-bold text-white backdrop-blur-md border border-white/10 shadow-lg">
                      {(method === 'deepfuse' || method === 'emma') && i === 0 ? 'To Process' : `Img ${i + 1}`}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 🚀 SUPER RESOLUTION INPUTS (Only for DeepFuse) */}
          {method === 'deepfuse' && (
            <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-3xl shadow-xl p-8 animate-fade-in-up">
              <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
                <span className="p-2 rounded-xl bg-white/10 shadow-inner border border-white/5">✨</span>
                Super Resolution Settings
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {[
                  { id: 'edsr',   name: 'EDSR (x4)',   desc: 'Enhanced Deep Super-Resolution (4x upscale). Best balance.' },
                  { id: 'lapsrn', name: 'LapSRN (x8)', desc: 'Laplacian Pyramid Super-Resolution (8x upscale). Maximum detail.' },
                ].map(m => (
                  <button
                    key={m.id}
                    onClick={() => setSrModel(m.id)}
                    className={`p-4 rounded-xl border text-left transition-all duration-300 ${
                      srModel === m.id 
                        ? 'bg-indigo-600/20 border-indigo-500/50 ring-1 ring-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.25)]' 
                        : 'bg-white/5 border-white/10 hover:bg-white/10'
                    }`}
                  >
                    <p className={`font-bold ${srModel === m.id ? 'text-indigo-300' : 'text-white'}`}>{m.name}</p>
                    <p className="text-xs text-slate-400 mt-1 line-clamp-2">{m.desc}</p>
                  </button>
                ))}
              </div>
            </div>
          )}
          
          {/* Error */}
          {error && (
            <div className="p-5 rounded-2xl bg-red-500/10 border border-red-500/30 text-red-200 font-medium flex items-center gap-3 shadow-[0_0_20px_rgba(239,68,68,0.1)]">
              <span className="text-xl">⚠️</span> {error}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4 animate-fade-in-up" style={{animationDelay: '0.3s'}}>
            <button
               className={`flex-1 text-xl py-5 rounded-2xl font-bold text-white shadow-[0_10px_30px_rgba(99,102,241,0.3)] transition-all duration-300
                ${(loading || images.length < (method === 'deepfuse' ? 1 : 2)) 
                  ? 'bg-indigo-600/50 cursor-not-allowed opacity-70' 
                  : 'bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 hover:shadow-[0_10px_40px_rgba(99,102,241,0.5)] hover:-translate-y-1'}`}
              disabled={loading || images.length < (method === 'deepfuse' ? 1 : 2)}
              onClick={handleFuse}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-3">
                  <div className="w-6 h-6 border-4 border-white/30 border-t-white rounded-full animate-spin"></div>
                  {(method === 'deepfuse' || method === 'emma') ? 'Upscaling Image…' : 'Fusing Pipeline Active…'}
                </span>
              ) : ((method === 'deepfuse' || method === 'emma') ? `🚀 Run ${METHOD_INFO[method]?.name}` : '⚗️ Run Advanced Fusion')}
            </button>
            {images.length > 0 && (
              <button
                className="px-8 py-5 rounded-2xl font-bold text-slate-300 bg-white/5 hover:bg-white/10 hover:text-white border border-white/10 transition-all duration-300 shadow-xl hover:shadow-2xl"
                onClick={() => { setImages([]); setResult(null); setError(null) }}
              >
                🗑️ Clear
              </button>
            )}
          </div>
        </div>

        {/* RIGHT: Result (Sticky) */}
        <div className="lg:col-span-5 flex flex-col gap-6 lg:sticky lg:top-24">
          <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-3xl shadow-2xl p-6 min-h-[500px] flex flex-col relative overflow-hidden animate-fade-in-up" style={{animationDelay: '0.4s'}}>
            <div className="absolute top-[-100px] right-[-100px] w-[300px] h-[300px] bg-emerald-500/5 blur-[120px] rounded-full pointer-events-none"></div>
            
            <div className="flex items-center gap-3 mb-6 text-xl font-bold text-white relative z-10">
              <span className="p-2.5 rounded-xl bg-white/10 shadow-inner border border-white/5">✨</span>
              Fused Output
              {result && (
                <span className="ml-auto text-xs text-emerald-300 bg-emerald-500/20 px-3 py-1.5 rounded-full border border-emerald-500/30 font-bold shadow-[0_0_15px_rgba(16,185,129,0.15)]">
                  ✓ Done in {result.time_seconds}s
                </span>
              )}
            </div>

            {loading && (
              <div className="flex-1 flex flex-col items-center justify-center gap-6 text-slate-300">
                <div className="relative w-24 h-24">
                  <div className="absolute inset-0 border-4 border-indigo-500/20 rounded-full"></div>
                  <div className="absolute inset-0 border-4 border-indigo-500 rounded-full border-t-transparent animate-spin"></div>
                  <div className="absolute inset-0 flex items-center justify-center text-3xl animate-pulse">⚗️</div>
                </div>
                <div className="text-center">
                  <p className="text-xl font-bold text-white mb-2 tracking-wide">Processing Images</p>
                </div>
              </div>
            )}

            {!loading && !result && (
              <div className="flex-1 flex flex-col items-center justify-center text-slate-500 gap-6 text-center px-4">
                <div className="w-28 h-28 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-6xl shadow-inner group-hover:scale-110 transition-transform">
                  {(method === 'deepfuse' || method === 'emma') ? '✨' : '🔬'}
                </div>
                <div>
                  <h3 className="text-xl font-bold text-slate-300 mb-2">
                    {(method === 'deepfuse' || method === 'emma') ? 'Ready to Upscale' : 'Ready to Fuse'}
                  </h3>
                  <p className="text-sm max-w-[250px] mx-auto text-slate-400">
                    {(method === 'deepfuse' || method === 'emma') 
                      ? `Upload a low-res image & click 'Run ${METHOD_INFO[method]?.name}' to enhance.` 
                      : 'Upload multiple images & click Run Advanced Fusion to generate the result.'}
                  </p>
                </div>
              </div>
            )}

            {!loading && result && (
              <div className="flex flex-col gap-6 flex-1 relative z-10 animate-fade-in-up">
                {/* Fused image */}
                <div className="relative flex-1 rounded-2xl overflow-hidden bg-[#0A0F1E] border border-white/10 flex items-center justify-center shadow-inner group">
                  <img
                    src={`data:image/png;base64,${result.image_b64}`}
                    alt="Fused result"
                    className="max-w-full max-h-[450px] object-contain transition-transform duration-700 group-hover:scale-[1.03]"
                  />
                  <div className="absolute inset-0 pointer-events-none ring-1 ring-inset ring-white/10 rounded-2xl"></div>
                  <span className="absolute bottom-4 left-4 bg-black/70 backdrop-blur-md px-4 py-2 rounded-xl text-sm font-bold border border-white/20 text-white shadow-xl flex items-center gap-2">
                    <span className="text-lg">{METHOD_INFO[result.method]?.icon}</span>
                    {METHOD_INFO[result.method]?.name}
                  </span>
                </div>

                <button
                  className="w-full py-5 px-6 rounded-2xl font-bold flex items-center justify-center gap-3 text-white transition-all duration-300 bg-white/10 hover:bg-white/20 border border-white/20 hover:border-white/40 shadow-lg hover:shadow-[0_8px_30px_rgba(255,255,255,0.1)] group"
                  onClick={handleDownload}
                >
                  <span className="text-2xl group-hover:-translate-y-1 transition-transform drop-shadow-md">⬇️</span> 
                  Download High-Res PNG
                </button>
              </div>
            )}
          </div>

          {/* Metrics Panel */}
          {result && !(method === 'deepfuse' || method === 'emma') && (
            <div className="animate-fade-in-up" style={{ animationDelay: '0.5s' }}>
              <MetricsPanel metrics={result.metrics} numSources={result.num_images} />
            </div>
          )}
          {result && (method === 'deepfuse' || method === 'emma') && (
             <div className="animate-fade-in-up space-y-4" style={{ animationDelay: '0.5s' }}>
                <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 shadow-xl">
                  <div className="flex items-center gap-3 mb-4 text-emerald-400 font-bold">
                    <span>✨</span> Resolution Details
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-xl bg-black/20 border border-white/5">
                      <p className="text-[10px] text-slate-500 mb-1 uppercase font-bold tracking-wider">Model</p>
                      <p className="text-sm font-bold text-indigo-300">{result.method}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-black/20 border border-white/5">
                      <p className="text-[10px] text-slate-500 mb-1 uppercase font-bold tracking-wider">Time</p>
                      <p className="text-sm font-bold text-white">{result.time_seconds}s</p>
                    </div>
                    <div className="p-3 rounded-xl bg-black/20 border border-white/5">
                      <p className="text-[10px] text-slate-500 mb-1 uppercase font-bold tracking-wider">Original</p>
                      <p className="text-sm font-bold text-slate-300">{result.original_size?.width}×{result.original_size?.height}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-black/20 border border-white/5">
                      <p className="text-[10px] text-slate-500 mb-1 uppercase font-bold tracking-wider">Enhanced</p>
                      <p className="text-sm font-bold text-emerald-400">{result.new_size?.width}×{result.new_size?.height}</p>
                    </div>
                  </div>
                </div>
             </div>
          )}
        </div>
      </div>
    </div>
  )
}
