import { useState, useRef } from 'react'

// ── Metric card config ─────────────────────────────────────────────────────
const METRIC_META = {
  ssim:    { label: 'SSIM',             unit: '',    color: 'text-cyan-400',    bg: 'from-cyan-500/20 to-cyan-600/10',    border: 'border-cyan-500/30',    icon: '🔵', good: 'higher', tip: 'Structural Similarity Index — 1.0 is perfect.' },
  psnr:    { label: 'PSNR',             unit: ' dB', color: 'text-emerald-400', bg: 'from-emerald-500/20 to-emerald-600/10', border: 'border-emerald-500/30', icon: '🟢', good: 'higher', tip: 'Peak Signal-to-Noise Ratio — higher means less noise.' },
  mse:     { label: 'MSE',              unit: '',    color: 'text-rose-400',     bg: 'from-rose-500/20 to-rose-600/10',    border: 'border-rose-500/30',    icon: '🔴', good: 'lower',  tip: 'Mean Squared Error — lower means closer to original.' },
  entropy: { label: 'Entropy',          unit: ' bits',color: 'text-violet-400', bg: 'from-violet-500/20 to-violet-600/10', border: 'border-violet-500/30', icon: '🟣', good: 'higher', tip: 'Shannon Entropy — higher means richer detail/information.' },
  mi:      { label: 'Mutual Information',unit: '',   color: 'text-amber-400',   bg: 'from-amber-500/20 to-amber-600/10',  border: 'border-amber-500/30',  icon: '🟡', good: 'higher', tip: 'Mutual Information — measures shared information between images.' },
}

// ── Gauge bar ──────────────────────────────────────────────────────────────
function GaugeBar({ value, max, colorClass, reverse = false }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  const fill = reverse ? 100 - pct : pct
  return (
    <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden mt-2">
      <div
        className={`h-full rounded-full transition-all duration-1000 ease-out ${colorClass}`}
        style={{ width: `${fill}%` }}
      />
    </div>
  )
}

// ── Single metric card ─────────────────────────────────────────────────────
function MetricCard({ metricKey, value }) {
  const m = METRIC_META[metricKey]
  const MAX = { ssim: 1, psnr: 50, mse: 5000, entropy: 8, mi: 5 }

  return (
    <div className={`relative rounded-2xl border p-5 bg-gradient-to-br ${m.bg} ${m.border} backdrop-blur-sm overflow-hidden group hover:scale-[1.02] transition-transform duration-300`}>
      {/* glow orb */}
      <div className="absolute -top-6 -right-6 w-20 h-20 rounded-full opacity-20 blur-2xl bg-current" />

      <div className="flex items-start justify-between mb-1">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">{m.label}</span>
        <span className="text-lg">{m.icon}</span>
      </div>

      <div className={`text-3xl font-extrabold font-mono tracking-tight ${m.color}`}>
        {typeof value === 'number' ? value.toFixed(4) : '—'}{m.unit}
      </div>

      <GaugeBar
        value={value ?? 0}
        max={MAX[metricKey]}
        colorClass={`bg-gradient-to-r ${m.bg.replace('/20', '').replace('/10', '')}`}
        reverse={metricKey === 'mse'}
      />

      <p className="mt-3 text-[11px] text-slate-500 leading-snug">{m.tip}</p>
      <div className={`mt-2 inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full border ${m.border} ${m.color}`}>
        {m.good === 'higher' ? '↑ Higher is better' : '↓ Lower is better'}
      </div>
    </div>
  )
}

// ── Image upload slot ──────────────────────────────────────────────────────
function ImageSlot({ label, badge, badgeColor, img, onDrop, onClear, inputRef, onDim }) {
  const [dragging, setDragging] = useState(false)

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <span className={`px-3 py-1 rounded-full text-xs font-bold border ${badgeColor}`}>{badge}</span>
        <span className="text-sm font-semibold text-slate-300">{label}</span>
      </div>
      <div
        className={`relative rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer overflow-hidden aspect-square flex items-center justify-center
          ${dragging ? 'border-indigo-400 bg-indigo-500/20 scale-[1.02]' : img ? 'border-white/20 bg-black/40' : 'border-white/15 bg-white/5 hover:border-indigo-400/60 hover:bg-indigo-500/10'}`}
        onClick={() => !img && inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); onDrop(e.dataTransfer.files[0]) }}
      >
        {img ? (
          <>
            <img 
              src={img.url} 
              alt={label} 
              className="w-full h-full object-contain p-2" 
              onLoad={(e) => {
                if (!img.w) onDim(e.target.naturalWidth, e.target.naturalHeight)
              }}
            />
            <button
              onClick={e => { e.stopPropagation(); onClear() }}
              className="absolute top-2 right-2 w-7 h-7 rounded-full bg-red-500/90 text-white text-sm flex items-center justify-center hover:bg-red-400 transition-colors shadow-lg"
            >×</button>
            <div className="absolute bottom-2 left-2 right-2 text-center">
              <span className="inline-block bg-black/70 backdrop-blur-sm text-xs text-slate-300 px-3 py-1 rounded-full border border-white/10 truncate max-w-full">
                {img.name}
              </span>
            </div>
          </>
        ) : (
          <div className="text-center space-y-3 p-4">
            <div className="text-4xl opacity-50">🖼️</div>
            <p className="text-slate-400 text-sm font-medium">Drop image or click to browse</p>
            <p className="text-slate-600 text-xs">PNG · JPG · WEBP</p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={e => e.target.files[0] && onDrop(e.target.files[0])}
        />
      </div>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function ImageQualityPage() {
  const [original, setOriginal] = useState(null)
  const [enhanced, setEnhanced] = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [result,   setResult]   = useState(null)
  const [error,    setError]    = useState(null)

  const origRef = useRef()
  const enhRef  = useRef()

  const makeImg = file => file ? ({ file, url: URL.createObjectURL(file), name: file.name, w: 0, h: 0 }) : null

  const setOrig = file => { setOriginal(makeImg(file)); setResult(null); setError(null) }
  const setEnh  = file => { setEnhanced(makeImg(file)); setResult(null); setError(null) }
  const clearOrig = () => { if (original) URL.revokeObjectURL(original.url); setOriginal(null); setResult(null) }
  const clearEnh  = () => { if (enhanced) URL.revokeObjectURL(enhanced.url);  setEnhanced(null);  setResult(null) }

  const analyse = async () => {
    if (!original || !enhanced) { setError('Please upload both images.'); return }

    setLoading(true); setError(null); setResult(null)

    const origName = original.name.toLowerCase()
    const enhName = enhanced.name.toLowerCase()

    // 1. Validation: "Check the image same or not"
    // Since some models (like Super Resolution) change dimensions, check aspect ratio instead of absolute pixels.
    if (original.w && enhanced.w && original.h && enhanced.h) {
      const origRatio = original.w / original.h;
      const enhRatio = enhanced.w / enhanced.h;
      // Allow minor floating point differences (e.g., due to slight rounding in resizing)
      if (Math.abs(origRatio - enhRatio) > 0.05) {
        setError("Error: Images are not same. Please upload processed versions of the same original image.")
        setLoading(false); return
      }
    }



    // Ensure requests go directly to the newly generalized backend metrics logic

    try {
      const fd = new FormData()
      fd.append('original', original.file, original.file.name)
      fd.append('enhanced', enhanced.file, enhanced.file.name)
      const res  = await fetch('/api/image-quality', { method: 'POST', body: fd })
      const data = await res.json()
      if (!res.ok || data.error) throw new Error(data.error || 'Analysis failed')
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const ready = original && enhanced

  return (
    <div className="w-full flex flex-col gap-10 animate-in fade-in slide-in-from-bottom-4 duration-700">

      {/* ── HERO ── */}
      <div className="text-center pt-8 pb-2">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 mb-5 rounded-full bg-violet-500/15 border border-violet-400/25 text-violet-300 font-medium text-sm">
          🔬 Image Quality Analyser
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold mb-5 tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white via-violet-200 to-violet-400">
          Original vs Enhanced
        </h1>
        <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
          Upload your <span className="text-white font-semibold">original image</span> and the <span className="text-violet-300 font-semibold">model-enhanced version</span>.
          We measure SSIM, PSNR, MSE, Entropy and Mutual&nbsp;Information to quantify the improvement.
        </p>
      </div>

      {/* ── UPLOAD ROW ── */}
      <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-3xl shadow-2xl p-8">
        <div className="grid md:grid-cols-2 gap-8">
          <ImageSlot
            label="Original / Reference Image"
            badge="REF"
            badgeColor="border-slate-500/50 text-slate-400 bg-slate-800/60"
            img={original}
            onDrop={setOrig}
            onClear={clearOrig}
            onDim={(w,h) => setOriginal(prev => ({...prev, w, h}))}
            inputRef={origRef}
          />
          <ImageSlot
            label="Enhanced / Model Output"
            badge="ENHANCED"
            badgeColor="border-violet-500/50 text-violet-300 bg-violet-900/40"
            img={enhanced}
            onDrop={setEnh}
            onClear={clearEnh}
            onDim={(w,h) => setEnhanced(prev => ({...prev, w, h}))}
            inputRef={enhRef}
          />
        </div>

        {error && (
          <div className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 flex items-center gap-3 text-sm">
            <span className="text-xl">⚠️</span> {error}
          </div>
        )}

        <div className="flex gap-4 mt-8">
          <button
            id="btn-analyse"
            onClick={analyse}
            disabled={!ready || loading}
            className={`flex-1 py-4 rounded-xl font-bold text-lg transition-all duration-300 flex items-center justify-center gap-3 ${
              !ready || loading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-white/5'
                : 'bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white shadow-[0_8px_25px_rgba(139,92,246,0.4)] hover:shadow-[0_10px_30px_rgba(139,92,246,0.55)] hover:-translate-y-0.5'
            }`}
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Analysing Images…
              </>
            ) : (
              <><span>🔬</span> Analyse Quality Metrics</>
            )}
          </button>
          {(original || enhanced) && (
            <button
              onClick={() => { clearOrig(); clearEnh(); setError(null); setResult(null) }}
              className="px-6 py-4 rounded-xl font-medium text-slate-400 bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-sm"
            >
              🗑️ Clear All
            </button>
          )}
        </div>
      </div>

      {/* ── LOADING ── */}
      {loading && (
        <div className="flex flex-col items-center gap-5 py-16 text-slate-400">
          <div className="relative w-16 h-16">
            <div className="absolute inset-0 rounded-full border-4 border-violet-500/30 border-t-violet-500 animate-spin" />
            <div className="absolute inset-2 rounded-full border-4 border-indigo-500/20 border-b-indigo-400 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '0.7s' }} />
          </div>
          <p className="text-lg font-semibold">Computing image quality metrics…</p>
          <p className="text-sm text-slate-600">SSIM · Entropy · Mutual Information</p>
        </div>
      )}

      {/* ── RESULTS ── */}
      {result && !loading && (
        <>
          {/* Visual comparison strip */}
          <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-3xl shadow-2xl p-8">
            <div className="flex items-center gap-3 mb-8">
              <span className="p-2.5 rounded-xl bg-white/10 text-lg">🖼️</span>
              <h2 className="text-2xl font-bold text-white">Visual Comparison</h2>
            </div>
            <div className="grid md:grid-cols-3 gap-6">
              {/* Original */}
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded-full bg-slate-700 text-slate-300 text-xs font-bold border border-slate-600">REF</span>
                  <span className="text-sm text-slate-400 font-medium">Original Image</span>
                </div>
                <div className="aspect-square rounded-2xl overflow-hidden border border-white/10 bg-black/40">
                  <img src={original.url} alt="Original" className="w-full h-full object-contain p-2" />
                </div>
                <p className="text-xs text-slate-600 text-center truncate">{original.name}</p>
              </div>

              {/* Enhanced */}
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded-full bg-violet-900 text-violet-300 text-xs font-bold border border-violet-700">ENHANCED</span>
                  <span className="text-sm text-slate-400 font-medium">Model Output</span>
                </div>
                <div className="aspect-square rounded-2xl overflow-hidden border border-violet-500/30 bg-black/40 shadow-[0_0_20px_rgba(139,92,246,0.15)]">
                  <img src={enhanced.url} alt="Enhanced" className="w-full h-full object-contain p-2" />
                </div>
                <p className="text-xs text-slate-600 text-center truncate">{enhanced.name}</p>
              </div>

              {/* Diff / summary */}
              <div className="flex flex-col justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <span className="px-2 py-0.5 rounded-full bg-indigo-900 text-indigo-300 text-xs font-bold border border-indigo-700">ΔDIFF</span>
                    <span className="text-sm text-slate-400 font-medium">Quality Delta</span>
                  </div>
                  <div className="space-y-3">
                    {['ssim', 'entropy', 'mi'].map(k => {
                      const m = METRIC_META[k]
                      const v = result.metrics[k]
                      const orig = result.original_metrics?.[k]
                      const delta = orig != null ? v - orig : null
                      const improved = delta != null && (m.good === 'higher' ? delta > 0 : delta < 0)
                      return (
                        <div key={k} className="flex items-center justify-between bg-white/5 rounded-xl px-4 py-2.5 border border-white/5">
                          <span className="text-xs text-slate-400 font-semibold">{m.label}</span>
                          <div className="flex items-center gap-2">
                            <span className={`font-mono text-sm font-bold ${m.color}`}>{typeof v === 'number' ? v.toFixed(3) : '—'}{m.unit}</span>
                            {delta != null && (
                              <span className={`text-xs font-bold px-1.5 py-0.5 rounded-md ${improved ? 'text-emerald-400 bg-emerald-500/10' : 'text-rose-400 bg-rose-500/10'}`}>
                                {improved ? '▲' : '▼'} {Math.abs(delta).toFixed(3)}
                              </span>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Enhanced-image metrics */}
          <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-3xl shadow-2xl p-8">
            <div className="flex items-center gap-3 mb-8">
              <span className="p-2.5 rounded-xl bg-white/10 text-lg">📊</span>
              <div>
                <h2 className="text-2xl font-bold text-white">Quantitative Metrics</h2>
                <p className="text-sm text-slate-500 mt-0.5">Enhanced image vs. original reference</p>
              </div>
            </div>
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
              {['ssim', 'entropy', 'mi'].map(k => (
                <MetricCard key={k} metricKey={k} value={result.metrics[k]} />
              ))}
            </div>
          </div>

          {/* Research-paper-style table */}
          <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-3xl shadow-2xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <span className="p-2.5 rounded-xl bg-white/10 text-lg">📋</span>
              <h2 className="text-2xl font-bold text-white">Metrics Summary Table</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-3 px-4 text-slate-400 font-semibold">Metric</th>
                    <th className="text-center py-3 px-4 text-slate-400 font-semibold">Original Image</th>
                    <th className="text-center py-3 px-4 text-slate-400 font-semibold">Enhanced Image</th>
                    <th className="text-center py-3 px-4 text-slate-400 font-semibold">Change</th>
                  </tr>
                </thead>
                <tbody>
                    {['ssim', 'entropy', 'mi'].map(k => {
                      const m = METRIC_META[k]
                      const v = result.metrics[k]
                      const orig = result.original_metrics?.[k]
                      // For MSE, improvement is orig - v (lower is better)
                      // For others, improvement is v - orig (higher is better)
                      const delta = orig != null ? (m.good === 'higher' ? v - orig : orig - v) : null
                      const improved = delta != null && delta >= 0
                      return (
                        <tr key={k} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                          <td className="py-3.5 px-4 font-semibold text-slate-200">
                            <div className="flex items-center gap-2">
                              <span>{m.icon}</span>
                              <div>
                                <div>{m.label}</div>
                                <div className="text-[10px] text-slate-600 font-normal">{m.good === 'higher' ? '↑ higher = better' : '↓ lower = better'}</div>
                              </div>
                            </div>
                          </td>
                          <td className="py-3.5 px-4 text-center font-mono text-slate-300">
                            {orig != null ? orig.toFixed(4) : <span className="text-slate-600 text-xs italic">self</span>}{m.unit}
                          </td>
                          <td className={`py-3.5 px-4 text-center font-mono font-bold ${m.color}`}>
                            {typeof v === 'number' ? v.toFixed(4) : '—'}{m.unit}
                          </td>
                          <td className="py-3.5 px-4 text-center font-mono">
                            {delta != null ? (
                              <span className={`font-bold ${improved ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {delta >= 0 ? '+' : ''}{delta.toFixed(4)}
                              </span>
                            ) : <span className="text-slate-600">—</span>}
                          </td>
                        </tr>
                      )
                    })}
                </tbody>
              </table>
            </div>

            {/* Summary */}
            <div className="mt-8 p-5 rounded-2xl bg-indigo-500/10 border border-indigo-500/25">
              <p className="text-xs font-bold text-indigo-400 uppercase tracking-widest mb-2">💡 Summary</p>
              <p className="text-sm text-slate-300 italic leading-relaxed">
                "We evaluate enhancement performance using <strong className="text-white">SSIM</strong>,
                <strong className="text-white"> Entropy</strong>, and <strong className="text-white">Mutual Information</strong> to ensure the enhanced image
                preserves structural similarity, maximises signal quality, and retains rich information content from the original."
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  )
}