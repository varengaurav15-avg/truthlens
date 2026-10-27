import { useState, useRef, useEffect, useCallback } from 'react'
import { Upload, Activity, AlertTriangle, CheckCircle2, XCircle, HelpCircle, Loader2 } from 'lucide-react'
import gsap from 'gsap'
import {
  analyzeText,
  analyzeImage,
  analyzeFusion,
  deriveAxes,
  verdictColor,
  ApiError,
} from '../lib/api'
import { ScoreRing, TruthGraph } from './TruthGraph'
import Corrections from './Corrections'

const LABEL_ICON = {
  SUPPORTED: CheckCircle2,
  REFUTED: XCircle,
  NOT_ENOUGH_INFO: HelpCircle,
}
const LABEL_COLOR = {
  SUPPORTED: '#4ade80',
  REFUTED: '#f87171',
  NOT_ENOUGH_INFO: '#888888',
}

export default function DetectorPanel({ activeTab }) {
  const panelRef = useRef(null)
  const resultsRef = useRef(null)

  const [text, setText] = useState('')
  const [maxClaims, setMaxClaims] = useState(10)
  const [sourceRep, setSourceRep] = useState(0.5)
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)
  const [elapsed, setElapsed] = useState(0)

  /* panel transition on tab change */
  useEffect(() => {
    if (panelRef.current) {
      gsap.fromTo(
        panelRef.current,
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.8, ease: 'power3.out' },
      )
    }
    setResults(null)
    setError(null)
    setLoading(false)
  }, [activeTab])

  /* elapsed timer while a request is in flight */
  useEffect(() => {
    if (!loading) return
    setElapsed(0)
    const t0 = Date.now()
    const id = setInterval(() => setElapsed((Date.now() - t0) / 1000), 100)
    return () => clearInterval(id)
  }, [loading])

  /* GSAP stagger reveal — score ring, then graph, then explanation, then claims */
  useEffect(() => {
    if (!results || !resultsRef.current) return
    const items = resultsRef.current.querySelectorAll('.reveal')
    if (!items.length) return
    const tl = gsap.timeline()
    tl.fromTo(
      items,
      { opacity: 0, y: 26, filter: 'blur(6px)' },
      {
        opacity: 1,
        y: 0,
        filter: 'blur(0px)',
        duration: 0.85,
        ease: 'power3.out',
        stagger: 0.13,
      },
    )
    return () => tl.kill()
  }, [results])

  const handleImageChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
  }

  const run = useCallback(async (fn) => {
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      setResults(await fn())
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : (err?.message ?? 'Unexpected error during analysis.'),
      )
      // eslint-disable-next-line no-console
      console.error('[TruthLens]', err)
    } finally {
      setLoading(false)
    }
  }, [])

  const onRun = () => {
    if (activeTab === 'text') return run(() => analyzeText(text, maxClaims))
    if (activeTab === 'image') return run(() => analyzeImage(imageFile))
    return run(() => analyzeFusion({ text, file: imageFile, sourceReputation: sourceRep }))
  }

  const disabled =
    loading ||
    (activeTab === 'text' && !text.trim()) ||
    (activeTab === 'image' && !imageFile) ||
    (activeTab === 'fusion' && !text.trim() && !imageFile)

  /* ── normalise the three response shapes into one view model ── */
  const score = results?.credibility_score ?? 0
  const claims = results?.claims ?? results?.text_analysis?.claims ?? []
  const heatmap =
    results?.gradcam_heatmap_b64 ?? results?.vision_analysis?.gradcam_heatmap_b64 ?? null
  const explanation =
    results?.explanation ||
    [results?.text_analysis?.explanation, results?.vision_analysis?.explanation]
      .filter((s) => s && s !== 'No text provided.' && s !== 'No image provided.')
      .join('   ·   ')
  const axes = deriveAxes(results, activeTab)
  const colour = verdictColor(score)

  const counts = results
    ? {
        supported: results.supported_count ?? results.text_analysis?.supported_count ?? 0,
        refuted: results.refuted_count ?? results.text_analysis?.refuted_count ?? 0,
        nei: results.nei_count ?? results.text_analysis?.nei_count ?? 0,
      }
    : null

  return (
    <>
      {/* ══════════ LEFT: INPUTS ══════════ */}
      <div className="panel-container" ref={panelRef}>
        <div className="glass-panel" style={{ padding: '32px' }}>
          {activeTab === 'text' && (
            <>
              <h1 className="section-title">Text Dossier</h1>
              <p className="section-desc">
                Extract verifiable claims, run named entity recognition, and cross-reference
                against the Knowledge Graph using NLI entailment.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <textarea
                  className="glass-input"
                  placeholder="Paste subject text, transcript, or article..."
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                />
                <div>
                  <div className="slider-row">
                    <span>Max claims</span>
                    <span className="font-mono">{maxClaims}</span>
                  </div>
                  <input
                    type="range"
                    min="3"
                    max="30"
                    value={maxClaims}
                    onChange={(e) => setMaxClaims(Number(e.target.value))}
                    className="range"
                  />
                </div>
              </div>
            </>
          )}

          {activeTab === 'image' && (
            <>
              <h1 className="section-title">Visual Evidence</h1>
              <p className="section-desc">
                Detect synthetic generation using EfficientNet-B0 inference, FFT frequency
                anomaly analysis, and Grad-CAM saliency.
              </p>
              <div className="file-upload-zone">
                <input type="file" accept="image/*" onChange={handleImageChange} />
                {imagePreview ? (
                  <img
                    src={imagePreview}
                    alt="Preview"
                    style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '4px' }}
                  />
                ) : (
                  <>
                    <Upload
                      size={32}
                      color="var(--text-secondary)"
                      style={{ margin: '0 auto 16px' }}
                    />
                    <div>DRAG &amp; DROP IMAGE FILE</div>
                    <div
                      style={{
                        fontSize: '12px',
                        color: 'var(--text-secondary)',
                        marginTop: '8px',
                      }}
                    >
                      JPG, PNG, WEBP
                    </div>
                  </>
                )}
              </div>
            </>
          )}

          {activeTab === 'fusion' && (
            <>
              <h1 className="section-title">Multimodal Fusion</h1>
              <p className="section-desc">
                Confidence-weighted fusion of textual claim verification and visual deepfake
                detection into a unified credibility matrix.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <textarea
                  className="glass-input"
                  placeholder="Paste subject text (optional)..."
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  style={{ minHeight: '110px' }}
                />
                <div className="file-upload-zone" style={{ padding: '24px' }}>
                  <input type="file" accept="image/*" onChange={handleImageChange} />
                  {imagePreview ? (
                    <img
                      src={imagePreview}
                      alt="Preview"
                      style={{ maxWidth: '100%', maxHeight: '100px', borderRadius: '4px' }}
                    />
                  ) : (
                    <div>UPLOAD IMAGE EVIDENCE (OPTIONAL)</div>
                  )}
                </div>
                <div>
                  <div className="slider-row">
                    <span>Source reputation prior</span>
                    <span className="font-mono">{sourceRep.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={sourceRep}
                    onChange={(e) => setSourceRep(Number(e.target.value))}
                    className="range"
                  />
                </div>
              </div>
            </>
          )}

          <button
            className="glass-button"
            onClick={onRun}
            disabled={disabled}
            style={{
              opacity: disabled ? 0.4 : 1,
              width: '100%',
              marginTop: '28px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
            }}
          >
            {loading && <Loader2 size={14} className="spin" />}
            {loading
              ? `Analyzing — ${elapsed.toFixed(1)}s`
              : activeTab === 'text'
                ? 'Run Analysis'
                : activeTab === 'image'
                  ? 'Scan Image'
                  : 'Synthesize Data'}
          </button>

          {loading && (
            <p className="hint-text">
              First run loads model weights — this can take a few minutes.
            </p>
          )}

          {error && (
            <div className="error-box">
              <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
              <div>
                <div className="error-title">Analysis failed</div>
                <div className="error-detail">{error}</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ══════════ RIGHT: RESULTS ══════════ */}
      <div className="results-panel">
        {!results && !loading && (
          <div className="glass-panel empty-state">
            <Activity size={26} color="var(--text-muted)" />
            <p>
              Awaiting analysis.
              <br />
              Submit evidence to generate a truth report.
            </p>
          </div>
        )}

        {loading && (
          <div className="glass-panel empty-state">
            <Loader2 size={26} className="spin" color="var(--text-secondary)" />
            <p>
              Running pipeline…
              <br />
              <span className="font-mono" style={{ fontSize: 12 }}>
                {elapsed.toFixed(1)}s elapsed
              </span>
            </p>
          </div>
        )}

        {results && (
          <div
            ref={resultsRef}
            className="glass-panel"
            style={{
              flex: 1,
              padding: '32px',
              display: 'flex',
              flexDirection: 'column',
              gap: '28px',
              overflowY: 'auto',
            }}
          >
            <div className="reveal" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Activity size={20} />
              <h2 style={{ fontSize: '22px', fontWeight: 300 }}>Analysis Report</h2>
              {results.risk_level && (
                <span className="risk-chip" style={{ borderColor: colour, color: colour }}>
                  RISK {results.risk_level}
                </span>
              )}
            </div>

            {/* 1 — glowing truth percentage */}
            <div className="reveal">
              <ScoreRing score={score} />
            </div>

            {/* 2 — truth efficiency graph */}
            {axes.length > 0 && (
              <div className="reveal">
                <h3 className="result-heading">Truth Efficiency Graph</h3>
                <TruthGraph axes={axes} score={score} />
              </div>
            )}

            {/* 3 — claim tally */}
            {counts && counts.supported + counts.refuted + counts.nei > 0 && (
              <div className="reveal metric-grid">
                <div className="metric-cell">
                  <div className="font-mono metric-val" style={{ color: '#4ade80' }}>
                    {counts.supported}
                  </div>
                  <div className="metric-lbl">Supported</div>
                </div>
                <div className="metric-cell">
                  <div className="font-mono metric-val" style={{ color: '#f87171' }}>
                    {counts.refuted}
                  </div>
                  <div className="metric-lbl">Refuted</div>
                </div>
                <div className="metric-cell">
                  <div className="font-mono metric-val" style={{ color: '#888' }}>
                    {counts.nei}
                  </div>
                  <div className="metric-lbl">Inconclusive</div>
                </div>
              </div>
            )}

            {/* 4 — image-specific metrics */}
            {results.is_fake_prob !== undefined && (
              <div className="reveal metric-grid">
                <div className="metric-cell">
                  <div className="font-mono metric-val">
                    {Math.round((results.is_fake_prob ?? 0) * 100)}%
                  </div>
                  <div className="metric-lbl">Fake Prob</div>
                </div>
                <div className="metric-cell">
                  <div className="font-mono metric-val">{results.num_faces ?? 0}</div>
                  <div className="metric-lbl">Faces</div>
                </div>
                <div className="metric-cell">
                  <div className="font-mono metric-val">
                    {(results.high_frequency_energy ?? 0).toFixed(3)}
                  </div>
                  <div className="metric-lbl">HF Energy</div>
                </div>
              </div>
            )}

            {/* 5 — fusion weights */}
            {results.fusion_weights && (
              <div className="reveal">
                <h3 className="result-heading">Adaptive Fusion Weights</h3>
                <div className="metric-grid">
                  {Object.entries(results.fusion_weights).map(([k, v]) => (
                    <div className="metric-cell" key={k}>
                      <div className="font-mono metric-val">{Math.round(v * 100)}%</div>
                      <div className="metric-lbl">{k}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 6 — explanation */}
            {explanation && (
              <div className="reveal explain-box" style={{ borderLeftColor: colour }}>
                <p>{explanation}</p>
              </div>
            )}

            {/* 6b — why it's false, and what's actually true */}
            {claims.length > 0 && (
              <div className="reveal">
                <Corrections claims={claims} />
              </div>
            )}

            {/* 7 — claims */}
            {claims.length > 0 && (
              <div className="reveal">
                <h3 className="result-heading">Extracted Claims ({claims.length})</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {claims.map((c, i) => {
                    const label = c.verification?.label ?? 'NOT_ENOUGH_INFO'
                    const Icon = LABEL_ICON[label] ?? HelpCircle
                    const lc = LABEL_COLOR[label] ?? '#888'
                    return (
                      <div key={i} className="claim-card" style={{ borderLeftColor: lc }}>
                        <div className="claim-head">
                          <Icon size={13} color={lc} style={{ flexShrink: 0, marginTop: 2 }} />
                          <span className="claim-text">{c.claim}</span>
                          <span className="font-mono claim-score" style={{ color: lc }}>
                            {Math.round((c.verification?.score ?? 0.5) * 100)}%
                          </span>
                        </div>
                        {c.verification?.reason && (
                          <div className="claim-meta">{c.verification.reason}</div>
                        )}
                        {c.verification?.best_fact && (
                          <div className="claim-meta">
                            Evidence:&nbsp;
                            <span style={{ color: 'var(--text-primary)' }}>
                              {c.verification.best_fact}
                            </span>
                            {c.verification.source && (
                              <>
                                {' '}
                                {c.verification.source_url ? (
                                  <a
                                    href={c.verification.source_url}
                                    target="_blank"
                                    rel="noreferrer"
                                    style={{ color: lc, textDecoration: 'none' }}
                                  >
                                    [{c.verification.source}]
                                  </a>
                                ) : (
                                  <span style={{ color: 'var(--text-muted)' }}>
                                    [{c.verification.source}]
                                  </span>
                                )}
                              </>
                            )}
                          </div>
                        )}
                        {c.entities?.length > 0 && (
                          <div className="chip-row">
                            {c.entities.map((e, j) => (
                              <span key={j} className="chip font-mono">
                                {e.text} · {e.label}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* 8 — Grad-CAM */}
            {heatmap && (
              <div className="reveal">
                <h3 className="result-heading">Thermal Saliency Map</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  {imagePreview && (
                    <div>
                      <div className="metric-lbl" style={{ marginBottom: 8 }}>
                        Original
                      </div>
                      <img src={imagePreview} alt="Original" className="evidence-img" />
                    </div>
                  )}
                  <div>
                    <div className="metric-lbl" style={{ marginBottom: 8 }}>
                      Grad-CAM
                    </div>
                    <img
                      src={`data:image/png;base64,${heatmap}`}
                      alt="Grad-CAM saliency"
                      className="evidence-img"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  )
}
