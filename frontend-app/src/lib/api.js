/**
 * TruthLens API client.
 *
 * Talks to the FastAPI backend in src/api/. Every response shape below is
 * verified against the actual route handlers — do not guess field names here.
 *
 *   GET  /health                 -> { status, models: { efficientnet, nli_model, ner_pipeline } }
 *   POST /text/analyze           -> { credibility_score, verdict, claims[], explanation,
 *                                     supported_count, refuted_count, nei_count }
 *   POST /vision/analyze_image   -> { credibility_score, is_fake_prob, cnn_fake_prob,
 *                                     frequency_fake_score, high_frequency_energy,
 *                                     num_faces, face_boxes[], explanation,
 *                                     gradcam_heatmap_b64 }
 *   POST /fusion/analyze         -> { credibility_score, verdict, risk_level, confidence,
 *                                     fusion_weights{}, signals{}, text_analysis{},
 *                                     vision_analysis{} }
 */

/**
 * Resolve the backend origin.
 *
 * Defensive on purpose: PowerShell's `Out-File -Encoding utf8` writes a UTF-8
 * BOM (EF BB BF), which corrupts the FIRST key of a .env file — it parses as
 * "﻿VITE_API_URL" instead of "VITE_API_URL" and Vite silently ignores it.
 * We scan every key and strip BOM/whitespace so a BOM-prefixed variable still
 * resolves rather than failing quietly and falling back to the wrong port.
 */
const BOM = /^﻿/

function resolveApiBase() {
  const env = import.meta.env ?? {}
  let value = env.VITE_API_URL

  if (!value) {
    for (const [key, val] of Object.entries(env)) {
      if (key.replace(BOM, '').trim() === 'VITE_API_URL') {
        value = val
        break
      }
    }
  }

  return String(value ?? 'http://127.0.0.1:8000')
    .replace(BOM, '')
    .trim()
    .replace(/\/+$/, '')
}

export const API_BASE = resolveApiBase()

/** First call after boot can be slow — transformers may download weights. */
const LONG_TIMEOUT = 240_000
const SHORT_TIMEOUT = 6_000

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request(path, options = {}, timeoutMs = LONG_TIMEOUT) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  let res
  try {
    res = await fetch(`${API_BASE}${path}`, { ...options, signal: controller.signal })
  } catch (err) {
    clearTimeout(timer)
    if (err?.name === 'AbortError') {
      throw new ApiError(
        'Request timed out. The first analysis downloads model weights and can take several minutes.',
      )
    }
    throw new ApiError(
      `Cannot reach the backend at ${API_BASE}. Start it with:  python -m uvicorn src.api.main:app --port 8000`,
    )
  }
  clearTimeout(timer)

  const contentType = res.headers.get('content-type') || ''
  const isJson = contentType.includes('application/json')
  const body = isJson ? await res.json().catch(() => null) : await res.text().catch(() => '')

  if (!res.ok) {
    let detail
    if (body && typeof body === 'object') {
      // FastAPI returns { detail: "..." } or { detail: [{ msg, loc }] } for 422
      if (Array.isArray(body.detail)) {
        detail = body.detail.map((d) => d.msg || JSON.stringify(d)).join('; ')
      } else {
        detail = body.detail || JSON.stringify(body)
      }
    } else {
      detail = String(body || '').slice(0, 240) || res.statusText
    }
    throw new ApiError(detail, res.status)
  }

  return body
}

export function health() {
  return request('/health', { method: 'GET' }, SHORT_TIMEOUT)
}

export function analyzeText(text, maxClaims = 10) {
  return request('/text/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, max_claims: maxClaims }),
  })
}

export function analyzeImage(file) {
  const fd = new FormData()
  fd.append('file', file)
  return request('/vision/analyze_image', { method: 'POST', body: fd })
}

export function analyzeFusion({ text = '', file = null, sourceReputation = 0.5 }) {
  const fd = new FormData()
  // Backend declares text as Form("") — always send the field, even if blank.
  fd.append('text', text || '')
  fd.append('source_reputation', String(sourceReputation))
  if (file) fd.append('file', file)
  return request('/fusion/analyze', { method: 'POST', body: fd })
}

/* ────────────────────────────────────────────────────────────────
   Verdict helpers — thresholds mirror src/fusion_module/scoring.py
   ──────────────────────────────────────────────────────────────── */

export function verdictOf(score) {
  if (score >= 0.7) return 'CREDIBLE'
  if (score >= 0.4) return 'SUSPICIOUS'
  return 'LIKELY FAKE'
}

export function verdictColor(score) {
  if (score >= 0.7) return '#4ade80'
  if (score >= 0.4) return '#fbbf24'
  return '#f87171'
}

/**
 * Derive Truth Efficiency axes from whatever the backend actually returned.
 * Every value is real — nothing is fabricated to fill the chart.
 */
export function deriveAxes(r, tab) {
  if (!r) return []

  const claimStats = (src) => {
    const s = src?.supported_count ?? 0
    const f = src?.refuted_count ?? 0
    const n = src?.nei_count ?? 0
    const total = s + f + n
    return { s, f, n, total }
  }

  if (tab === 'text') {
    const { s, f, total } = claimStats(r)
    // Labels stay short — the radar wraps at one space, so two-word maximum.
    return [
      { label: 'Credibility', value: r.credibility_score ?? 0 },
      { label: 'KG Support', value: total ? s / total : 0 },
      { label: 'Coverage', value: total ? (s + f) / total : 0 },
      { label: 'Claims', value: Math.min(1, (r.claims?.length ?? 0) / 10) },
      {
        label: 'Entities',
        value: Math.min(
          1,
          (r.claims?.reduce((a, c) => a + (c.entities?.length ?? 0), 0) ?? 0) /
            Math.max(1, (r.claims?.length ?? 0) * 3),
        ),
      },
    ]
  }

  if (tab === 'image') {
    return [
      { label: 'Authenticity', value: r.credibility_score ?? 0 },
      { label: 'CNN', value: 1 - (r.cnn_fake_prob ?? 0) },
      { label: 'Frequency', value: 1 - (r.frequency_fake_score ?? 0) },
      { label: 'Spectral', value: 1 - Math.min(1, r.high_frequency_energy ?? 0) },
      { label: 'Faces', value: Math.min(1, (r.num_faces ?? 0) / 3) },
    ]
  }

  // fusion
  const sig = r.signals ?? {}
  const { s, f, total } = claimStats(r.text_analysis)
  return [
    { label: 'Text', value: sig.text_credibility ?? 0 },
    { label: 'Visual', value: sig.visual_credibility ?? 0 },
    { label: 'Source', value: sig.source_reputation ?? 0 },
    { label: 'KG Support', value: total ? s / total : 0 },
    { label: 'Confidence', value: r.confidence ?? 0 },
    { label: 'Coverage', value: total ? (s + f) / total : 0 },
  ]
}
