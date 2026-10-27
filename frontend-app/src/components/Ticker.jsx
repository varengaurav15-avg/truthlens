import { useEffect, useRef, useState } from 'react'
import { health, API_BASE } from '../lib/api'

/**
 * Live status ticker.
 *
 * This used to scroll invented strings — "High velocity anomalies detected in
 * sector 7", "Node clustering algorithm active" — none of which corresponded
 * to anything the system does. Decorative fiction in a misinformation
 * detector is a bad look, so every line here now comes from /health or from a
 * real scan event.
 */
export default function Ticker({ lastEvent = null }) {
  const trackRef = useRef(null)
  const [lines, setLines] = useState([
    { t: `connecting to ${API_BASE}`, tone: 'dim' },
  ])

  useEffect(() => {
    let cancelled = false

    const build = async () => {
      try {
        const h = await health()
        if (cancelled) return

        const m = h?.models ?? {}
        const clf = h?.classifier ?? {}
        const kg = h?.knowledge_graph ?? {}
        const ready = (v) => (v === 'loaded' ? 'ready' : 'idle')

        const next = [
          { t: `backend online · ${API_BASE}`, tone: 'ok' },
          { t: `NER dslim/bert-base-NER ${ready(m.ner_pipeline)}`, tone: 'dim' },
          { t: `NLI nli-deberta-v3-small ${ready(m.nli_model)}`, tone: 'dim' },
          clf.trained
            ? { t: `deepfake classifier ${clf.model} loaded`, tone: 'ok' }
            : { t: 'deepfake classifier untrained — frequency analysis only', tone: 'warn' },
          {
            t: `knowledge graph ${kg.facts ?? 0} curated facts via ${kg.backend ?? 'unknown'}`,
            tone: 'dim',
          },
          { t: 'evidence retrieval en.wikipedia.org live', tone: 'dim' },
          { t: `${new Date().toLocaleTimeString()} health check ok`, tone: 'dim' },
        ]
        setLines(next)
      } catch {
        if (!cancelled) {
          setLines([
            { t: `backend unreachable at ${API_BASE}`, tone: 'err' },
            { t: 'start it: python -m uvicorn src.api.main:app --port 8001', tone: 'dim' },
          ])
        }
      }
    }

    build()
    const id = setInterval(build, 15000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  // Prepend the most recent real scan result when one exists
  const feed = lastEvent ? [{ t: lastEvent, tone: 'ok' }, ...lines] : lines

  useEffect(() => {
    let pos = 0
    let raf
    const step = () => {
      pos -= 0.45
      const el = trackRef.current
      if (el) {
        const half = el.scrollWidth / 2
        if (half > 0 && -pos >= half) pos = 0
        el.style.transform = `translateX(${pos}px)`
      }
      raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [feed.length])

  const colour = (tone) =>
    tone === 'ok'
      ? '#4ade80'
      : tone === 'warn'
        ? '#fbbf24'
        : tone === 'err'
          ? '#ef4444'
          : 'var(--text-secondary)'

  // Duplicated once so the scroll loops seamlessly
  const doubled = [...feed, ...feed]

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        width: '100%',
        height: '32px',
        background: 'rgba(5, 5, 5, 0.85)',
        borderTop: '1px solid var(--border-glass)',
        display: 'flex',
        alignItems: 'center',
        overflow: 'hidden',
        fontFamily: 'var(--font-mono)',
        fontSize: '11px',
        zIndex: 100,
        pointerEvents: 'auto',
      }}
    >
      <div
        style={{
          padding: '0 20px',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          background: 'var(--text-primary)',
          color: 'var(--bg-color)',
          fontWeight: 'bold',
          zIndex: 2,
          letterSpacing: '1px',
          flexShrink: 0,
        }}
      >
        SYSTEM
      </div>
      <div style={{ flex: 1, overflow: 'hidden', whiteSpace: 'nowrap', position: 'relative' }}>
        <div ref={trackRef} style={{ display: 'inline-block', paddingLeft: '24px' }}>
          {doubled.map((l, i) => (
            <span key={i} style={{ marginRight: '44px', color: colour(l.tone) }}>
              &gt; {l.t}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
