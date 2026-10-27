import { useEffect, useRef, useState } from 'react'
import gsap from 'gsap'
import { verdictOf, verdictColor } from '../lib/api'

/* ══════════════════════════════════════════════════════════════
   ScoreRing — glowing truth percentage inside an animated ring
   ══════════════════════════════════════════════════════════════ */
export function ScoreRing({ score = 0, label = 'Truth Efficiency', size = 260 }) {
  const [display, setDisplay] = useState(0)
  const arcRef = useRef(null)
  const glowRef = useRef(null)

  const R = size / 2 - 26
  const C = 2 * Math.PI * R
  const colour = verdictColor(score)
  const verdict = verdictOf(score)

  useEffect(() => {
    const counter = { v: 0 }
    const tl = gsap.timeline()

    tl.to(counter, {
      v: score * 100,
      duration: 1.6,
      ease: 'power3.out',
      onUpdate: () => setDisplay(counter.v),
    })

    if (arcRef.current) {
      tl.fromTo(
        arcRef.current,
        { strokeDashoffset: C },
        { strokeDashoffset: C * (1 - score), duration: 1.6, ease: 'power3.out' },
        0,
      )
    }
    if (glowRef.current) {
      tl.fromTo(
        glowRef.current,
        { opacity: 0, scale: 0.94 },
        { opacity: 1, scale: 1, duration: 1.2, ease: 'power2.out' },
        0.15,
      )
    }
    return () => tl.kill()
  }, [score, C])

  return (
    <div
      style={{
        position: 'relative',
        width: size,
        height: size,
        margin: '0 auto',
        display: 'grid',
        placeItems: 'center',
      }}
    >
      {/* ambient bloom behind the ring */}
      <div
        style={{
          position: 'absolute',
          inset: '18%',
          borderRadius: '50%',
          background: `radial-gradient(circle, ${colour}22 0%, transparent 68%)`,
          filter: 'blur(22px)',
          pointerEvents: 'none',
        }}
      />

      <svg
        width={size}
        height={size}
        style={{ position: 'absolute', inset: 0, transform: 'rotate(-90deg)' }}
      >
        <defs>
          <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor={colour} stopOpacity="0.35" />
            <stop offset="55%" stopColor={colour} stopOpacity="1" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0.85" />
          </linearGradient>
          <filter id="ringGlow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="5" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* tick marks */}
        {Array.from({ length: 60 }).map((_, i) => {
          const a = (i / 60) * Math.PI * 2
          const inner = R + 10
          const outer = R + (i % 5 === 0 ? 17 : 13)
          return (
            <line
              key={i}
              x1={size / 2 + Math.cos(a) * inner}
              y1={size / 2 + Math.sin(a) * inner}
              x2={size / 2 + Math.cos(a) * outer}
              y2={size / 2 + Math.sin(a) * outer}
              stroke="rgba(255,255,255,0.16)"
              strokeWidth={i % 5 === 0 ? 1 : 0.5}
            />
          )
        })}

        <circle
          cx={size / 2}
          cy={size / 2}
          r={R}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth="3"
        />
        <circle
          ref={arcRef}
          cx={size / 2}
          cy={size / 2}
          r={R}
          fill="none"
          stroke="url(#ringGrad)"
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={C}
          strokeDashoffset={C}
          filter="url(#ringGlow)"
        />
      </svg>

      <div ref={glowRef} style={{ position: 'relative', textAlign: 'center' }}>
        <div
          className="font-mono"
          style={{
            fontSize: size * 0.26,
            fontWeight: 200,
            lineHeight: 1,
            letterSpacing: '-0.04em',
            color: '#fff',
            textShadow: `0 0 28px ${colour}cc, 0 0 60px ${colour}55`,
          }}
        >
          {display.toFixed(1)}
          <span style={{ fontSize: size * 0.10, opacity: 0.5, marginLeft: 2 }}>%</span>
        </div>
        <div
          style={{
            fontSize: 10,
            letterSpacing: '0.32em',
            textTransform: 'uppercase',
            color: 'var(--text-secondary)',
            marginTop: 12,
          }}
        >
          {label}
        </div>
        <div
          className="font-mono"
          style={{
            fontSize: 11,
            letterSpacing: '0.22em',
            marginTop: 10,
            color: colour,
            textShadow: `0 0 14px ${colour}77`,
          }}
        >
          {verdict}
        </div>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   TruthGraph — radar over the real signals the backend returned
   ══════════════════════════════════════════════════════════════ */
export function TruthGraph({ axes = [], score = 0 }) {
  const polyRef = useRef(null)
  const dotsRef = useRef([])
  const colour = verdictColor(score)

  // Wider than tall: axis labels sit outside the polygon and need horizontal
  // room. A square viewBox clipped "ENTITY DENSITY" and "KG SUPPORT".
  const W = 340
  const H = 290
  const cx = W / 2
  const cy = H / 2 - 4
  const R = 74            // pulled in from 0.30*size to leave a label gutter
  const LABEL_GAP = 20
  const N = axes.length

  useEffect(() => {
    if (!N) return
    const tl = gsap.timeline()
    if (polyRef.current) {
      tl.fromTo(
        polyRef.current,
        { scale: 0.1, opacity: 0, transformOrigin: '50% 50%' },
        { scale: 1, opacity: 1, duration: 1.1, ease: 'power3.out' },
      )
    }
    const dots = dotsRef.current.filter(Boolean)
    if (dots.length) {
      tl.fromTo(
        dots,
        { scale: 0, transformOrigin: '50% 50%' },
        { scale: 1, duration: 0.5, ease: 'back.out(2.4)', stagger: 0.07 },
        0.5,
      )
    }
    return () => tl.kill()
  }, [axes, N])

  if (!N) return null

  const pt = (i, r) => {
    const a = -Math.PI / 2 + (i / N) * Math.PI * 2
    return [cx + Math.cos(a) * r, cy + Math.sin(a) * r]
  }

  const clamp = (v) => Math.max(0.02, Math.min(1, Number.isFinite(v) ? v : 0))
  const dataPts = axes.map((ax, i) => pt(i, R * clamp(ax.value)))
  const polyStr = dataPts.map((p) => p.map((n) => n.toFixed(1)).join(',')).join(' ')

  return (
    <div style={{ position: 'relative', width: '100%', maxWidth: 360, margin: '0 auto' }}>
      <svg
        width="100%"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ display: 'block', overflow: 'hidden' }}
      >
        <defs>
          <radialGradient id="radarFill">
            <stop offset="0%" stopColor={colour} stopOpacity="0.34" />
            <stop offset="100%" stopColor={colour} stopOpacity="0.06" />
          </radialGradient>
        </defs>

        {/* concentric grid */}
        {[0.25, 0.5, 0.75, 1].map((f, ri) => (
          <polygon
            key={ri}
            points={axes
              .map((_, i) => pt(i, R * f).map((n) => n.toFixed(1)).join(','))
              .join(' ')}
            fill="none"
            stroke="rgba(255,255,255,0.07)"
            strokeWidth="0.7"
          />
        ))}

        {/* spokes */}
        {axes.map((_, i) => {
          const [x, y] = pt(i, R)
          return (
            <line
              key={i}
              x1={cx}
              y1={cy}
              x2={x}
              y2={y}
              stroke="rgba(255,255,255,0.07)"
              strokeWidth="0.7"
            />
          )
        })}

        {/* data polygon */}
        <polygon
          ref={polyRef}
          points={polyStr}
          fill="url(#radarFill)"
          stroke={colour}
          strokeWidth="1.4"
          strokeLinejoin="round"
          style={{ filter: `drop-shadow(0 0 8px ${colour}66)` }}
        />

        {/* vertices */}
        {dataPts.map(([x, y], i) => (
          <circle
            key={i}
            ref={(el) => (dotsRef.current[i] = el)}
            cx={x}
            cy={y}
            r="3.2"
            fill="#0a0a0a"
            stroke={colour}
            strokeWidth="1.5"
          />
        ))}

        {/* axis labels + values — long names wrap onto a second line so they
            never run past the viewBox edge */}
        {axes.map((ax, i) => {
          const [lx, ly] = pt(i, R + LABEL_GAP)
          const a = -Math.PI / 2 + (i / N) * Math.PI * 2
          const cos = Math.cos(a)
          const sin = Math.sin(a)
          const anchor = Math.abs(cos) < 0.3 ? 'middle' : cos > 0 ? 'start' : 'end'

          const words = String(ax.label).split(' ')
          const lines = words.length > 1 ? [words[0], words.slice(1).join(' ')] : words

          // nudge upward when the label sits above centre so the value below
          // it doesn't collide with the polygon
          const yBase = ly + (sin < -0.3 ? -6 : sin > 0.3 ? 4 : 0)

          return (
            <g key={i}>
              {lines.map((line, li) => (
                <text
                  key={li}
                  x={lx}
                  y={yBase + li * 9}
                  textAnchor={anchor}
                  fill="rgba(255,255,255,0.55)"
                  fontSize="7.5"
                  fontFamily="var(--font-sans)"
                  letterSpacing="1.1"
                  style={{ textTransform: 'uppercase' }}
                >
                  {line}
                </text>
              ))}
              <text
                x={lx}
                y={yBase + lines.length * 9 + 2}
                textAnchor={anchor}
                fill={colour}
                fontSize="9.5"
                fontFamily="var(--font-mono)"
              >
                {Math.round(clamp(ax.value) * 100)}%
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

export default TruthGraph
