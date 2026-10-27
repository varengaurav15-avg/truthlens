import { useEffect, useRef } from 'react'
import { XCircle, CheckCircle2, ExternalLink, ShieldAlert, HelpCircle } from 'lucide-react'
import gsap from 'gsap'

/**
 * Correction report.
 *
 * Answers the two questions a reader actually has when something is flagged:
 *   1. WHY is this false?
 *   2. What is the truth instead?
 *
 * Every correction is drawn from retrieved evidence and carries a source link.
 * Nothing here is generated free-hand — if no evidence supports a correction,
 * the row says so rather than inventing one.
 */
export default function Corrections({ claims = [] }) {
  const ref = useRef(null)

  const refuted = claims.filter((c) => c.verification?.label === 'REFUTED')
  const unverified = claims.filter((c) => c.verification?.label === 'NOT_ENOUGH_INFO')

  useEffect(() => {
    if (!ref.current || refuted.length === 0) return
    const items = ref.current.querySelectorAll('.corr-item')
    const tl = gsap.timeline()
    tl.fromTo(
      items,
      { opacity: 0, x: -14 },
      { opacity: 1, x: 0, duration: 0.6, ease: 'power3.out', stagger: 0.11 },
    )
    return () => tl.kill()
  }, [claims, refuted.length])

  if (refuted.length === 0 && unverified.length === 0) return null

  return (
    <div ref={ref}>
      <h3 className="result-heading">Correction Report</h3>

      {refuted.length > 0 && (
        <div className="corr-banner">
          <ShieldAlert size={16} color="#f87171" style={{ flexShrink: 0 }} />
          <span>
            <strong style={{ color: '#f87171' }}>
              {refuted.length} false {refuted.length === 1 ? 'statement' : 'statements'}
            </strong>{' '}
            detected in this text. Corrections below are drawn from the cited evidence.
          </span>
        </div>
      )}

      {refuted.map((c, i) => {
        const v = c.verification || {}
        const truth = v.correction || v.best_fact
        return (
          <div key={i} className="corr-item">
            {/* the false statement */}
            <div className="corr-row">
              <XCircle size={14} color="#f87171" style={{ flexShrink: 0, marginTop: 2 }} />
              <div style={{ minWidth: 0 }}>
                <div className="corr-tag" style={{ color: '#f87171' }}>
                  Claimed
                </div>
                <div className="corr-claim">{v.restated_claim || c.claim}</div>
              </div>
            </div>

            {/* why it is false */}
            {v.reason && (
              <div className="corr-why">
                <span className="corr-tag">Why it&apos;s false</span>
                <p>{v.reason}</p>
              </div>
            )}

            {/* the truth */}
            {truth && (
              <div className="corr-row corr-truth">
                <CheckCircle2 size={14} color="#4ade80" style={{ flexShrink: 0, marginTop: 2 }} />
                <div style={{ minWidth: 0 }}>
                  <div className="corr-tag" style={{ color: '#4ade80' }}>
                    Actually
                  </div>
                  <div className="corr-fact">{truth}</div>
                  {v.source && (
                    <div className="corr-src">
                      {v.source_url ? (
                        <a href={v.source_url} target="_blank" rel="noreferrer">
                          {v.source}
                          <ExternalLink size={9} style={{ marginLeft: 4 }} />
                        </a>
                      ) : (
                        <span>{v.source}</span>
                      )}
                      {v.judge && v.judge !== 'nli' && (
                        <span className="corr-judge">adjudicated by {v.judge}</span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )
      })}

      {unverified.length > 0 && (
        <div className="corr-unverified">
          <HelpCircle size={13} color="var(--text-muted)" style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <div className="corr-tag">Could not verify</div>
            <p>
              {unverified.length}{' '}
              {unverified.length === 1 ? 'statement has' : 'statements have'} no supporting or
              contradicting evidence in the sources checked. Unverified is not the same as false.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
