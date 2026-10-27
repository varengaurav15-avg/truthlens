import { useEffect, useState, useCallback } from 'react'
import { Cpu, Server, Shield, Network, Brain, RefreshCw } from 'lucide-react'
import { health, API_BASE } from '../lib/api'

const POLL_MS = 8000

export default function Sidebar() {
  const [state, setState] = useState({ status: 'connecting', models: {}, latency: null })

  const probe = useCallback(async () => {
    const t0 = performance.now()
    try {
      const data = await health()
      setState({
        status: 'online',
        models: data?.models ?? {},
        classifier: data?.classifier ?? null,
        kg: data?.knowledge_graph ?? null,
        latency: Math.round(performance.now() - t0),
      })
    } catch {
      setState({ status: 'offline', models: {}, classifier: null, kg: null, latency: null })
    }
  }, [])

  useEffect(() => {
    probe()
    const id = setInterval(probe, POLL_MS)
    return () => clearInterval(id)
  }, [probe])

  const online = state.status === 'online'
  const modelRow = (key) => {
    if (!online) return { text: 'OFFLINE', color: 'var(--text-muted)' }
    const v = state.models?.[key]
    if (v === 'loaded') return { text: 'READY', color: '#4ade80' }
    if (v === 'not_loaded') return { text: 'IDLE', color: '#fbbf24' }
    return { text: 'UNKNOWN', color: 'var(--text-muted)' }
  }

  const vision = modelRow('efficientnet')
  const nli = modelRow('nli_model')
  const ner = modelRow('ner_pipeline')

  const statusColor = online ? '#4ade80' : state.status === 'connecting' ? '#fbbf24' : '#f87171'
  const statusText = online ? 'ONLINE' : state.status === 'connecting' ? 'PROBING' : 'OFFLINE'

  const Row = ({ icon: Icon, label, value, color }) => (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
        <Icon size={14} color="var(--text-secondary)" /> {label}
      </div>
      <div className="font-mono" style={{ fontSize: '11px', color: color ?? 'var(--text-primary)' }}>
        {value}
      </div>
    </div>
  )

  return (
    <div
      className="glass-panel sidebar-container"
      style={{
        width: '280px',
        padding: '24px',
        display: 'flex',
        flexDirection: 'column',
        gap: '28px',
      }}
    >
      <div>
        <div className="sidebar-heading">
          <span>SYSTEM STATUS</span>
          <button onClick={probe} className="icon-btn" title="Re-probe backend">
            <RefreshCw size={11} />
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
              <Server size={14} color="var(--text-secondary)" /> Backend API
            </div>
            <div
              className="font-mono"
              style={{
                color: statusColor,
                fontSize: '11px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              {online && <div className="pulse-dot" />}
              {statusText}
            </div>
          </div>

          <Row icon={Cpu} label="Vision Engine" value={vision.text} color={vision.color} />
          <Row icon={Brain} label="NLI Verifier" value={nli.text} color={nli.color} />
          <Row icon={Network} label="NER Pipeline" value={ner.text} color={ner.color} />
          <Row
            icon={Server}
            label="Round Trip"
            value={state.latency !== null ? `${state.latency}ms` : '—'}
            color="var(--text-secondary)"
          />
        </div>

        {online && state.classifier && (
          <div
            style={{
              marginTop: 14,
              padding: '10px 12px',
              borderRadius: 6,
              border: `1px solid ${state.classifier.trained ? 'rgba(74,222,128,.28)' : 'rgba(251,191,36,.32)'}`,
              background: state.classifier.trained
                ? 'rgba(74,222,128,.05)'
                : 'rgba(251,191,36,.06)',
            }}
          >
            <div
              className="font-mono"
              style={{
                fontSize: 9,
                letterSpacing: 1.4,
                color: state.classifier.trained ? '#4ade80' : '#fbbf24',
                marginBottom: 5,
              }}
            >
              {state.classifier.trained ? 'TRAINED CLASSIFIER' : 'UNTRAINED FALLBACK'}
            </div>
            <div style={{ fontSize: 10, lineHeight: 1.6, color: 'var(--text-muted)' }}>
              {state.classifier.trained
                ? 'ViT fine-tuned on real vs. deepfake faces'
                : 'CNN verdict suppressed — frequency analysis only'}
            </div>
          </div>
        )}

        {online && state.kg && (
          <div style={{ marginTop: 12 }}>
            <Row
              icon={Network}
              label="Knowledge Graph"
              value={`${state.kg.facts} facts`}
              color={state.kg.connected ? 'var(--text-primary)' : '#f87171'}
            />
            <div
              className="font-mono"
              style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 5, textAlign: 'right' }}
            >
              {state.kg.backend === 'neo4j' ? 'neo4j' : 'offline store'}
            </div>
          </div>
        )}
      </div>

      <div style={{ height: '1px', background: 'var(--border-glass)' }} />

      <div>
        <div className="sidebar-heading">
          <span>ENDPOINT</span>
        </div>
        <div
          className="font-mono"
          style={{ fontSize: '10px', color: 'var(--text-secondary)', wordBreak: 'break-all' }}
        >
          {API_BASE}
        </div>

        {!online && (
          <div className="offline-hint">
            <div style={{ marginBottom: 6, color: 'var(--text-primary)' }}>Backend unreachable</div>
            <code>python -m uvicorn src.api.main:app --port 8000</code>
          </div>
        )}
      </div>

      <div style={{ height: '1px', background: 'var(--border-glass)' }} />

      <div>
        <div className="sidebar-heading">
          <span>PIPELINE</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {[
            ['Claim extraction', 'Sentence segmentation'],
            ['Entity recognition', 'dslim/bert-base-NER'],
            ['Fact verification', 'nli-deberta-v3-small'],
            ['Deepfake CNN', 'EfficientNet-B0'],
            ['Saliency', 'Grad-CAM'],
            ['Fusion', 'Confidence-weighted'],
          ].map(([k, v]) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{k}</span>
              <span
                className="font-mono"
                style={{ fontSize: '9.5px', color: 'var(--text-muted)', textAlign: 'right' }}
              >
                {v}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div
        style={{
          marginTop: 'auto',
          padding: '16px',
          border: '1px solid var(--border-glass)',
          borderRadius: '8px',
          background: 'rgba(255,255,255,0.02)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <Shield size={16} color="var(--text-primary)" />
          <span style={{ fontSize: '12px', letterSpacing: '1px' }}>LOCAL INFERENCE</span>
        </div>
        <div className="font-mono" style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>
          NO TELEMETRY // ON-DEVICE
        </div>
      </div>
    </div>
  )
}
