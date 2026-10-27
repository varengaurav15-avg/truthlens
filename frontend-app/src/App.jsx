import { useState, Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { Loader } from '@react-three/drei'
import { Search } from 'lucide-react'
import Scene from './components/Scene'
import DetectorPanel from './components/DetectorPanel'
import Sidebar from './components/Sidebar'
import Ticker from './components/Ticker'

export default function App() {
  const [activeTab, setActiveTab] = useState('text')

  return (
    <>
      {/* 3D Background */}
      <Canvas 
        shadows 
        camera={{ position: [0, 0, 10], fov: 45 }}
        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 0 }}
      >
        <Suspense fallback={null}>
          <Scene />
        </Suspense>
      </Canvas>
      <Loader />

      {/* Glassmorphic UI Overlay */}
      <div className="ui-layer">
        <header className="glass-panel" style={{ borderTop: 'none', borderLeft: 'none', borderRight: 'none', borderRadius: 0 }}>
          <div className="logo">
            <div className="logo-icon"><Search size={16} /></div>
            TruthLens
          </div>
          <nav className="nav-links">
            <div className={`nav-link ${activeTab === 'text' ? 'active' : ''}`} onClick={() => setActiveTab('text')}>
              Text Analysis
            </div>
            <div className={`nav-link ${activeTab === 'image' ? 'active' : ''}`} onClick={() => setActiveTab('image')}>
              Image Deepfake
            </div>
            <div className={`nav-link ${activeTab === 'fusion' ? 'active' : ''}`} onClick={() => setActiveTab('fusion')}>
              Multimodal Fusion
            </div>
          </nav>
        </header>

        <main className="main-content" style={{ paddingBottom: '80px', display: 'flex', gap: '32px' }}>
          <Sidebar />
          <div style={{ flex: 1, display: 'flex', gap: '32px', overflow: 'hidden' }}>
            <DetectorPanel activeTab={activeTab} />
          </div>
        </main>
        
        <Ticker />
      </div>
    </>
  )
}
