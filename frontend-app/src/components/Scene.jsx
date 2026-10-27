import { useRef, useMemo, useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Environment, Sparkles, Sphere } from '@react-three/drei'
import * as THREE from 'three'

function ParallaxGroup({ children }) {
  const groupRef = useRef()
  const { mouse } = useThree()
  
  useFrame((state, delta) => {
    if(groupRef.current) {
      // Extremely smooth, heavy, luxurious parallax
      groupRef.current.position.x = THREE.MathUtils.lerp(groupRef.current.position.x, mouse.x * 1.2, 1.5 * delta)
      groupRef.current.position.y = THREE.MathUtils.lerp(groupRef.current.position.y, mouse.y * 1.2, 1.5 * delta)
      groupRef.current.rotation.y = THREE.MathUtils.lerp(groupRef.current.rotation.y, mouse.x * 0.05, 1.5 * delta)
      groupRef.current.rotation.x = THREE.MathUtils.lerp(groupRef.current.rotation.x, -mouse.y * 0.05, 1.5 * delta)
    }
  })
  
  return <group ref={groupRef}>{children}</group>
}

export default function Scene() {
  const lotusCoreRef = useRef()
  const orbitingRef = useRef()
  const { viewport } = useThree()

  // Generate a true 3D Petal Geometry using Bezier curves (Optimized Polycount)
  const petalGeometry = useMemo(() => {
    const shape = new THREE.Shape()
    shape.moveTo(0, 0)
    // Draw organic petal curves
    shape.bezierCurveTo(1.5, 1.5, 0.5, 4, 0, 5)
    shape.bezierCurveTo(-0.5, 4, -1.5, 1.5, 0, 0)
    
    const geom = new THREE.ExtrudeGeometry(shape, { 
      depth: 0.1, // Thinner, more delicate
      bevelEnabled: true, 
      bevelThickness: 0.02, 
      bevelSize: 0.02, 
      bevelSegments: 2,
      curveSegments: 8
    })
    geom.center()
    return geom
  }, [])

  // Procedural Matrices for InstancedMesh
  const { coreMatrices, orbitingMatrices } = useMemo(() => {
    const dummy = new THREE.Object3D()
    
    // Core Lotus Layers - tuned to look like a blooming lotus (Optimized count for 60fps)
    const layers = [
      { count: 6, radius: 0.2, tilt: Math.PI / 12, scale: 0.6, height: 0.5 },
      { count: 10, radius: 0.6, tilt: Math.PI / 6, scale: 0.8, height: 0 },
      { count: 14, radius: 1.2, tilt: Math.PI / 4, scale: 1.0, height: -0.5 },
      { count: 20, radius: 1.8, tilt: Math.PI / 3, scale: 1.1, height: -1.0 }
    ]
    
    const core = []
    layers.forEach(layer => {
      for(let i = 0; i < layer.count; i++) {
        const angle = (i / layer.count) * Math.PI * 2
        dummy.position.set(0, layer.height, 0)
        // Rotate to face outward radially
        dummy.rotation.set(0, angle, 0)
        // Push outward from center
        dummy.translateZ(layer.radius)
        // Tilt the petal backward to open the bloom
        dummy.rotateX(layer.tilt)
        dummy.scale.setScalar(layer.scale)
        dummy.updateMatrix()
        core.push(dummy.matrix.clone())
      }
    })

    // Orbiting Petals (Small, crystally, reduced count for performance)
    const orbit = []
    for(let i = 0; i < 40; i++) {
      const angle = (i / 40) * Math.PI * 2 * 3 // Spiral 3 times
      const r = 2.0 + Math.random() * 2.5
      const h = (Math.random() - 0.5) * 6
      dummy.position.set(0, h, 0)
      dummy.rotation.set(Math.random() * Math.PI, angle, Math.random() * Math.PI)
      dummy.translateZ(r)
      // Small scale for delicate, beautiful floating crystals
      dummy.scale.setScalar(0.15 + Math.random() * 0.1)
      dummy.updateMatrix()
      orbit.push(dummy.matrix.clone())
    }

    return { coreMatrices: core, orbitingMatrices: orbit }
  }, [])

  // Apply matrices on mount
  useEffect(() => {
    if (lotusCoreRef.current) {
      coreMatrices.forEach((matrix, i) => lotusCoreRef.current.setMatrixAt(i, matrix))
      lotusCoreRef.current.instanceMatrix.needsUpdate = true
    }
    if (orbitingRef.current) {
      orbitingMatrices.forEach((matrix, i) => orbitingRef.current.setMatrixAt(i, matrix))
      orbitingRef.current.instanceMatrix.needsUpdate = true
    }
  }, [coreMatrices, orbitingMatrices])

  useFrame((state, delta) => {
    const t = state.clock.getElapsedTime()
    if(lotusCoreRef.current) {
      // Gentle, majestic spin and bob
      lotusCoreRef.current.rotation.y = t * 0.05
      lotusCoreRef.current.position.y = Math.sin(t * 0.5) * 0.15
    }
    if(orbitingRef.current) {
      // Very slow, fluid orbit spin
      orbitingRef.current.rotation.y = t * 0.1
      orbitingRef.current.rotation.z = Math.sin(t * 0.1) * 0.05
      
      // Extremely subtle petal independent spin
      orbitingRef.current.children.forEach((child, i) => {
        child.children[0].rotation.z += delta * (0.1 + i * 0.01)
      })
    }
  })

  // Pure Glass Material (Restored heavy transmission for the beautiful crystal look)
  const glassMaterial = (
    <meshPhysicalMaterial 
      transmission={1} 
      opacity={1} 
      metalness={0.1} 
      roughness={0.02} 
      ior={1.5} 
      thickness={1.5} 
      envMapIntensity={2.5} 
      color="#ffffff" 
      side={THREE.DoubleSide}
    />
  )

  return (
    <>
      <color attach="background" args={['#000000']} />
      <fog attach="fog" args={['#000000', 5, 25]} />

      <ambientLight intensity={0.5} />
      <spotLight position={[10, 10, 10]} intensity={8} angle={0.6} penumbra={1} color="#ffffff" />
      <pointLight position={[-10, -10, -10]} intensity={4} color="#555555" />
      
      <ParallaxGroup>
        
        <group position={[viewport.width / 4, -0.5, 0]}>
          
          {/* Intense Golden Glowing Core */}
          <pointLight position={[0, 0.2, 0]} intensity={200} distance={15} color="#ffaa00" decay={1.5} />
          <Sphere args={[0.2, 16, 16]} position={[0, 0.2, 0]}>
            <meshBasicMaterial color="#ffaa00" toneMapped={false} />
          </Sphere>

          {/* Lotus Crystal Stem */}
          <mesh position={[0, -2.5, 0]}>
            <cylinderGeometry args={[0.08, 0.04, 5, 16]} />
            <meshPhysicalMaterial transmission={1} opacity={1} roughness={0.05} ior={1.5} color="#ffffff" />
          </mesh>

          {/* Core Lotus InstancedMesh using True Petal Geometry */}
          <instancedMesh ref={lotusCoreRef} args={[petalGeometry, null, coreMatrices.length]}>
            {glassMaterial}
          </instancedMesh>

          {/* Orbiting Petals InstancedMesh */}
          <instancedMesh ref={orbitingRef} args={[petalGeometry, null, orbitingMatrices.length]}>
            <meshPhysicalMaterial 
              transmission={0.9} 
              opacity={1} 
              metalness={0.1} 
              roughness={0.1} 
              ior={1.5} 
              thickness={0.2} 
              envMapIntensity={2.5} 
              color="#ffffff" 
            />
          </instancedMesh>

          {/* Magical Particles (Restored) */}
          <Sparkles count={100} scale={15} size={2} speed={0.2} opacity={0.6} color="#ffffff" />
        </group>
        
      </ParallaxGroup>

      {/* High contrast studio for crazy reflections */}
      <Environment preset="studio" environmentIntensity={1.5} />
    </>
  )
}
