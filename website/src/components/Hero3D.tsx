import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Float, PerspectiveCamera, MeshTransmissionMaterial, Sparkles } from '@react-three/drei';
import * as THREE from 'three';

function OpticalPrism() {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.15;
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={1}>
      <mesh ref={meshRef} position={[0, 0, 0]}>
        <octahedronGeometry args={[2.5, 0]} />
        <MeshTransmissionMaterial
          backside
          backsideThickness={1}
          thickness={2.5}
          chromaticAberration={0.5}
          ior={1.3}
          transmission={1}
          opacity={1}
          transparent
          clearcoat={1}
          clearcoatRoughness={0}
          roughness={0.05}
          color="#e2e8f0"
        />
        {/* Wireframe overlay to emphasize the technical aspect */}
        <mesh>
          <octahedronGeometry args={[2.501, 0]} />
          <meshBasicMaterial color="#00F0FF" wireframe transparent opacity={0.15} />
        </mesh>
      </mesh>
    </Float>
  );
}

export default function Hero3D() {
  return (
    <div className="w-full h-full absolute inset-0 z-20 pointer-events-none">
      <Canvas>
        <PerspectiveCamera makeDefault position={[0, 0, 8]} fov={45} />
        
        {/* Dramatic Lab Lighting */}
        <ambientLight intensity={0.2} color="#050505" />
        <spotLight position={[10, 10, 10]} angle={0.2} penumbra={1} intensity={2} color="#00F0FF" castShadow />
        <spotLight position={[-10, -10, -10]} angle={0.2} penumbra={1} intensity={1.5} color="#FF0055" />
        <pointLight position={[0, 0, 0]} intensity={0.5} color="#ffffff" />
        
        <Sparkles count={150} scale={12} size={1.5} speed={0.2} opacity={0.6} color="#00F0FF" />
        <Sparkles count={50} scale={12} size={2} speed={0.4} opacity={0.4} color="#FF0055" />

        <OpticalPrism />
        
        <OrbitControls 
          enableZoom={false} 
          enablePan={false} 
          enableRotate={false}
        />
      </Canvas>
    </div>
  );
}
