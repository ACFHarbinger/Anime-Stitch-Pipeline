import React, { useEffect, useRef } from 'react';

export default function Viewfinder2D() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let mouseX = window.innerWidth / 2;
    let mouseY = window.innerHeight / 2;

    const handleMouseMove = (e: MouseEvent) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
    };
    window.addEventListener('mousemove', handleMouseMove);

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', resize);
    resize();

    let time = 0;

    const draw = () => {
      time += 0.01;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw grid
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.05)';
      ctx.lineWidth = 1;
      const gridSize = 100;
      for (let x = (time * 10) % gridSize; x < canvas.width; x += gridSize) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = (time * 10) % gridSize; y < canvas.height; y += gridSize) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }

      // Draw crosshair tracking mouse
      const crossSize = 30;
      ctx.strokeStyle = 'rgba(255, 0, 85, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(mouseX - crossSize, mouseY);
      ctx.lineTo(mouseX + crossSize, mouseY);
      ctx.moveTo(mouseX, mouseY - crossSize);
      ctx.lineTo(mouseX, mouseY + crossSize);
      ctx.stroke();

      // Outer targeting brackets
      const bracketSize = 20;
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.6)';
      const drawBracket = (x: number, y: number, mx: number, my: number) => {
        ctx.beginPath();
        ctx.moveTo(x + mx * bracketSize, y);
        ctx.lineTo(x, y);
        ctx.lineTo(x, y + my * bracketSize);
        ctx.stroke();
      };
      
      const gap = 60;
      drawBracket(mouseX - gap, mouseY - gap, 1, 1);
      drawBracket(mouseX + gap, mouseY - gap, -1, 1);
      drawBracket(mouseX - gap, mouseY + gap, 1, -1);
      drawBracket(mouseX + gap, mouseY + gap, -1, -1);

      // Random matching points
      ctx.fillStyle = 'rgba(0, 240, 255, 0.8)';
      for (let i = 0; i < 5; i++) {
        const px = mouseX + Math.sin(time * 2 + i) * 150;
        const py = mouseY + Math.cos(time * 3 + i) * 100;
        ctx.beginPath();
        ctx.arc(px, py, 2, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.beginPath();
        ctx.moveTo(mouseX, mouseY);
        ctx.lineTo(px, py);
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.1)';
        ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas 
      ref={canvasRef} 
      className="absolute inset-0 z-10 pointer-events-none"
      style={{ mixBlendMode: 'screen' }}
    />
  );
}
