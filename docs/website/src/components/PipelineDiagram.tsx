import { useEffect, useRef } from "react";
import "./PipelineDiagram.css";

/**
 * Hand-rolled <canvas> ASP pipeline stages (VGP-style, no three.js/d3).
 * Stages mirror anime_stitch_*.json time breakdown fields.
 */

interface Stage {
  key: string;
  label: string;
  weight: number;
}

const STAGES: Stage[] = [
  { key: "birefnet", label: "BiRefNet", weight: 0.13 },
  { key: "matching", label: "Match", weight: 0.11 },
  { key: "bundle_adjust", label: "BA", weight: 0.1 },
  { key: "ecc", label: "ECC", weight: 0.06 },
  { key: "render", label: "Render", weight: 0.29 },
  { key: "composite", label: "Composite", weight: 0.31 },
];

function ease(t: number): number {
  const x = Math.min(1, Math.max(0, t));
  return x * x * x * (x * (x * 6 - 15) + 10);
}

export interface PipelineDiagramProps {
  height?: number;
}

export default function PipelineDiagram({ height = 220 }: PipelineDiagramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;
    const dpr = Math.min(2, window.devicePixelRatio || 1);

    let width = wrap.clientWidth || 600;
    let raf = 0;
    let start = performance.now();
    const dwellMs = 1400;
    const cycleMs = prefersReducedMotion ? 1 : 2800;

    function resize() {
      width = wrap!.clientWidth || 600;
      canvas!.width = width * dpr;
      canvas!.height = height * dpr;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const nodeCount = STAGES.length;
    const padding = 40;
    const nodeX = (i: number) =>
      padding + (i / (nodeCount - 1)) * (width - padding * 2);
    const baseY = height / 2 - 8;
    const maxRadius = 22;

    function frame(now: number) {
      const elapsed = (now - start) % (cycleMs + dwellMs);
      const t = ease(Math.min(1, elapsed / cycleMs));
      const spanX = width - padding * 2;

      ctx!.clearRect(0, 0, width, height);

      // soft ground glow
      const ground = ctx!.createLinearGradient(0, 0, width, 0);
      ground.addColorStop(0, "rgba(124, 231, 220, 0.04)");
      ground.addColorStop(0.5, "rgba(154, 140, 255, 0.08)");
      ground.addColorStop(1, "rgba(239, 183, 110, 0.04)");
      ctx!.fillStyle = ground;
      ctx!.fillRect(0, 0, width, height);

      // flow spine
      ctx!.strokeStyle = "rgba(154, 140, 255, 0.35)";
      ctx!.lineWidth = 2;
      ctx!.beginPath();
      STAGES.forEach((_, i) => {
        const x = nodeX(i);
        if (i === 0) ctx!.moveTo(x, baseY);
        else ctx!.lineTo(x, baseY);
      });
      ctx!.stroke();

      // pulse
      if (!prefersReducedMotion) {
        const pulseX = padding + t * spanX;
        const grad = ctx!.createRadialGradient(pulseX, baseY, 0, pulseX, baseY, 48);
        grad.addColorStop(0, "rgba(124, 231, 220, 0.55)");
        grad.addColorStop(1, "rgba(124, 231, 220, 0)");
        ctx!.fillStyle = grad;
        ctx!.fillRect(pulseX - 48, baseY - 48, 96, 96);
      }

      STAGES.forEach((stage, i) => {
        const x = nodeX(i);
        const r = 11 + stage.weight * maxRadius;
        const active =
          !prefersReducedMotion &&
          Math.abs(padding + t * spanX - x) < spanX / (nodeCount * 1.2);

        ctx!.beginPath();
        ctx!.arc(x, baseY, r, 0, Math.PI * 2);
        ctx!.fillStyle = active
          ? "rgba(124, 231, 220, 0.28)"
          : "rgba(154, 140, 255, 0.16)";
        ctx!.fill();
        ctx!.strokeStyle = active ? "#7ce7dc" : "#9a8cff";
        ctx!.lineWidth = active ? 2 : 1.4;
        ctx!.stroke();

        ctx!.fillStyle = "#e8e6ef";
        ctx!.font = '600 11px "DM Mono", ui-monospace, monospace';
        ctx!.textAlign = "center";
        ctx!.fillText(stage.label, x, baseY + r + 18);
      });

      raf = requestAnimationFrame(frame);
    }

    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [height]);

  return (
    <div className="pipeline-diagram" ref={wrapRef}>
      <canvas ref={canvasRef} style={{ width: "100%", height }} />
      <p className="pipeline-caption">
        ASP stitch stages sized by typical runtime share — not a live per-run
        clock. Human coherence is scored separately on the quality dashboard.
      </p>
    </div>
  );
}
