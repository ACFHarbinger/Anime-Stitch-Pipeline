import { motion } from "framer-motion";
import { lazy, Suspense, useState } from "react";
import { ExternalLink } from "lucide-react";
import Viewfinder2D from "../components/Viewfinder2D";
import PipelineDiagram from "../components/PipelineDiagram";
import { site } from "../configs/site";
import "../App.css";

const Hero3D = lazy(() => import("../components/Hero3D"));
const accent = site.accent === "rose"
  ? { text: "text-[#ff5c93]", border: "border-[#ff0055]", bg: "bg-[#ff0055]/10", glow: "shadow-[0_0_30px_rgba(255,0,85,0.15)]" }
  : site.accent === "violet"
    ? { text: "text-[#b895ff]", border: "border-[#9b6cff]", bg: "bg-[#9b6cff]/10", glow: "shadow-[0_0_30px_rgba(155,108,255,0.15)]" }
    : { text: "text-[#00f0ff]", border: "border-[#00f0ff]", bg: "bg-[#00f0ff]/10", glow: "shadow-[0_0_30px_rgba(0,240,255,0.15)]" };

export default function Home() {
  const [active, setActive] = useState(0);
  return <div className="home-page min-h-screen bg-[#050505] text-[#e2e8f0] overflow-hidden">
    <section className="relative min-h-[90vh] max-w-[1400px] mx-auto px-8 flex items-center overflow-hidden">
      <div className="absolute inset-0"><img src={`${import.meta.env.BASE_URL}anime_lab_hero.png`} alt="" className="w-full h-full object-cover opacity-25 saturate-150 mix-blend-screen" /><div className="absolute inset-0 bg-gradient-to-t from-[#050505] via-[#050505]/65 to-transparent" /></div>
      <Viewfinder2D /><Suspense fallback={null}><Hero3D /></Suspense>
      <motion.div initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 1.1 }} className="relative z-30 space-y-6 max-w-2xl mt-20">
        <div className={`inline-flex items-center gap-3 px-5 py-2 rounded border ${accent.border}/30 ${accent.bg} ${accent.text} text-xs font-mono tracking-widest uppercase`}><span className="w-2 h-2 rounded-full bg-[#00f0ff] animate-pulse" />{site.badge}</div>
        <h1 className="text-6xl md:text-8xl font-bold leading-[0.95] tracking-tighter text-transparent bg-clip-text bg-gradient-to-br from-white via-[#e2e8f0] to-[#8c92a0]">{site.heroTitle[0]}<br /><span>{site.heroTitle[1]}</span></h1>
        <p className="text-[#a0a5b5] text-xl max-w-lg leading-relaxed font-light">{site.heroDescription}</p>
        <div className="flex flex-wrap gap-4 pt-8"><a href="#modules" className={`px-7 py-4 rounded border ${accent.border} ${accent.bg} ${accent.text} font-mono ${accent.glow}`}>Explore the system</a><a href={site.repository} target="_blank" rel="noreferrer" className="px-7 py-4 rounded border border-[#333538] bg-black/40 text-[#e2e8f0] font-mono">View repository <ExternalLink className="inline ml-2" size={14} /></a></div>
      </motion.div>
    </section>
    <section id="modules" className="relative z-10 py-28 px-8 max-w-[1400px] mx-auto border-t border-[#1a1c23] bg-gradient-to-b from-[#0a0a0c] to-[#050505]">
      <div className="mb-16 text-center"><span className={`${accent.text} text-xs font-mono tracking-[0.2em] font-bold uppercase`}>{site.shortName} / SYSTEM MODULES</span><h2 className="text-4xl md:text-5xl font-bold mt-4">A focused instrument.</h2></div>
      <div className="grid md:grid-cols-3 gap-8">{site.modules.map((item, index) => { const Icon = item.icon; const selected = active === index; return <motion.article key={item.title} whileHover={{ y: -5 }} onClick={() => setActive(index)} className={`p-9 border bg-[#0a0a0c] group cursor-pointer ${selected ? accent.border + "/70 " + accent.glow : "border-[#1a1c23]"}`}><div className="flex justify-between items-start mb-14"><span className="font-mono text-4xl font-bold text-[#1a1c23]">{item.number}</span><Icon size={28} className={`${accent.text} opacity-70`} /></div><h3 className="text-2xl font-bold mb-4">{item.title}</h3><p className="text-[#8c92a0] leading-relaxed">{item.text}</p></motion.article>; })}</div>
      <motion.div key={active} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className={`mt-8 border ${accent.border}/20 bg-[#08090c] p-7`}><span className={`${accent.text} text-[10px] font-mono tracking-[0.2em]`}>ACTIVE MODULE / {site.modules[active].number}</span><p className="mt-3 text-[#8c92a0]">{site.modules[active].detail}</p></motion.div>
    </section>
    <section className="relative z-10 py-28 px-8 max-w-[1400px] mx-auto border-t border-[#1a1c23]"><div className="grid lg:grid-cols-[.7fr_1.3fr] gap-12 items-center"><div><span className="text-[#ffcf4a] text-xs font-mono tracking-[0.2em] font-bold uppercase">{site.shortName} / PIPELINE OBSERVATORY</span><h2 className="text-4xl md:text-5xl font-bold mt-4">Follow the signal<br />through the system.</h2><p className="mt-5 text-[#8c92a0] leading-relaxed">Every boundary is measurable, reviewable, and documented.</p></div><div className={`border border-[#1a1c23] bg-[#0a0a0c] p-5 ${accent.glow}`}><PipelineDiagram height={240} /></div></div><div className="mt-8 flex flex-wrap justify-center gap-3 text-[10px] font-mono text-[#8c92a0]">{site.stages.map(stage => <span key={stage} className={`border border-[#1a1c23] px-3 py-2 ${accent.text}`}>{stage}</span>)}</div></section>
    <section className="relative z-10 py-20 px-8 max-w-[1400px] mx-auto border-t border-[#1a1c23]"><span className={`${accent.text} text-xs font-mono tracking-[0.2em] font-bold uppercase`}>DOCUMENTATION / {site.shortName}</span><h2 className="text-4xl font-bold mt-4">Read the engineering trail.</h2><a href={site.repository} target="_blank" rel="noreferrer" className={`inline-block mt-7 ${accent.text} font-mono text-sm`}>Open repository documentation ↗</a></section>
    <footer className="border-t border-[#1a1c23] px-8 py-12 flex justify-between text-[#4a4d57] text-[10px] font-mono"><span>{site.name.toUpperCase()} / IMAGE-TOOLKIT</span><a className={accent.text} href="https://acfharbinger.github.io/Image-Toolkit/">Return to Image-Toolkit ↗</a></footer>
  </div>;
}

