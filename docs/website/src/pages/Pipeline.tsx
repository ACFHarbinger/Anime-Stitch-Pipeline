import { motion } from "framer-motion";
import PipelineDiagram from "../components/PipelineDiagram";
import { site } from "../configs/site";

export default function Pipeline() {
  return <div className="max-w-[1400px] mx-auto px-8 pt-32 pb-24 min-h-screen">
    <span className="text-[#ffcf4a] text-xs font-mono tracking-[0.2em] font-bold uppercase">{site.shortName} / PIPELINE OBSERVATORY</span>
    <h1 className="text-4xl md:text-6xl font-bold mt-4 tracking-tight text-[#e2e8f0]">A measurable path from input<br />to finished work.</h1>
    <p className="mt-5 max-w-2xl text-lg text-[#8c92a0] leading-relaxed">{site.heroDescription} The stages below make the hand-offs visible so experiments, failures, and improvements remain reviewable.</p>
    <div className="mt-14 border border-[#1a1c23] bg-[#0a0a0c] p-6 md:p-10 shadow-[0_0_30px_rgba(0,240,255,0.08)]"><div className="flex justify-between text-[#4a4d57] text-[10px] font-mono tracking-[0.18em] uppercase mb-4"><span>{site.name} / SYSTEM FLOW</span><span className="text-[#00f0ff]">● ACTIVE</span></div><PipelineDiagram height={300} /></div>
    <div className="grid md:grid-cols-3 gap-5 mt-8">{site.stages.map((stage, index) => <motion.article key={stage} initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: index * .06 }} className="border border-[#1a1c23] bg-[#0a0a0c] p-6"><span className="text-[#00f0ff] text-[10px] font-mono">0{index + 1}</span><h2 className="mt-5 text-xl font-bold text-[#e2e8f0]">{stage}</h2><p className="mt-3 text-sm leading-relaxed text-[#8c92a0]">A documented boundary in the {site.shortName} processing graph.</p></motion.article>)}</div>
    <a href={site.repository} target="_blank" rel="noreferrer" className="inline-block mt-10 text-[#00f0ff] font-mono text-sm">Inspect implementation on GitHub ↗</a>
  </div>;
}
