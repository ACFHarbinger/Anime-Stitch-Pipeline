import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import { Aperture, ExternalLink } from "lucide-react";
import Home from "./pages/Home";
import Docs from "./pages/Docs";
import Pipeline from "./pages/Pipeline";
import { site } from "./configs/site";
import "./App.css";

function Nav() {
  const { pathname } = useLocation();
  const active = (prefix: string) => pathname.startsWith(prefix);
  return <nav className="site-nav" aria-label="Primary"><div className="site-nav-inner">
    <Link to="/" className="brand"><span className="brand-mark"><Aperture size={18} strokeWidth={2.2} /></span><span className="brand-text"><strong>{site.name}</strong><em>{site.shortName} / Image-Toolkit</em></span></Link>
    <div className="nav-links"><Link to="/" className={!active("/dashboard") && !active("/docs") && !active("/pipeline") ? "active" : undefined}>Home</Link><Link to="/pipeline" className={active("/pipeline") ? "active" : undefined}>Pipeline</Link><Link to="/docs" className={active("/docs") ? "active" : undefined}>Docs</Link><Link to="/dashboard" className={active("/dashboard") ? "active" : undefined}>Signals</Link></div>
    <a href="https://acfharbinger.github.io/Image-Toolkit/" className="nav-cta">Back to Image-Toolkit <ExternalLink size={13} /></a>
  </div></nav>;
}

export default function App() { return <BrowserRouter basename={import.meta.env.BASE_URL}><div className="site-shell"><Nav /><main><Routes><Route path="/" element={<Home />} /><Route path="/pipeline" element={<Pipeline />} /><Route path="/docs" element={<Docs />} /></Routes></main></div></BrowserRouter>; }
