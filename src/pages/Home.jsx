import React from 'react';
import { Link } from 'react-router-dom';
import { ElitePlanCard } from '../components/ui/ElitePlanCard';

const Home = () => {
    return (
        <div className="home-page">
            {/* Massive Atmospheric Hero */}
            <section className="hero-section">
                {/* Background Glow Effects */}
                <div className="hero-glow"></div>
                <div className="hero-grid-overlay"></div>

                {/* Content */}
                <div className="hero-content">
                    <div className="hero-badge">
                        <span className="badge-dot"></span>
                        Academic Prototype
                    </div>

                    <h1 className="hero-title">
                        Silent Trails
                    </h1>
                    <p className="hero-tagline">
                        <span className="hero-highlight">Profiling Human Patterns Across the Digitalverse.</span>
                    </p>

                    <p className="hero-subtitle">
                        Advanced threat intelligence and OSINT reconnaissance at your fingertips
                    </p>

                    <Link to="/digital-recon" className="hero-cta">
                        Launch App
                    </Link>
                </div>
            </section>

            {/* Module Cards */}
            <section className="modules-section">
                <h2 className="section-title">Security Modules</h2>
                <div className="modules-grid">
                    {/* Digital Recon */}
                    <ElitePlanCard
                        title="Digital Recon"
                        subtitle="OSINT & Reconnaissance"
                        description="Advanced tracking system. Instantly locate assets using SpiderFoot and multi-source threat intelligence."
                        imageUrl="/src/assets/digital_recon_card_bg.png"
                        highlights={["SpiderFoot Integration", "Phishing Detection", "Dark Web Scan", "IP Tracking"]}
                        link="/digital-recon"
                    />

                    {/* Deepfake Forensics */}
                    <ElitePlanCard
                        title="Deepfake Forensics"
                        subtitle="AI Integrity Analysis"
                        description="Verify authenticity. Detect AI-generated manipulation in video and audio streams."
                        imageUrl="/src/assets/deepfake_forensics_card_bg.png"
                        highlights={["Lip-Sync Detection", "Voice Pattern Analysis", "Metadata Forensic", "Real-time Scan"]}
                        link="/deepfake-forensics"
                    />

                    {/* Timeline Engine */}
                    <ElitePlanCard
                        title="Timeline Engine"
                        subtitle="Forensic Intelligence"
                        description="Reconstruct digital footprints into chronological narratives from fragmented OSINT data."
                        imageUrl="/src/assets/timeline_engine_card_bg.png"
                        highlights={["Event Correlation", "Anomaly Detection", "Investigation Reports", "Activity Heatmap"]}
                        link="/timeline"
                    />
                </div>
            </section>

            {/* Trust Bar */}
            <section className="trust-bar">
                <span>Powered by</span>
                <div className="trust-logos">
                    <span>🕷️ SpiderFoot</span>
                    <span>🔒 VirusTotal</span>
                    <span>📡 URLhaus</span>
                    <span>🗄️ Supabase</span>
                </div>
            </section>
        </div>
    );
};

export default Home;
