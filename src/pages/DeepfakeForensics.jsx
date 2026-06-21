import React, { useState } from 'react';
import './DeepfakeForensics.css';

const DeepfakeForensics = () => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [currentStep, setCurrentStep] = useState('');
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [dragActive, setDragActive] = useState(false);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
        else if (e.type === "dragleave") setDragActive(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
    };

    const handleFileChange = (e) => {
        if (e.target.files?.[0]) handleFile(e.target.files[0]);
    };

    const handleFile = (selectedFile) => {
        const isImage = selectedFile.type.startsWith('image/');
        const isVideo = selectedFile.type.startsWith('video/');
        if (!isImage && !isVideo) {
            setError('Please upload an image (JPG, PNG, WebP) or video (MP4, AVI) file');
            return;
        }
        setError(null);
        setFile(selectedFile);
        setPreview(URL.createObjectURL(selectedFile));
        setResults(null);
    };

    const analyzeFile = async () => {
        if (!file) return;
        setAnalyzing(true);
        setProgress(0);
        setResults(null);
        setError(null);

        const isVideo = file.type.startsWith('video/');
        const steps = isVideo
            ? [
                { msg: 'Uploading video to Forensics engine...', pct: 10 },
                { msg: 'Extracting frames from video...', pct: 25 },
                { msg: 'Running face detection per frame...', pct: 45 },
                { msg: 'Spatial analysis (SigLIP2 Vision Model)...', pct: 60 },
                { msg: 'Temporal pattern analysis...', pct: 78 },
                { msg: 'Aggregating frame-level verdicts...', pct: 90 },
                { msg: 'Generating forensic report...', pct: 97 },
            ]
            : [
                { msg: 'Uploading image to Forensics engine...', pct: 15 },
                { msg: 'Running face detection...', pct: 35 },
                { msg: 'Spatial analysis (SigLIP2 Vision Model)...', pct: 60 },
                { msg: 'Attention forgery localization...', pct: 82 },
                { msg: 'Generating forensic verdict...', pct: 97 },
            ];

        // Animate steps while the real request is in-flight
        let stepIdx = 0;
        const interval = setInterval(() => {
            if (stepIdx < steps.length) {
                setCurrentStep(steps[stepIdx].msg);
                setProgress(steps[stepIdx].pct);
                stepIdx++;
            }
        }, isVideo ? 2500 : 1200);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('http://localhost:5000/api/deepfake-detect', {
                method: 'POST',
                body: formData,
            });

            clearInterval(interval);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || errData.hint || `Server error ${response.status}`);
            }

            const data = await response.json();
            setProgress(100);
            setCurrentStep('Analysis Complete');

            setTimeout(() => {
                setResults({
                    ...data,
                    file_name: file.name,
                    file_size: (file.size / 1024).toFixed(1) + ' KB',
                    file_type: isVideo ? 'Video' : 'Image',
                    message: null,
                });
                setAnalyzing(false);
            }, 400);

        } catch (err) {
            clearInterval(interval);
            setError(`Analysis failed: ${err.message}`);
            setAnalyzing(false);
            setProgress(0);
        }
    };

    const reset = () => {
        setFile(null);
        setPreview(null);
        setResults(null);
        setProgress(0);
        setError(null);
    };

    return (
        <div className="deepfake-container">
            <div className="df-header">
                <h1>Deepfake Forensics Engine</h1>
                <p>Multi-stream deepfake detection: Spatial + Frequency + Attention + Temporal analysis</p>
            </div>

            {/* ERROR MESSAGE */}
            {error && (
                <div className="df-error">
                    <span>⚠️</span> {error}
                    <button onClick={() => setError(null)}>✕</button>
                </div>
            )}

            {/* UPLOAD ZONE */}
            {!file && (
                <div
                    className={`df-upload-zone ${dragActive ? 'drag-active' : ''}`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => document.getElementById('file-upload').click()}
                >
                    <input type="file" id="file-upload" hidden onChange={handleFileChange} accept="image/*,video/*" />
                    <div className="upload-icon">
                        <img src="/digital.ico" alt="Upload" style={{ width: '64px', opacity: 0.8 }} />
                    </div>
                    <h3>Drag & Drop Evidence File</h3>
                    <p>Supported: JPG, PNG, WEBP, MP4, AVI</p>
                    <div className="upload-techniques">
                        <span>🧊 DINOv2 Spatial</span>
                        <span>📊 FFT Frequency</span>
                        <span>🎯 Attention Maps</span>
                        <span>🎬 Temporal Analysis</span>
                    </div>
                </div>
            )}

            {/* FILE PREVIEW + ANALYZE BUTTON */}
            {file && !analyzing && !results && (
                <div className="df-preview-stage">
                    <div className="preview-section">
                        <h3>Selected File</h3>
                        {file.type.startsWith('image/') ? (
                            <img src={preview} alt="Evidence" className="evidence-preview" />
                        ) : (
                            <video src={preview} controls className="evidence-preview" />
                        )}
                        <p>{file.name} — {(file.size / 1024).toFixed(1)} KB</p>
                    </div>
                    <div className="action-buttons">
                        <button className="analyze-btn" onClick={analyzeFile}>🔍 Analyze</button>
                        <button className="reset-btn" onClick={reset}>Cancel</button>
                    </div>
                </div>
            )}

            {/* ANALYSIS PROGRESS */}
            {analyzing && (
                <div className="df-analysis">
                    <h2>Analyzing Evidence...</h2>
                    <div className="progress-bar-container">
                        <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
                    </div>
                    <div className="progress-label">{currentStep}</div>
                    <div className="analysis-steps">
                        <span className={progress > 15 ? 'active' : ''}>Upload</span>
                        <span className={progress > 30 ? 'active' : ''}>Face Extract</span>
                        <span className={progress > 50 ? 'active' : ''}>Spatial</span>
                        <span className={progress > 65 ? 'active' : ''}>Frequency</span>
                        <span className={progress > 80 ? 'active' : ''}>Attention</span>
                        <span className={progress >= 100 ? 'active' : ''}>Verdict</span>
                    </div>
                </div>
            )}

            {/* RESULTS DASHBOARD */}
            {!analyzing && results && (
                <div className="df-results">
                    <div className={`verdict-banner ${
                        results.verdict === 'MANIPULATED' ? 'fake' :
                        results.verdict === 'SUSPICIOUS' ? 'suspicious' :
                        results.verdict === 'PENDING' ? 'pending' : 'real'
                        }`}>
                        <div className="verdict-icon">
                            {results.verdict === 'MANIPULATED' ? '🚨' :
                             results.verdict === 'SUSPICIOUS' ? '⚠️' :
                             results.verdict === 'PENDING' ? '🔧' : '✅'}
                        </div>
                        <div className="verdict-info">
                            <h2>{results.verdict === 'MANIPULATED' ? 'MANIPULATION DETECTED' :
                                 results.verdict === 'SUSPICIOUS' ? 'SUSPICIOUS — POSSIBLE MANIPULATION' :
                                 results.verdict === 'PENDING' ? 'MODEL NOT CONNECTED' : 'LIKELY AUTHENTIC'}</h2>
                            <p>
                                {results.message ||
                                    `Confidence: ${results.confidence}% | Fake Probability: ${results.fake_probability}%`}
                            </p>
                        </div>
                    </div>

                    <div className="result-grid">
                        {/* File Info */}
                        <div className="result-card">
                            <h3>File Info</h3>
                            <div className="info-row"><span>Filename</span><span>{results.file_name}</span></div>
                            <div className="info-row"><span>Size</span><span>{results.file_size}</span></div>
                            <div className="info-row"><span>Type</span><span>{results.file_type}</span></div>
                            {results.frame_count > 1 && (
                                <div className="info-row"><span>Frames Analyzed</span><span>{results.frame_count}</span></div>
                            )}
                            {results.suspicious_frames != null && (
                                <div className="info-row"><span>Suspicious Frames</span>
                                    <span style={{ color: results.suspicious_frames > 0 ? '#f97316' : '#22d3ee' }}>
                                        {results.suspicious_frames} / {results.frame_count}
                                    </span>
                                </div>
                            )}
                        </div>

                        {/* Forensic Stream Scores */}
                        <div className="result-card">
                            <h3>Stream Analysis</h3>
                            <div className="info-row">
                                <span>🧊 Spatial Score</span>
                                <span>{results.streams?.spatial != null ? `${results.streams.spatial}%` : '—'}</span>
                            </div>
                            <div className="info-row">
                                <span>📊 Frequency Score</span>
                                <span>{results.streams?.frequency != null ? `${results.streams.frequency}%` : '—'}</span>
                            </div>
                            <div className="info-row">
                                <span>🎯 Attention Score</span>
                                <span>{results.streams?.attention != null ? `${results.streams.attention}%` : '—'}</span>
                            </div>
                            {results.streams?.temporal != null && (
                                <div className="info-row">
                                    <span>🎬 Temporal Score</span>
                                    <span>{results.streams.temporal}%</span>
                                </div>
                            )}
                            <div className="info-row" style={{ marginTop: '8px', borderTop: '1px solid rgba(6,182,212,0.2)', paddingTop: '8px' }}>
                                <span><strong>Overall Fake Probability</strong></span>
                                <span style={{ color: results.fake_probability >= 50 ? '#f97316' : '#22d3ee', fontWeight: 700 }}>
                                    {results.fake_probability}%
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Video Frame Timeline */}
                    {results.timeline && results.timeline.length > 0 && (
                        <div className="result-card" style={{ marginTop: '16px' }}>
                            <h3>🎬 Frame-by-Frame Timeline</h3>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
                                Each frame analyzed independently — red = suspicious
                            </p>
                            <div className="frame-timeline">
                                {results.timeline.map((frame, idx) => (
                                    <div key={idx} className={`frame-card ${frame.suspicious ? 'frame-suspicious' : 'frame-clean'}`}>
                                        {frame.thumbnail && (
                                            <img
                                                src={`data:image/jpeg;base64,${frame.thumbnail}`}
                                                alt={`Frame ${frame.frame_index}`}
                                                className="frame-thumb"
                                            />
                                        )}
                                        <div className="frame-label">
                                            <span>#{frame.frame_index + 1}</span>
                                            <span style={{ color: frame.suspicious ? '#f97316' : '#22d3ee', fontWeight: 600 }}>
                                                {frame.fake_probability}%
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    <div className="preview-section" style={{ marginTop: '16px' }}>
                        <h3>Analyzed File</h3>
                        {file?.type.startsWith('image/') ? (
                            <img src={preview} alt="Evidence" className="evidence-preview" />
                        ) : (
                            <video src={preview} controls className="evidence-preview" />
                        )}
                    </div>

                    <button className="reset-btn" onClick={reset}>Analyze Another File</button>
                </div>
            )}
        </div>
    );
};

export default DeepfakeForensics;
