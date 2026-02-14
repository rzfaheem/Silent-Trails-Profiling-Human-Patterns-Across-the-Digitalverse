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

        // TODO: Connect to DeepShield-X API when model is deployed
        // const formData = new FormData();
        // formData.append('file', file);
        // const response = await fetch('http://localhost:8000/detect', { method: 'POST', body: formData });
        // const data = await response.json();

        // Placeholder — will be replaced with real API call
        const steps = [
            { msg: 'Uploading to DeepShield-X engine...', pct: 15 },
            { msg: 'Extracting face regions (RetinaFace)...', pct: 30 },
            { msg: 'Spatial analysis (DINOv2 + LoRA)...', pct: 50 },
            { msg: 'Frequency domain analysis (FFT)...', pct: 65 },
            { msg: 'Attention forgery localization...', pct: 80 },
            { msg: 'Generating forensic verdict...', pct: 95 },
        ];

        let stepIdx = 0;
        const interval = setInterval(() => {
            if (stepIdx < steps.length) {
                setCurrentStep(steps[stepIdx].msg);
                setProgress(steps[stepIdx].pct);
                stepIdx++;
            } else {
                clearInterval(interval);
                setProgress(100);
                setCurrentStep('Analysis Complete');
                setTimeout(() => {
                    setResults({
                        verdict: 'PENDING',
                        confidence: 0,
                        manipulation_score: 0,
                        message: 'DeepShield-X model not yet connected. Train and deploy the model first.',
                        file_name: file.name,
                        file_size: (file.size / 1024).toFixed(1) + ' KB',
                        file_type: file.type.startsWith('video/') ? 'Video' : 'Image',
                    });
                    setAnalyzing(false);
                }, 500);
            }
        }, 600);
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
                <h1>DeepShield-X Forensics Engine</h1>
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
                        <button className="analyze-btn" onClick={analyzeFile}>🔍 Analyze with DeepShield-X</button>
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
                    <div className={`verdict-banner ${results.verdict === 'MANIPULATED' ? 'fake' :
                            results.verdict === 'PENDING' ? 'pending' : 'real'
                        }`}>
                        <div className="verdict-icon">
                            {results.verdict === 'MANIPULATED' ? '⚠️' :
                                results.verdict === 'PENDING' ? '🔧' : '✅'}
                        </div>
                        <div className="verdict-info">
                            <h2>{results.verdict === 'MANIPULATED' ? 'MANIPULATION DETECTED' :
                                results.verdict === 'PENDING' ? 'MODEL NOT CONNECTED' : 'LIKELY AUTHENTIC'}</h2>
                            <p>{results.message || `Confidence: ${results.confidence}% | Score: ${results.manipulation_score}/100`}</p>
                        </div>
                    </div>

                    <div className="result-grid">
                        <div className="result-card">
                            <h3>File Info</h3>
                            <div className="info-row"><span>Filename</span> <span>{results.file_name}</span></div>
                            <div className="info-row"><span>Size</span> <span>{results.file_size}</span></div>
                            <div className="info-row"><span>Type</span> <span>{results.file_type}</span></div>
                        </div>

                        {/* Placeholder cards for future real results */}
                        <div className="result-card">
                            <h3>Forensic Analysis</h3>
                            <div className="info-row"><span>Spatial Score</span> <span>—</span></div>
                            <div className="info-row"><span>Frequency Score</span> <span>—</span></div>
                            <div className="info-row"><span>Attention Score</span> <span>—</span></div>
                        </div>
                    </div>

                    <div className="preview-section">
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
