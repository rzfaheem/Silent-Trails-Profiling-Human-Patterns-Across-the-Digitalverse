import React, { useState } from 'react';
import PhishingForm from '../components/modules/PhishingDetection/PhishingForm';
import PhishingResults from '../components/modules/PhishingDetection/PhishingResults';
import Loader from '../components/common/Loader';
import { analyzePhishing } from '../services/mockApi';

const PhishingDetection = () => {
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const handleAnalyze = async (input) => {
        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const data = await analyzePhishing(input);
            setResults(data);
        } catch (err) {
            setError('An error occurred while analyzing the content. Please try again.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="module-page">
            {/* Header */}
            <div className="module-header">
                <div className="module-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                        <line x1="12" y1="9" x2="12" y2="13" />
                        <line x1="12" y1="17" x2="12.01" y2="17" />
                    </svg>
                </div>
                <h1>Phishing URL & Message Detection</h1>
                <p>
                    Analyze suspicious URLs and messages for phishing indicators.
                    Our system detects common phishing patterns and threats.
                </p>
            </div>

            {/* Form */}
            <PhishingForm onSubmit={handleAnalyze} loading={loading} />

            {/* Loading State */}
            {loading && (
                <Loader text="Analyzing for phishing indicators..." />
            )}

            {/* Error State */}
            {error && (
                <div style={{
                    marginTop: 'var(--spacing-xl)',
                    padding: 'var(--spacing-lg)',
                    backgroundColor: 'rgba(229, 62, 62, 0.1)',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--danger)',
                    textAlign: 'center'
                }}>
                    {error}
                </div>
            )}

            {/* Results */}
            {!loading && results && <PhishingResults results={results} />}
        </div>
    );
};

export default PhishingDetection;
