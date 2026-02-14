import React, { useState } from 'react';
import EmailForm from '../components/modules/EmailAnalysis/EmailForm';
import EmailResults from '../components/modules/EmailAnalysis/EmailResults';
import Loader from '../components/common/Loader';
import { analyzeEmail } from '../services/mockApi';

const EmailAnalysis = () => {
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const handleAnalyze = async (email) => {
        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const data = await analyzeEmail(email);
            setResults(data);
        } catch (err) {
            setError('An error occurred while analyzing the email. Please try again.');
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
                        <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
                        <polyline points="22,6 12,13 2,6" />
                    </svg>
                </div>
                <h1>Email Leak & Exposure Analysis</h1>
                <p>
                    Check if your email address has been compromised in known data breaches.
                    Enter an email to scan our database of breach records.
                </p>
            </div>

            {/* Form */}
            <EmailForm onSubmit={handleAnalyze} loading={loading} />

            {/* Loading State */}
            {loading && (
                <Loader text="Scanning breach databases..." />
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
            {!loading && results && <EmailResults results={results} />}
        </div>
    );
};

export default EmailAnalysis;
