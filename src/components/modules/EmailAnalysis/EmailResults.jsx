import React from 'react';

const EmailResults = ({ results }) => {
    if (!results) return null;

    const { breachCount, breaches } = results;

    if (breachCount === 0) {
        return (
            <div className="results-section">
                <div className="results-list" style={{ textAlign: 'center', padding: 'var(--spacing-3xl)' }}>
                    <div style={{ fontSize: '48px', marginBottom: 'var(--spacing-md)' }}>✅</div>
                    <h3 style={{ color: 'var(--accent)', marginBottom: 'var(--spacing-sm)' }}>
                        No Breaches Found
                    </h3>
                    <p>This email address has not appeared in any known data breaches.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="results-section">
            <div className="results-list">
                <h3 className="results-title" style={{ marginBottom: 'var(--spacing-lg)' }}>
                    ⚠️ Found in {breachCount} Breach{breachCount !== 1 ? 'es' : ''}
                </h3>
                {breaches.map((breach, index) => (
                    <div key={index} className="result-item">
                        <span className="result-item-title">{breach.name}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default EmailResults;
