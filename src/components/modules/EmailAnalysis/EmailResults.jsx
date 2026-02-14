import React from 'react';

const EmailResults = ({ results }) => {
    if (!results) return null;

    const { email, breachCount, breaches, riskLevel, recommendations } = results;

    return (
        <div className="results-section">
            {/* Summary Card */}
            <div className="results-summary">
                <div className="results-header">
                    <h3 className="results-title">Analysis Results</h3>
                    <span className={`risk-badge ${riskLevel}`}>
                        {riskLevel} risk
                    </span>
                </div>

                <p style={{ marginBottom: 'var(--spacing-md)' }}>
                    <strong>Email analyzed:</strong> {email}
                </p>

                <div className="summary-stats">
                    <div className="stat-item">
                        <span className={`stat-value ${riskLevel === 'high' ? 'risk-high' : riskLevel === 'medium' ? 'risk-medium' : 'risk-low'}`}>
                            {breachCount}
                        </span>
                        <span className="stat-label">Breaches Found</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-value">
                            {breaches.reduce((acc, b) => acc + b.dataTypes.length, 0)}
                        </span>
                        <span className="stat-label">Data Types Exposed</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-value" style={{ textTransform: 'capitalize' }}>
                            {riskLevel}
                        </span>
                        <span className="stat-label">Risk Level</span>
                    </div>
                </div>
            </div>

            {/* Breaches List */}
            {breachCount > 0 && (
                <div className="results-list">
                    <h3 className="results-title" style={{ marginBottom: 'var(--spacing-lg)' }}>
                        Data Breaches
                    </h3>

                    {breaches.map((breach, index) => (
                        <div key={index} className="result-item">
                            <div className="result-item-header">
                                <span className="result-item-title">{breach.name}</span>
                                <span className={`risk-badge ${breach.severity}`}>
                                    {breach.severity}
                                </span>
                            </div>
                            <p className="result-item-meta">
                                Breach date: {new Date(breach.date).toLocaleDateString('en-US', {
                                    year: 'numeric',
                                    month: 'long',
                                    day: 'numeric'
                                })}
                            </p>
                            <div className="result-item-body">
                                <p style={{ marginBottom: 'var(--spacing-sm)' }}>Exposed data types:</p>
                                <div>
                                    {breach.dataTypes.map((type, i) => (
                                        <span key={i} className="tag">{type}</span>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* No Breaches */}
            {breachCount === 0 && (
                <div className="results-list" style={{ textAlign: 'center', padding: 'var(--spacing-3xl)' }}>
                    <div style={{ fontSize: '48px', marginBottom: 'var(--spacing-md)' }}>✅</div>
                    <h3 style={{ color: 'var(--accent)', marginBottom: 'var(--spacing-sm)' }}>
                        Good News!
                    </h3>
                    <p>No data breaches found for this email address.</p>
                </div>
            )}

            {/* Recommendations */}
            <div className="results-list" style={{ marginTop: 'var(--spacing-xl)' }}>
                <h3 className="results-title" style={{ marginBottom: 'var(--spacing-lg)' }}>
                    Recommendations
                </h3>
                <ul style={{ paddingLeft: 'var(--spacing-lg)' }}>
                    {recommendations.map((rec, index) => (
                        <li key={index} style={{
                            marginBottom: 'var(--spacing-sm)',
                            color: 'var(--text-secondary)'
                        }}>
                            {rec}
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default EmailResults;
