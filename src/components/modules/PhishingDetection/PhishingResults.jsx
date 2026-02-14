import React from 'react';

const PhishingResults = ({ results }) => {
    if (!results) return null;

    const { input, isPhishing, confidence, riskLevel, indicators, recommendation, stats, sources, urlhausData } = results;

    // Handle API error state
    if (isPhishing === null) {
        return (
            <div className="results-section">
                <div className="phishing-verdict" style={{ backgroundColor: 'rgba(113, 128, 150, 0.1)', border: '2px solid #718096' }}>
                    <div className="verdict-icon" style={{ backgroundColor: '#718096' }}>
                        ❓
                    </div>
                    <h3 className="verdict-title" style={{ color: '#718096' }}>Unable to Verify</h3>
                    <p style={{ color: 'var(--text-secondary)', marginBottom: 0 }}>
                        {recommendation}
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="results-section">
            {/* Verdict Card */}
            <div className={`phishing-verdict ${isPhishing ? 'dangerous' : 'safe'}`}>
                <div className="verdict-icon">
                    {isPhishing ? '⚠️' : '✓'}
                </div>
                <h3 className="verdict-title">
                    {isPhishing ? 'Potential Threat Detected' : 'Appears Safe'}
                </h3>
                <p style={{ color: 'var(--text-secondary)', marginBottom: 0 }}>
                    {recommendation}
                </p>

                <div className="confidence-section">
                    <p className="confidence-label">Confidence Level</p>
                    <div className="progress-bar" style={{ maxWidth: '300px', margin: '0 auto', marginTop: 'var(--spacing-sm)' }}>
                        <div
                            className={`progress-bar-fill ${riskLevel}`}
                            style={{ width: `${confidence}%` }}
                        />
                    </div>
                    <p className={`confidence-value ${isPhishing ? 'risk-high' : 'risk-low'}`}>
                        {confidence}%
                    </p>
                </div>
            </div>

            {/* Source Badges */}
            {sources && sources.length > 0 && (
                <div style={{
                    textAlign: 'center',
                    marginBottom: 'var(--spacing-lg)',
                    marginTop: 'calc(var(--spacing-lg) * -1)',
                    display: 'flex',
                    justifyContent: 'center',
                    gap: 'var(--spacing-sm)',
                    flexWrap: 'wrap'
                }}>
                    {sources.includes('VirusTotal') && (
                        <span style={{
                            display: 'inline-block',
                            padding: 'var(--spacing-xs) var(--spacing-md)',
                            backgroundColor: 'rgba(49, 130, 206, 0.1)',
                            color: 'var(--info)',
                            borderRadius: 'var(--radius-full)',
                            fontSize: 'var(--font-size-xs)',
                            fontWeight: 'var(--font-medium)'
                        }}>
                            🛡️ VirusTotal
                        </span>
                    )}
                    {sources.includes('URLhaus') && (
                        <span style={{
                            display: 'inline-block',
                            padding: 'var(--spacing-xs) var(--spacing-md)',
                            backgroundColor: 'rgba(128, 90, 213, 0.1)',
                            color: '#805ad5',
                            borderRadius: 'var(--radius-full)',
                            fontSize: 'var(--font-size-xs)',
                            fontWeight: 'var(--font-medium)'
                        }}>
                            🦠 URLhaus
                        </span>
                    )}
                </div>
            )}

            {/* URLhaus Alert (if found in database) */}
            {urlhausData && (
                <div style={{
                    backgroundColor: 'rgba(229, 62, 62, 0.1)',
                    border: '2px solid var(--danger)',
                    borderRadius: 'var(--radius-lg)',
                    padding: 'var(--spacing-lg)',
                    marginBottom: 'var(--spacing-xl)'
                }}>
                    <h4 style={{ color: 'var(--danger)', marginBottom: 'var(--spacing-md)', display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
                        🦠 URLhaus Malware Database Match
                    </h4>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 'var(--spacing-md)' }}>
                        <div>
                            <span style={{ color: 'var(--text-muted)', fontSize: 'var(--font-size-xs)' }}>Threat Type</span>
                            <p style={{ fontWeight: 'var(--font-semibold)', margin: 0, color: 'var(--danger)' }}>
                                {urlhausData.threat || 'Unknown'}
                            </p>
                        </div>
                        <div>
                            <span style={{ color: 'var(--text-muted)', fontSize: 'var(--font-size-xs)' }}>Status</span>
                            <p style={{ fontWeight: 'var(--font-semibold)', margin: 0 }}>
                                {urlhausData.status || 'Active'}
                            </p>
                        </div>
                        {urlhausData.tags && urlhausData.tags.length > 0 && (
                            <div>
                                <span style={{ color: 'var(--text-muted)', fontSize: 'var(--font-size-xs)' }}>Tags</span>
                                <p style={{ margin: 0 }}>
                                    {urlhausData.tags.map((tag, i) => (
                                        <span key={i} className="tag" style={{ backgroundColor: 'rgba(229, 62, 62, 0.2)', color: 'var(--danger)' }}>
                                            {tag}
                                        </span>
                                    ))}
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Security Scan Stats */}
            {stats && (stats.total > 0 || stats.urlhausListed) && (
                <div className="results-summary">
                    <h3 className="results-title" style={{ marginBottom: 'var(--spacing-lg)' }}>
                        Security Scan Results
                    </h3>
                    <div className="summary-stats">
                        <div className="stat-item">
                            <span className="stat-value" style={{ color: stats.malicious > 0 ? 'var(--danger)' : 'var(--accent)' }}>
                                {stats.malicious}
                            </span>
                            <span className="stat-label">Malicious</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-value" style={{ color: stats.suspicious > 0 ? 'var(--warning)' : 'var(--accent)' }}>
                                {stats.suspicious}
                            </span>
                            <span className="stat-label">Suspicious</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-value" style={{ color: 'var(--accent)' }}>
                                {stats.harmless}
                            </span>
                            <span className="stat-label">Safe</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-value" style={{ color: stats.urlhausListed ? 'var(--danger)' : 'var(--accent)' }}>
                                {stats.urlhausListed ? '⚠️' : '✓'}
                            </span>
                            <span className="stat-label">URLhaus</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Analyzed Input */}
            <div className="results-summary">
                <h3 className="results-title" style={{ marginBottom: 'var(--spacing-md)' }}>
                    Analyzed Content
                </h3>
                <div style={{
                    padding: 'var(--spacing-md)',
                    backgroundColor: 'var(--background)',
                    borderRadius: 'var(--radius-md)',
                    fontFamily: 'monospace',
                    fontSize: 'var(--font-size-sm)',
                    wordBreak: 'break-all'
                }}>
                    {input}
                </div>
            </div>

            {/* Indicators */}
            <div className="results-list">
                <h3 className="results-title" style={{ marginBottom: 'var(--spacing-lg)' }}>
                    {isPhishing ? 'Threat Indicators Found' : 'Analysis Details'}
                </h3>

                <ul className="indicators-list">
                    {indicators.map((indicator, index) => (
                        <li key={index} style={{
                            borderBottom: indicator.startsWith('   └─') ? 'none' : undefined,
                            paddingTop: indicator.startsWith('   └─') ? '0' : undefined,
                            paddingLeft: indicator.startsWith('   └─') ? 'var(--spacing-lg)' : undefined
                        }}>
                            {!indicator.startsWith('   └─') && (
                                <span className="indicator-icon">
                                    {indicator.includes('✅') ? '✅' : indicator.includes('🛡️') ? '🛡️' : indicator.includes('🦠') ? '🦠' : isPhishing ? '⚠' : '•'}
                                </span>
                            )}
                            <span style={{ color: indicator.includes('✅') ? 'var(--accent)' : undefined }}>
                                {indicator.replace(/^   └─ /, '')}
                            </span>
                        </li>
                    ))}
                </ul>
            </div>

            {/* Tips */}
            <div className="results-list" style={{ marginTop: 'var(--spacing-xl)' }}>
                <h3 className="results-title" style={{ marginBottom: 'var(--spacing-lg)' }}>
                    Safety Tips
                </h3>
                <ul style={{ paddingLeft: 'var(--spacing-lg)' }}>
                    <li style={{ marginBottom: 'var(--spacing-sm)', color: 'var(--text-secondary)' }}>
                        Always verify the sender's identity through official channels
                    </li>
                    <li style={{ marginBottom: 'var(--spacing-sm)', color: 'var(--text-secondary)' }}>
                        Never click on suspicious links in emails or messages
                    </li>
                    <li style={{ marginBottom: 'var(--spacing-sm)', color: 'var(--text-secondary)' }}>
                        Check for HTTPS and valid SSL certificates on websites
                    </li>
                    <li style={{ marginBottom: 'var(--spacing-sm)', color: 'var(--text-secondary)' }}>
                        Be wary of urgent requests for personal information
                    </li>
                    <li style={{ color: 'var(--text-secondary)' }}>
                        Report phishing attempts to relevant authorities
                    </li>
                </ul>
            </div>
        </div>
    );
};

export default PhishingResults;
