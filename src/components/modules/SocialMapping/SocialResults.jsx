import React from 'react';

const SocialResults = ({ results }) => {
    if (!results) return null;

    const { username, platforms, totalFound, totalSearched, digitalFootprintScore } = results;

    const foundPlatforms = platforms.filter(p => p.found);
    const notFoundPlatforms = platforms.filter(p => !p.found);

    return (
        <div className="results-section">
            {/* Summary Card */}
            <div className="results-summary">
                <div className="results-header">
                    <h3 className="results-title">Search Results</h3>
                    <span className="tag" style={{ fontSize: 'var(--font-size-sm)' }}>
                        @{username}
                    </span>
                </div>

                <div className="summary-stats">
                    <div className="stat-item">
                        <span className="stat-value" style={{ color: 'var(--accent)' }}>
                            {totalFound}
                        </span>
                        <span className="stat-label">Profiles Found</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-value">
                            {totalSearched}
                        </span>
                        <span className="stat-label">Platforms Searched</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-value">
                            {digitalFootprintScore}%
                        </span>
                        <span className="stat-label">Footprint Score</span>
                    </div>
                </div>

                {/* Footprint Progress Bar */}
                <div style={{ marginTop: 'var(--spacing-lg)' }}>
                    <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--spacing-xs)' }}>
                        Digital Footprint Level
                    </p>
                    <div className="progress-bar">
                        <div
                            className={`progress-bar-fill ${digitalFootprintScore > 70 ? 'high' : digitalFootprintScore > 40 ? 'medium' : 'low'}`}
                            style={{ width: `${digitalFootprintScore}%` }}
                        />
                    </div>
                </div>
            </div>

            {/* Found Platforms */}
            {foundPlatforms.length > 0 && (
                <div className="results-list">
                    <h3 className="results-title" style={{ marginBottom: 'var(--spacing-lg)' }}>
                        Profiles Found ({foundPlatforms.length})
                    </h3>

                    <div className="social-results-grid">
                        {foundPlatforms.map((platform, index) => (
                            <div key={index} className="platform-item">
                                <div className="platform-icon" style={{
                                    backgroundColor: 'rgba(56, 161, 105, 0.1)',
                                    color: 'var(--accent)'
                                }}>
                                    {platform.icon}
                                </div>
                                <div className="platform-info">
                                    <p className="platform-name">{platform.name}</p>
                                    <p className="platform-status found">✓ Profile found</p>
                                    {platform.followers !== null && (
                                        <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)' }}>
                                            ~{platform.followers.toLocaleString()} followers
                                        </p>
                                    )}
                                </div>
                                {platform.url && (
                                    <a
                                        href={platform.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="platform-link"
                                    >
                                        Visit →
                                    </a>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Not Found Platforms */}
            {notFoundPlatforms.length > 0 && (
                <div className="results-list" style={{ marginTop: 'var(--spacing-xl)' }}>
                    <h3 className="results-title" style={{ marginBottom: 'var(--spacing-lg)' }}>
                        Not Found ({notFoundPlatforms.length})
                    </h3>

                    <div className="social-results-grid">
                        {notFoundPlatforms.map((platform, index) => (
                            <div key={index} className="platform-item" style={{ opacity: 0.6 }}>
                                <div className="platform-icon" style={{
                                    backgroundColor: 'var(--background)',
                                    color: 'var(--text-muted)'
                                }}>
                                    {platform.icon}
                                </div>
                                <div className="platform-info">
                                    <p className="platform-name">{platform.name}</p>
                                    <p className="platform-status not-found">No profile found</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Privacy Note */}
            <div className="results-list" style={{ marginTop: 'var(--spacing-xl)', backgroundColor: 'rgba(49, 130, 206, 0.05)' }}>
                <h3 className="results-title" style={{ marginBottom: 'var(--spacing-md)', color: 'var(--info)' }}>
                    ℹ️ Privacy Note
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)', marginBottom: 0 }}>
                    This analysis only searches for publicly available profiles. Private or protected accounts
                    may not be detected. The results are for informational purposes and should be used responsibly.
                </p>
            </div>
        </div>
    );
};

export default SocialResults;
