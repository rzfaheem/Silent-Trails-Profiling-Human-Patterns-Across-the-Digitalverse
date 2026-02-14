import React, { useState, useEffect, useMemo } from 'react';
import { useAuth } from '../context/AuthContext';
import { supabase } from '../lib/supabase';
import './Timeline.css';

const Timeline = () => {
    const { user } = useAuth();
    const [investigations, setInvestigations] = useState([]);
    const [selectedInvestigation, setSelectedInvestigation] = useState(null);
    const [events, setEvents] = useState([]);
    const [loading, setLoading] = useState(false);
    const [showNewModal, setShowNewModal] = useState(false);
    const [newInvestigation, setNewInvestigation] = useState({ name: '', target: '', description: '' });
    const [viewMode, setViewMode] = useState('timeline'); // 'timeline' or 'grouped'
    const [activeTab, setActiveTab] = useState('investigations'); // 'investigations' or 'timeline'
    const [error, setError] = useState(null);

    // ========== FILTER STATE ==========
    const [searchQuery, setSearchQuery] = useState('');
    const [filterSeverity, setFilterSeverity] = useState([]); // multi-select
    const [filterType, setFilterType] = useState([]); // multi-select
    const [filterSource, setFilterSource] = useState([]); // multi-select

    // Fetch user's investigations on mount
    useEffect(() => {
        if (user) fetchInvestigations();
    }, [user]);

    const fetchInvestigations = async () => {
        try {
            const { data, error } = await supabase
                .from('investigations')
                .select('*, scans(count)')
                .eq('user_id', user.id)
                .order('created_at', { ascending: false });

            if (error) throw error;
            setInvestigations(data || []);
        } catch (err) {
            console.error('Error fetching investigations:', err);
            setError(err.message);
        }
    };

    const createInvestigation = async () => {
        if (!newInvestigation.name.trim() || !newInvestigation.target.trim()) {
            setError('Name and target are required');
            return;
        }

        try {
            setLoading(true);
            const { data, error } = await supabase
                .from('investigations')
                .insert({
                    user_id: user.id,
                    name: newInvestigation.name,
                    target: newInvestigation.target,
                    description: newInvestigation.description
                })
                .select()
                .single();

            if (error) throw error;

            setInvestigations(prev => [data, ...prev]);
            setShowNewModal(false);
            setNewInvestigation({ name: '', target: '', description: '' });
            setError(null);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const selectInvestigation = async (investigation) => {
        setSelectedInvestigation(investigation);
        setActiveTab('timeline');
        await fetchTimelineEvents(investigation.id);
    };

    const fetchTimelineEvents = async (investigationId) => {
        try {
            setLoading(true);
            const { data, error } = await supabase
                .from('events')
                .select('*, scans!inner(investigation_id, scan_type, target)')
                .eq('scans.investigation_id', investigationId)
                .order('timestamp', { ascending: true, nullsFirst: false });

            if (error) throw error;
            setEvents(data || []);
        } catch (err) {
            console.error('Error fetching events:', err);
            setEvents([]);
        } finally {
            setLoading(false);
        }
    };

    const deleteInvestigation = async (id, e) => {
        e.stopPropagation();
        if (!window.confirm('Delete this investigation and all its data?')) return;

        try {
            const { error } = await supabase
                .from('investigations')
                .delete()
                .eq('id', id);

            if (error) throw error;
            setInvestigations(prev => prev.filter(inv => inv.id !== id));
            if (selectedInvestigation?.id === id) {
                setSelectedInvestigation(null);
                setEvents([]);
                setActiveTab('investigations');
            }
        } catch (err) {
            setError(err.message);
        }
    };

    // ========== COMPUTED DATA ==========

    // Group events by type
    const groupedEvents = events.reduce((acc, event) => {
        const type = event.event_type || 'unknown';
        if (!acc[type]) acc[type] = [];
        acc[type].push(event);
        return acc;
    }, {});

    // Detect activity spikes (anomaly detection)
    const detectAnomalies = () => {
        if (events.length < 3) return [];

        const dateGroups = {};
        events.forEach(event => {
            if (event.timestamp) {
                const date = new Date(event.timestamp).toISOString().split('T')[0];
                dateGroups[date] = (dateGroups[date] || 0) + 1;
            }
        });

        const counts = Object.values(dateGroups);
        const avg = counts.reduce((a, b) => a + b, 0) / counts.length;
        const threshold = avg * 2;

        return Object.entries(dateGroups)
            .filter(([_, count]) => count > threshold)
            .map(([date, count]) => ({ date, count, avg: Math.round(avg) }));
    };

    const anomalies = detectAnomalies();

    // ========== RISK SCORE COMPUTATION ==========
    const riskAnalysis = useMemo(() => {
        if (events.length === 0) return null;

        // Severity counts
        const severity = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
        const sources = {};
        const types = { identity: 0, security: 0, infrastructure: 0, unknown: 0 };

        // Infrastructure intel
        const domains = new Set();
        const ipAddresses = new Set();
        const emails = new Set();
        const accounts = new Set();
        const leaks = [];
        const phishingEvents = [];

        events.forEach(event => {
            // Count severities
            const sev = event.severity || 'info';
            if (severity[sev] !== undefined) severity[sev]++;
            else severity.info++;

            // Count sources
            const src = event.source || 'Unknown';
            sources[src] = (sources[src] || 0) + 1;

            // Count types
            const type = event.event_type || 'unknown';
            if (types[type] !== undefined) types[type]++;
            else types.unknown++;

            // Extract infrastructure data from titles
            const title = event.title || '';
            const titleLower = title.toLowerCase();

            if (titleLower.includes('domain') || titleLower.includes('subdomain')) {
                const domainMatch = title.match(/:\s*(.+)/);
                if (domainMatch) domains.add(domainMatch[1].trim());
            }

            if (titleLower.includes('ip address') || titleLower.includes('ip found')) {
                const ipMatch = title.match(/:\s*(.+)/);
                if (ipMatch) ipAddresses.add(ipMatch[1].trim());
            }

            if (titleLower.includes('email')) {
                const emailMatch = title.match(/:\s*(.+)/);
                if (emailMatch) emails.add(emailMatch[1].trim());
            }

            if (titleLower.includes('account') || titleLower.includes('username') || titleLower.includes('profile')) {
                const acctMatch = title.match(/:\s*(.+)/);
                if (acctMatch) accounts.add(acctMatch[1].trim());
            }

            if (titleLower.includes('leak') || titleLower.includes('breach') || titleLower.includes('credential')) {
                leaks.push(event);
            }

            if (titleLower.includes('phishing') || titleLower.includes('malicious') || titleLower.includes('suspicious url')) {
                phishingEvents.push(event);
            }
        });

        // Compute risk score (0-100)
        let riskScore = 0;
        riskScore += severity.critical * 25;
        riskScore += severity.high * 15;
        riskScore += severity.medium * 5;
        riskScore += severity.low * 1;
        riskScore += leaks.length * 10;
        riskScore += phishingEvents.length * 8;
        riskScore = Math.min(100, riskScore);

        const riskLevel = riskScore >= 70 ? 'critical' : riskScore >= 45 ? 'high' : riskScore >= 20 ? 'medium' : 'low';

        // Date range
        const timestamps = events
            .filter(e => e.timestamp)
            .map(e => new Date(e.timestamp).getTime());
        const dateRange = timestamps.length > 0
            ? { from: new Date(Math.min(...timestamps)), to: new Date(Math.max(...timestamps)) }
            : null;

        // Auto-generate intelligence findings
        const findings = [];
        if (domains.size > 0) findings.push(`${domains.size} linked domain${domains.size > 1 ? 's' : ''} discovered`);
        if (ipAddresses.size > 0) findings.push(`${ipAddresses.size} IP address${ipAddresses.size > 1 ? 'es' : ''} identified`);
        if (emails.size > 0) findings.push(`${emails.size} email address${emails.size > 1 ? 'es' : ''} found`);
        if (accounts.size > 0) findings.push(`${accounts.size} social account${accounts.size > 1 ? 's' : ''} linked`);
        if (leaks.length > 0) findings.push(`${leaks.length} credential leak${leaks.length > 1 ? 's' : ''} detected`);
        if (phishingEvents.length > 0) findings.push(`${phishingEvents.length} phishing/malicious indicator${phishingEvents.length > 1 ? 's' : ''} flagged`);

        return {
            riskScore,
            riskLevel,
            severity,
            sources,
            types,
            dateRange,
            findings,
            infrastructure: {
                domains: [...domains],
                ipAddresses: [...ipAddresses],
                emails: [...emails],
                accounts: [...accounts],
            },
            leaks,
            phishingEvents,
            totalEvents: events.length,
        };
    }, [events]);

    // ========== FILTERED EVENTS ==========
    const filteredEvents = useMemo(() => {
        return events.filter(event => {
            // Search filter
            if (searchQuery) {
                const q = searchQuery.toLowerCase();
                const matchesSearch =
                    (event.title || '').toLowerCase().includes(q) ||
                    (event.description || '').toLowerCase().includes(q) ||
                    (event.source || '').toLowerCase().includes(q);
                if (!matchesSearch) return false;
            }

            // Severity filter
            if (filterSeverity.length > 0 && !filterSeverity.includes(event.severity || 'info')) {
                return false;
            }

            // Type filter
            if (filterType.length > 0 && !filterType.includes(event.event_type || 'unknown')) {
                return false;
            }

            // Source filter
            if (filterSource.length > 0 && !filterSource.includes(event.source || 'Unknown')) {
                return false;
            }

            return true;
        });
    }, [events, searchQuery, filterSeverity, filterType, filterSource]);

    // Filtered grouped events
    const filteredGroupedEvents = filteredEvents.reduce((acc, event) => {
        const type = event.event_type || 'unknown';
        if (!acc[type]) acc[type] = [];
        acc[type].push(event);
        return acc;
    }, {});

    // Get unique sources for filter
    const availableSources = useMemo(() => {
        const s = new Set();
        events.forEach(e => s.add(e.source || 'Unknown'));
        return [...s];
    }, [events]);

    // ========== TOGGLE FILTERS ==========
    const toggleFilter = (setter, value) => {
        setter(prev => prev.includes(value) ? prev.filter(v => v !== value) : [...prev, value]);
    };

    const clearFilters = () => {
        setSearchQuery('');
        setFilterSeverity([]);
        setFilterType([]);
        setFilterSource([]);
    };

    const hasActiveFilters = searchQuery || filterSeverity.length > 0 || filterType.length > 0 || filterSource.length > 0;

    // ========== HELPERS ==========
    const getSeverityColor = (severity) => {
        switch (severity) {
            case 'critical': return '#ff4444';
            case 'high': return '#ff8800';
            case 'medium': return '#ffcc00';
            case 'low': return '#00eb87';
            default: return '#8b9a8f';
        }
    };

    const getRiskColor = (level) => {
        switch (level) {
            case 'critical': return '#ff4444';
            case 'high': return '#ff8800';
            case 'medium': return '#ffcc00';
            case 'low': return '#00eb87';
            default: return '#8b9a8f';
        }
    };

    const getTypeIcon = (type) => {
        switch (type) {
            case 'identity': return '👤';
            case 'security': return '🔒';
            case 'infrastructure': return '🌐';
            default: return '📋';
        }
    };

    const getTypeLabel = (type) => {
        switch (type) {
            case 'identity': return 'Identity & Social';
            case 'security': return 'Security Events';
            case 'infrastructure': return 'Infrastructure';
            default: return 'Other';
        }
    };

    const formatDate = (dateStr) => {
        if (!dateStr) return 'Unknown Date';
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', {
            year: 'numeric', month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit'
        });
    };

    const formatShortDate = (dateStr) => {
        if (!dateStr) return '—';
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    };

    const exportJSON = () => {
        const exportData = {
            investigation: selectedInvestigation,
            riskAnalysis,
            events: events,
            anomalies: anomalies,
            exportedAt: new Date().toISOString(),
            generatedBy: 'Silent Trails Timeline Engine'
        };
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `timeline_${selectedInvestigation?.name?.replace(/\s/g, '_') || 'export'}.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    // ========== RENDER ==========
    return (
        <div className="timeline-page">
            {/* Header */}
            <div className="tl-header">
                <div className="tl-header-icon">⏳</div>
                <div className="tl-header-info">
                    <h1>Timeline Reconstruction Engine</h1>
                    <p>Transform fragmented OSINT data into meaningful chronological narratives</p>
                </div>
            </div>

            {/* Tabs */}
            <div className="tl-tabs">
                <button
                    className={`tl-tab ${activeTab === 'investigations' ? 'active' : ''}`}
                    onClick={() => setActiveTab('investigations')}
                >
                    📋 Investigations
                </button>
                <button
                    className={`tl-tab ${activeTab === 'timeline' ? 'active' : ''}`}
                    onClick={() => setActiveTab('timeline')}
                    disabled={!selectedInvestigation}
                >
                    ⏳ Timeline View
                </button>
            </div>

            {error && (
                <div className="tl-error">⚠️ {error}</div>
            )}

            {/* ==================== INVESTIGATIONS TAB ==================== */}
            {activeTab === 'investigations' && (
                <div className="tl-investigations">
                    <div className="tl-section-header">
                        <h2>Your Investigations</h2>
                        <button className="tl-new-btn" onClick={() => setShowNewModal(true)}>
                            + New Investigation
                        </button>
                    </div>

                    {investigations.length === 0 ? (
                        <div className="tl-empty">
                            <span className="tl-empty-icon">🔍</span>
                            <h3>No Investigations Yet</h3>
                            <p>Create your first investigation to start building timelines from your OSINT data.</p>
                            <button className="tl-new-btn" onClick={() => setShowNewModal(true)}>
                                + Create Investigation
                            </button>
                        </div>
                    ) : (
                        <div className="tl-inv-grid">
                            {investigations.map(inv => (
                                <div
                                    key={inv.id}
                                    className={`tl-inv-card ${selectedInvestigation?.id === inv.id ? 'selected' : ''}`}
                                    onClick={() => selectInvestigation(inv)}
                                >
                                    <div className="tl-inv-card-header">
                                        <h3>{inv.name}</h3>
                                        <span className={`tl-status ${inv.status}`}>{inv.status}</span>
                                    </div>
                                    <div className="tl-inv-target">🎯 {inv.target}</div>
                                    {inv.description && (
                                        <p className="tl-inv-desc">{inv.description}</p>
                                    )}
                                    <div className="tl-inv-footer">
                                        <span className="tl-inv-date">
                                            {formatDate(inv.created_at)}
                                        </span>
                                        <button
                                            className="tl-delete-btn"
                                            onClick={(e) => deleteInvestigation(inv.id, e)}
                                        >
                                            🗑️
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* ==================== TIMELINE TAB ==================== */}
            {activeTab === 'timeline' && selectedInvestigation && (
                <div className="tl-timeline-view">
                    {/* Investigation Info Bar */}
                    <div className="tl-info-bar">
                        <div className="tl-info-left">
                            <h2>{selectedInvestigation.name}</h2>
                            <span className="tl-info-target">🎯 {selectedInvestigation.target}</span>
                        </div>
                        <div className="tl-info-right">
                            <div className="tl-info-stat">
                                <span className="tl-info-number">{events.length}</span>
                                <span className="tl-info-label">Events</span>
                            </div>
                            <div className="tl-info-stat">
                                <span className="tl-info-number">{anomalies.length}</span>
                                <span className="tl-info-label">Anomalies</span>
                            </div>
                            <button className="tl-export-btn" onClick={exportJSON} disabled={events.length === 0}>
                                📥 Export JSON
                            </button>
                        </div>
                    </div>

                    {/* ======== STATS DASHBOARD ======== */}
                    {riskAnalysis && (
                        <div className="tl-dashboard">
                            {/* Risk Score */}
                            <div className="tl-risk-card">
                                <div className="tl-risk-gauge">
                                    <svg viewBox="0 0 120 120" className="tl-risk-svg">
                                        <circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
                                        <circle
                                            cx="60" cy="60" r="52" fill="none"
                                            stroke={getRiskColor(riskAnalysis.riskLevel)}
                                            strokeWidth="8"
                                            strokeLinecap="round"
                                            strokeDasharray={`${(riskAnalysis.riskScore / 100) * 327} 327`}
                                            transform="rotate(-90 60 60)"
                                            style={{ transition: 'stroke-dasharray 1s ease' }}
                                        />
                                    </svg>
                                    <div className="tl-risk-value">
                                        <span className="tl-risk-number" style={{ color: getRiskColor(riskAnalysis.riskLevel) }}>
                                            {riskAnalysis.riskScore}
                                        </span>
                                        <span className="tl-risk-label">RISK</span>
                                    </div>
                                </div>
                                <div className="tl-risk-level" style={{ color: getRiskColor(riskAnalysis.riskLevel) }}>
                                    {riskAnalysis.riskLevel.toUpperCase()} THREAT
                                </div>
                                {riskAnalysis.dateRange && (
                                    <div className="tl-risk-daterange">
                                        📅 {formatShortDate(riskAnalysis.dateRange.from)} — {formatShortDate(riskAnalysis.dateRange.to)}
                                    </div>
                                )}
                            </div>

                            {/* Severity Breakdown */}
                            <div className="tl-stats-card">
                                <h4>📊 Severity Breakdown</h4>
                                <div className="tl-severity-bars">
                                    {Object.entries(riskAnalysis.severity).map(([sev, count]) => (
                                        count > 0 && (
                                            <div key={sev} className="tl-sev-row">
                                                <span className="tl-sev-badge" style={{
                                                    background: `${getSeverityColor(sev)}20`,
                                                    color: getSeverityColor(sev)
                                                }}>
                                                    {sev.toUpperCase()}
                                                </span>
                                                <div className="tl-sev-bar-track">
                                                    <div className="tl-sev-bar-fill" style={{
                                                        width: `${Math.min(100, (count / riskAnalysis.totalEvents) * 100)}%`,
                                                        background: getSeverityColor(sev)
                                                    }} />
                                                </div>
                                                <span className="tl-sev-count">{count}</span>
                                            </div>
                                        )
                                    ))}
                                </div>
                            </div>

                            {/* Source Distribution */}
                            <div className="tl-stats-card">
                                <h4>📡 Data Sources</h4>
                                <div className="tl-source-list">
                                    {Object.entries(riskAnalysis.sources).map(([source, count]) => (
                                        <div key={source} className="tl-source-item">
                                            <span className="tl-source-icon">
                                                {source === 'SpiderFoot' ? '🕷️' : source === 'LeakCheck' ? '🔓' : source === 'VirusTotal' ? '🛡️' : '📡'}
                                            </span>
                                            <span className="tl-source-name">{source}</span>
                                            <span className="tl-source-count">{count}</span>
                                        </div>
                                    ))}
                                </div>

                                {/* Type Distribution */}
                                <div className="tl-type-dist">
                                    {Object.entries(riskAnalysis.types).map(([type, count]) => (
                                        count > 0 && (
                                            <div key={type} className="tl-type-chip">
                                                {getTypeIcon(type)} {getTypeLabel(type)}
                                                <span>{count}</span>
                                            </div>
                                        )
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* ======== INTELLIGENCE SUMMARY ======== */}
                    {riskAnalysis && riskAnalysis.findings.length > 0 && (
                        <div className="tl-intel-summary">
                            <h4>🧠 Intelligence Summary</h4>
                            <div className="tl-intel-findings">
                                {riskAnalysis.findings.map((finding, i) => (
                                    <div key={i} className="tl-intel-item">
                                        <span className="tl-intel-bullet">▸</span>
                                        <span>{finding}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* ======== INFRASTRUCTURE PANEL ======== */}
                    {riskAnalysis && (
                        riskAnalysis.infrastructure.domains.length > 0 ||
                        riskAnalysis.infrastructure.ipAddresses.length > 0 ||
                        riskAnalysis.infrastructure.emails.length > 0 ||
                        riskAnalysis.infrastructure.accounts.length > 0
                    ) && (
                            <div className="tl-infra-panel">
                                <h4>🌐 Infrastructure & Digital Footprint</h4>
                                <div className="tl-infra-grid">
                                    {/* Domains */}
                                    {riskAnalysis.infrastructure.domains.length > 0 && (
                                        <div className="tl-infra-section">
                                            <div className="tl-infra-section-header">
                                                <span>🔗 Domains</span>
                                                <span className="tl-infra-count">{riskAnalysis.infrastructure.domains.length}</span>
                                            </div>
                                            <div className="tl-infra-items">
                                                {riskAnalysis.infrastructure.domains.map((domain, i) => (
                                                    <div key={i} className="tl-infra-item domain">
                                                        <span className="tl-infra-tree-icon">├──</span>
                                                        <span>{domain}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* IP Addresses */}
                                    {riskAnalysis.infrastructure.ipAddresses.length > 0 && (
                                        <div className="tl-infra-section">
                                            <div className="tl-infra-section-header">
                                                <span>🖥️ IP Addresses</span>
                                                <span className="tl-infra-count">{riskAnalysis.infrastructure.ipAddresses.length}</span>
                                            </div>
                                            <div className="tl-infra-items">
                                                {riskAnalysis.infrastructure.ipAddresses.map((ip, i) => (
                                                    <div key={i} className="tl-infra-item ip">
                                                        <span className="tl-infra-dot" style={{ background: '#60a5fa' }} />
                                                        <span>{ip}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Emails */}
                                    {riskAnalysis.infrastructure.emails.length > 0 && (
                                        <div className="tl-infra-section">
                                            <div className="tl-infra-section-header">
                                                <span>📧 Email Addresses</span>
                                                <span className="tl-infra-count">{riskAnalysis.infrastructure.emails.length}</span>
                                            </div>
                                            <div className="tl-infra-items">
                                                {riskAnalysis.infrastructure.emails.map((email, i) => (
                                                    <div key={i} className="tl-infra-item email">
                                                        <span className="tl-infra-dot" style={{ background: '#f59e0b' }} />
                                                        <span>{email}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Accounts */}
                                    {riskAnalysis.infrastructure.accounts.length > 0 && (
                                        <div className="tl-infra-section">
                                            <div className="tl-infra-section-header">
                                                <span>👤 Linked Accounts</span>
                                                <span className="tl-infra-count">{riskAnalysis.infrastructure.accounts.length}</span>
                                            </div>
                                            <div className="tl-infra-items">
                                                {riskAnalysis.infrastructure.accounts.map((acct, i) => (
                                                    <div key={i} className="tl-infra-item account">
                                                        <span className="tl-infra-dot" style={{ background: '#a78bfa' }} />
                                                        <span>{acct}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                    {/* ======== ANOMALY ALERTS ======== */}
                    {anomalies.length > 0 && (
                        <div className="tl-anomalies">
                            <h3>🚨 Activity Anomalies Detected</h3>
                            <div className="tl-anomaly-list">
                                {anomalies.map((a, i) => (
                                    <div key={i} className="tl-anomaly-item">
                                        <span className="tl-anomaly-date">{a.date}</span>
                                        <span className="tl-anomaly-count">{a.count} events</span>
                                        <span className="tl-anomaly-note">({a.avg} avg/day)</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* ======== FILTER BAR ======== */}
                    {events.length > 0 && (
                        <div className="tl-filter-bar">
                            <div className="tl-filter-search">
                                <span className="tl-filter-search-icon">🔍</span>
                                <input
                                    type="text"
                                    placeholder="Search events..."
                                    value={searchQuery}
                                    onChange={e => setSearchQuery(e.target.value)}
                                />
                            </div>

                            <div className="tl-filter-groups">
                                {/* Severity Filters */}
                                <div className="tl-filter-group">
                                    <span className="tl-filter-label">Severity:</span>
                                    {['critical', 'high', 'medium', 'low', 'info'].map(sev => (
                                        <button
                                            key={sev}
                                            className={`tl-filter-chip ${filterSeverity.includes(sev) ? 'active' : ''}`}
                                            style={{
                                                '--chip-color': getSeverityColor(sev),
                                                borderColor: filterSeverity.includes(sev) ? getSeverityColor(sev) : undefined
                                            }}
                                            onClick={() => toggleFilter(setFilterSeverity, sev)}
                                        >
                                            {sev}
                                        </button>
                                    ))}
                                </div>

                                {/* Type Filters */}
                                <div className="tl-filter-group">
                                    <span className="tl-filter-label">Type:</span>
                                    {['identity', 'security', 'infrastructure'].map(type => (
                                        <button
                                            key={type}
                                            className={`tl-filter-chip ${filterType.includes(type) ? 'active' : ''}`}
                                            onClick={() => toggleFilter(setFilterType, type)}
                                        >
                                            {getTypeIcon(type)} {getTypeLabel(type)}
                                        </button>
                                    ))}
                                </div>

                                {/* Source Filters */}
                                {availableSources.length > 1 && (
                                    <div className="tl-filter-group">
                                        <span className="tl-filter-label">Source:</span>
                                        {availableSources.map(src => (
                                            <button
                                                key={src}
                                                className={`tl-filter-chip ${filterSource.includes(src) ? 'active' : ''}`}
                                                onClick={() => toggleFilter(setFilterSource, src)}
                                            >
                                                {src}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {/* Filter Status */}
                            <div className="tl-filter-status">
                                <span>Showing {filteredEvents.length} of {events.length} events</span>
                                {hasActiveFilters && (
                                    <button className="tl-filter-clear" onClick={clearFilters}>
                                        ✕ Clear Filters
                                    </button>
                                )}
                            </div>
                        </div>
                    )}

                    {/* View Mode Toggle */}
                    <div className="tl-view-toggle">
                        <button
                            className={viewMode === 'timeline' ? 'active' : ''}
                            onClick={() => setViewMode('timeline')}
                        >
                            📅 Chronological
                        </button>
                        <button
                            className={viewMode === 'grouped' ? 'active' : ''}
                            onClick={() => setViewMode('grouped')}
                        >
                            📂 Grouped
                        </button>
                    </div>

                    {/* Loading */}
                    {loading && (
                        <div className="tl-loading">
                            <div className="tl-spinner"></div>
                            <p>Reconstructing timeline...</p>
                        </div>
                    )}

                    {/* Empty State */}
                    {!loading && events.length === 0 && (
                        <div className="tl-empty">
                            <span className="tl-empty-icon">📊</span>
                            <h3>No Events Yet</h3>
                            <p>Run scans in Digital Recon to populate this investigation's timeline. Events from SpiderFoot scans, phishing checks, and breach lookups will appear here.</p>
                        </div>
                    )}

                    {/* No Results After Filter */}
                    {!loading && events.length > 0 && filteredEvents.length === 0 && (
                        <div className="tl-empty">
                            <span className="tl-empty-icon">🔍</span>
                            <h3>No Matching Events</h3>
                            <p>No events match your current filters. Try adjusting or clearing the filters.</p>
                            <button className="tl-new-btn" onClick={clearFilters}>Clear Filters</button>
                        </div>
                    )}

                    {/* ======== CHRONOLOGICAL TIMELINE ======== */}
                    {!loading && filteredEvents.length > 0 && viewMode === 'timeline' && (
                        <div className="tl-timeline">
                            <div className="tl-timeline-line"></div>
                            {filteredEvents.map((event, idx) => (
                                <div key={event.id} className={`tl-event ${idx % 2 === 0 ? 'left' : 'right'}`}>
                                    <div className="tl-event-dot" style={{
                                        borderColor: getSeverityColor(event.severity)
                                    }}>
                                        {getTypeIcon(event.event_type)}
                                    </div>
                                    <div className="tl-event-card">
                                        <div className="tl-event-header">
                                            <span className="tl-event-type" style={{
                                                background: `${getSeverityColor(event.severity)}20`,
                                                color: getSeverityColor(event.severity)
                                            }}>
                                                {event.severity?.toUpperCase() || 'INFO'}
                                            </span>
                                            <span className="tl-event-time">
                                                {event.timestamp_estimated && '~'}
                                                {formatDate(event.timestamp)}
                                            </span>
                                        </div>
                                        <h4 className="tl-event-title">{event.title}</h4>
                                        {event.description && (
                                            <p className="tl-event-desc">{event.description}</p>
                                        )}
                                        <div className="tl-event-meta">
                                            <span>📡 {event.source}</span>
                                            <span>🎯 {Math.round(event.confidence * 100)}% confidence</span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* ======== GROUPED VIEW ======== */}
                    {!loading && filteredEvents.length > 0 && viewMode === 'grouped' && (
                        <div className="tl-grouped">
                            {Object.entries(filteredGroupedEvents).map(([type, typeEvents]) => (
                                <div key={type} className="tl-group">
                                    <div className="tl-group-header">
                                        <span>{getTypeIcon(type)} {getTypeLabel(type)}</span>
                                        <span className="tl-group-count">{typeEvents.length}</span>
                                    </div>
                                    <div className="tl-group-events">
                                        {typeEvents.map(event => (
                                            <div key={event.id} className="tl-group-event">
                                                <div className="tl-group-event-severity"
                                                    style={{ background: getSeverityColor(event.severity) }}
                                                ></div>
                                                <div className="tl-group-event-info">
                                                    <span className="tl-group-event-title">{event.title}</span>
                                                    <span className="tl-group-event-time">{formatDate(event.timestamp)}</span>
                                                </div>
                                                <span className="tl-group-event-source">{event.source}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* ==================== NEW INVESTIGATION MODAL ==================== */}
            {showNewModal && (
                <div className="tl-modal-overlay" onClick={() => setShowNewModal(false)}>
                    <div className="tl-modal" onClick={e => e.stopPropagation()}>
                        <h2>New Investigation</h2>
                        <div className="tl-modal-form">
                            <div className="tl-form-group">
                                <label>Investigation Name</label>
                                <input
                                    type="text"
                                    value={newInvestigation.name}
                                    onChange={e => setNewInvestigation(prev => ({ ...prev, name: e.target.value }))}
                                    placeholder="e.g., John Doe Case"
                                />
                            </div>
                            <div className="tl-form-group">
                                <label>Target</label>
                                <input
                                    type="text"
                                    value={newInvestigation.target}
                                    onChange={e => setNewInvestigation(prev => ({ ...prev, target: e.target.value }))}
                                    placeholder="e.g., john@email.com or johndoe"
                                />
                            </div>
                            <div className="tl-form-group">
                                <label>Description (optional)</label>
                                <textarea
                                    value={newInvestigation.description}
                                    onChange={e => setNewInvestigation(prev => ({ ...prev, description: e.target.value }))}
                                    placeholder="Brief notes about this investigation..."
                                    rows={3}
                                />
                            </div>
                            <div className="tl-modal-actions">
                                <button className="tl-cancel-btn" onClick={() => setShowNewModal(false)}>
                                    Cancel
                                </button>
                                <button
                                    className="tl-create-btn"
                                    onClick={createInvestigation}
                                    disabled={loading || !newInvestigation.name.trim() || !newInvestigation.target.trim()}
                                >
                                    {loading ? 'Creating...' : 'Create Investigation'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Timeline;
