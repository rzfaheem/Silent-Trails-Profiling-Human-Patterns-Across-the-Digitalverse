import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../context/AuthContext';
import { supabase } from '../lib/supabase';
import Loader from '../components/common/Loader';
import './SocialMapping.css';

const BACKEND_URL = 'http://localhost:5000';

// Persist scan state across navigation via sessionStorage
const loadSession = (key, fallback) => {
    try {
        const val = sessionStorage.getItem(`st_recon_${key}`);
        return val ? JSON.parse(val) : fallback;
    } catch { return fallback; }
};
const saveSession = (key, val) => {
    try { sessionStorage.setItem(`st_recon_${key}`, JSON.stringify(val)); } catch { }
};

const SocialMapping = () => {
    const { user } = useAuth();

    // ========== SpiderFoot Data Type Categories ==========
    const DATA_TYPE_GROUPS = {
        identity: {
            label: 'Identity & Social',
            icon: '👤',
            types: [
                { id: 'EMAILADDR', label: 'Email Addresses', desc: 'Discover associated email accounts' },
                { id: 'PHONE_NUMBER', label: 'Phone Numbers', desc: 'Find linked phone numbers' },
                { id: 'SOCIAL_MEDIA', label: 'Social Media', desc: 'Social media profiles & activity' },
                { id: 'USERNAME', label: 'Usernames', desc: 'Linked usernames across platforms' },
                { id: 'HUMAN_NAME', label: 'Human Names', desc: 'Real names associated with target' },
                { id: 'AFFILIATE_INTERNET_NAME', label: 'Affiliations', desc: 'Related organizations & entities' }
            ]
        },
        infrastructure: {
            label: 'Infrastructure',
            icon: '🌐',
            types: [
                { id: 'INTERNET_NAME', label: 'Domains', desc: 'Associated domain names' },
                { id: 'IP_ADDRESS', label: 'IP Addresses', desc: 'Resolved IP addresses' },
                { id: 'DNS_TEXT', label: 'DNS Records', desc: 'DNS configuration data' },
                { id: 'WEBSERVER_BANNER', label: 'Web Servers', desc: 'Server technology & versions' },
                { id: 'SSL_CERTIFICATE_RAW', label: 'SSL Certificates', desc: 'Certificate details & validity' },
                { id: 'TCP_PORT_OPEN', label: 'Open Ports', desc: 'Discovered open network ports' }
            ]
        },
        security: {
            label: 'Security & Threats',
            icon: '🔒',
            types: [
                { id: 'VULNERABILITY_CVE_CRITICAL', label: 'Vulnerabilities', desc: 'Known CVEs and exploits' },
                { id: 'MALICIOUS_IPADDR', label: 'Malicious IPs', desc: 'Flagged malicious IP addresses' },
                { id: 'DARKNET_MENTION_URL', label: 'Dark Web Mentions', desc: 'References on the dark web' },
                { id: 'BLACKLISTED_IPADDR', label: 'Blacklisted', desc: 'Blacklisted IPs & domains' },
                { id: 'LEAKSITE_URL', label: 'Data Breaches', desc: 'Known data breach exposure' },
                { id: 'MALICIOUS_AFFILIATE_IPADDR', label: 'Threat Intel', desc: 'Threat intelligence indicators' }
            ]
        }
    };

    const PRESET_CATEGORIES = [
        { id: 'all', label: 'All', icon: '🔍', desc: 'Complete deep scan — all modules', speed: 'Slowest' },
        { id: 'footprint', label: 'Footprint', icon: '👣', desc: 'What info is exposed online', speed: 'Medium' },
        { id: 'investigate', label: 'Investigate', icon: '🔬', desc: 'Deep dive on suspicious targets', speed: 'Medium' },
        { id: 'passive', label: 'Passive', icon: '🕵️', desc: 'Stealth — no direct contact', speed: 'Fastest' }
    ];

    // Core scan state — restored from sessionStorage on mount
    const [target, setTarget] = useState(() => loadSession('target', ''));
    const [scanType, setScanType] = useState(() => loadSession('scanTypePreset', 'all'));
    const [scanConfigMode, setScanConfigMode] = useState(() => loadSession('scanConfigMode', 'preset'));
    const [selectedTypes, setSelectedTypes] = useState(() => loadSession('selectedTypes', []));
    const [loading, setLoading] = useState(false);
    const [scanId, setScanId] = useState(() => loadSession('scanId', null));
    const [scanStatus, setScanStatus] = useState(() => loadSession('scanStatus', null));
    const [results, setResults] = useState(null);
    const [phishingResults, setPhishingResults] = useState(null);
    const [scanMode, setScanMode] = useState(() => loadSession('scanMode', null));
    const [error, setError] = useState(null);
    const [breachResults, setBreachResults] = useState(() => loadSession('breachResults', null));
    const [previousScans, setPreviousScans] = useState([]);
    const [activeTab, setActiveTab] = useState('new');
    const pollingRef = useRef(null);

    // Investigation state
    const [investigations, setInvestigations] = useState([]);
    const [selectedInvestigation, setSelectedInvestigation] = useState(() => loadSession('invId', ''));
    const [showNewInv, setShowNewInv] = useState(false);
    const [newInvName, setNewInvName] = useState('');
    const [savingToTimeline, setSavingToTimeline] = useState(false);
    const [savedToTimeline, setSavedToTimeline] = useState(false);

    // ========== Persist key state to sessionStorage ==========
    useEffect(() => { saveSession('target', target); }, [target]);
    useEffect(() => { saveSession('scanId', scanId); }, [scanId]);
    useEffect(() => { saveSession('scanStatus', scanStatus); }, [scanStatus]);
    useEffect(() => { saveSession('scanMode', scanMode); }, [scanMode]);
    useEffect(() => { saveSession('invId', selectedInvestigation); }, [selectedInvestigation]);
    useEffect(() => { saveSession('breachResults', breachResults); }, [breachResults]);
    useEffect(() => { saveSession('scanConfigMode', scanConfigMode); }, [scanConfigMode]);
    useEffect(() => { saveSession('selectedTypes', selectedTypes); }, [selectedTypes]);
    useEffect(() => { saveSession('scanTypePreset', scanType); }, [scanType]);

    // ========== Data type toggle helpers ==========
    const toggleType = (typeId) => {
        setSelectedTypes(prev =>
            prev.includes(typeId) ? prev.filter(t => t !== typeId) : [...prev, typeId]
        );
    };

    const toggleGroup = (groupKey) => {
        const groupTypeIds = DATA_TYPE_GROUPS[groupKey].types.map(t => t.id);
        const allSelected = groupTypeIds.every(id => selectedTypes.includes(id));
        if (allSelected) {
            setSelectedTypes(prev => prev.filter(t => !groupTypeIds.includes(t)));
        } else {
            setSelectedTypes(prev => [...new Set([...prev, ...groupTypeIds])]);
        }
    };

    const selectAllTypes = () => {
        const allIds = Object.values(DATA_TYPE_GROUPS).flatMap(g => g.types.map(t => t.id));
        setSelectedTypes(allIds);
    };

    const clearAllTypes = () => setSelectedTypes([]);

    // ========== Helpers ==========
    const isURL = (input) => {
        const trimmed = input.trim();
        return trimmed.startsWith('http://') || trimmed.startsWith('https://');
    };

    // ========== AUTO-CLEAR on EMPTY TARGET ==========
    useEffect(() => {
        if (!target) {
            setResults(null);
            setBreachResults(null);
            setPhishingResults(null);
            setScanStatus(null); // Reset scan status if target cleared
            setScanId(null);
            setLoading(false);
        }
    }, [target]);

    // ========== INIT ==========
    useEffect(() => {
        if (user) fetchInvestigations();
        fetchPreviousScans();
    }, [user]);

    // ========== AUTO-RESUME on mount ==========
    useEffect(() => {
        const savedId = loadSession('scanId', null);
        const savedStatus = loadSession('scanStatus', null);

        if (savedId && savedStatus && (savedStatus === 'RUNNING' || savedStatus === 'STARTING')) {
            console.log('[Resume] Re-attaching to running scan:', savedId);
            setScanMode(loadSession('scanMode', 'osint'));
            setLoading(true);
            // Polling will start via the [scanId, loading] effect below
        }
    }, []); // Only runs once on mount

    // ========== POLLING ==========
    useEffect(() => {
        // Clear any old interval
        if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
        }

        if (scanId && loading) {
            const poll = () => checkScanStatus(scanId);
            poll(); // Immediate first check
            pollingRef.current = setInterval(poll, 5000);
        }

        return () => {
            if (pollingRef.current) {
                clearInterval(pollingRef.current);
                pollingRef.current = null;
            }
        };
    }, [scanId, loading]);

    // ========== INVESTIGATIONS ==========
    const fetchInvestigations = async () => {
        try {
            const { data, error } = await supabase
                .from('investigations')
                .select('*')
                .eq('user_id', user.id)
                .order('created_at', { ascending: false });
            if (error) throw error;
            setInvestigations(data || []);
        } catch (err) {
            console.error('Failed to fetch investigations:', err);
        }
    };

    const createQuickInvestigation = async () => {
        if (!newInvName.trim()) return;
        try {
            const { data, error } = await supabase
                .from('investigations')
                .insert({
                    user_id: user.id,
                    name: newInvName.trim(),
                    target: target.trim() || newInvName.trim(),
                    description: 'Created from Digital Recon scan'
                })
                .select()
                .single();
            if (error) throw error;
            setInvestigations(prev => [data, ...prev]);
            setSelectedInvestigation(data.id);
            setShowNewInv(false);
            setNewInvName('');
        } catch (err) {
            console.error('Failed to create investigation:', err);
        }
    };

    // ========== TIMELINE SAVE ==========
    const saveToTimeline = async (type, scanTarget, scanResults) => {
        if (!selectedInvestigation) {
            setError('Select an investigation first to sync to Timeline.');
            return;
        }

        try {
            setSavingToTimeline(true);

            // Get session token for authentication
            const { data: { session } } = await supabase.auth.getSession();
            const token = session?.access_token;

            if (!token) {
                throw new Error('User session expired. Please login again.');
            }

            const response = await fetch(`${BACKEND_URL}/api/save-scan`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    investigationId: selectedInvestigation,
                    scanType: type,
                    target: scanTarget,
                    results: scanResults,
                    userId: user.id
                })
            });

            const data = await response.json();

            if (data.success) {
                setSavedToTimeline(true);
                setTimeout(() => setSavedToTimeline(false), 5000);
            } else {
                throw new Error(data.message || data.error || 'Save failed');
            }
        } catch (err) {
            console.error('Failed to save to timeline:', err);
            setError(err.message || 'Failed to save to timeline. Use the Sync button to retry.');
        } finally {
            setSavingToTimeline(false);
        }
    };

    // ========== SCAN LIST ==========
    const fetchPreviousScans = async () => {
        try {
            const response = await fetch(`${BACKEND_URL}/api/profile-scans`);
            if (!response.ok) {
                console.error('Failed to fetch scans:', response.status);
                return;
            }
            const data = await response.json();
            setPreviousScans(data.scans || []);
        } catch (err) {
            console.error('Failed to fetch scans:', err);
        }
    };

    // ========== CHECK BREACHES ==========
    const checkBreaches = async (email) => {
        try {
            const response = await fetch(`${BACKEND_URL}/api/check-leak/${encodeURIComponent(email)}`);
            if (!response.ok) {
                console.error('Breach check failed:', response.status);
                return;
            }
            const data = await response.json();
            if (data.success) {
                setBreachResults(data);
                // Auto-save if investigation selected
                if (selectedInvestigation) {
                    saveToTimeline('manual', email, data);
                }
            } else {
                // Determine if we should show "Safe" or just ignore
                setBreachResults({ success: false, message: data.message || 'No breaches found' });
            }
        } catch (err) {
            console.error('Breach check failed:', err);
        }
    };

    // ========== START SCAN ==========
    const startScan = async () => {
        if (!selectedInvestigation) {
            setError('Please select or create an investigation first');
            return;
        }
        if (!target.trim()) {
            setError('Please enter a target');
            return;
        }

        setLoading(true);
        setError(null);
        setResults(null);
        setPhishingResults(null);
        setBreachResults(null);
        setSavedToTimeline(false);
        setScanStatus('STARTING');

        // Check for Email Breach
        if (target.includes('@') && target.includes('.')) {
            checkBreaches(target.trim());
        }

        if (isURL(target)) {
            setScanMode('url');
            try {
                setScanStatus('ANALYZING URL');
                const response = await fetch(`${BACKEND_URL}/api/check-url`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: target.trim() })
                });
                if (!response.ok) {
                    throw new Error(`Server error (${response.status})`);
                }
                const data = await response.json();
                if (data.error) throw new Error(data.message || 'Failed to analyze URL');

                setPhishingResults(data);
                setScanStatus('COMPLETE');
                setLoading(false);
                await saveToTimeline('phishing', target.trim(), data);
            } catch (err) {
                setError(err.message || 'Failed to analyze URL');
                setLoading(false);
            }
        } else {
            setScanMode('osint');
            try {
                const response = await fetch(`${BACKEND_URL}/api/profile-scan`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target: target.trim(), scanType })
                });
                if (!response.ok) {
                    const text = await response.text();
                    throw new Error(`Server error (${response.status}): ${text.substring(0, 100)}`);
                }
                const data = await response.json();
                if (data.error) throw new Error(data.message || 'Failed to start scan');

                setScanId(data.scanId);
                setScanStatus('RUNNING');
                setActiveTab('new');
                // Polling starts automatically via useEffect
            } catch (err) {
                setError(err.message || 'Failed to start scan. Make sure SpiderFoot is running.');
                setLoading(false);
            }
        }
    };

    // ========== STATUS CHECK ==========
    const checkScanStatus = async (id) => {
        try {
            const response = await fetch(`${BACKEND_URL}/api/profile-scan/${id}/status`);
            if (!response.ok) return;
            const data = await response.json();
            const status = data.status || 'RUNNING';
            setScanStatus(status);

            // Fetch intermediate results for real-time updates
            if (status === 'RUNNING' || status === 'STARTING') {
                try {
                    const resultsResponse = await fetch(`${BACKEND_URL}/api/profile-scan/${id}/results`);
                    if (resultsResponse.ok) {
                        const resultsData = await resultsResponse.json();
                        setResults(resultsData);
                    }
                } catch (err) {
                    // Ignore errors during polling
                }
            }

            if (data.finished || status === 'FINISHED' || status === 'ABORTED') {
                // Scan done — stop polling
                setLoading(false);
                if (pollingRef.current) {
                    clearInterval(pollingRef.current);
                    pollingRef.current = null;
                }
                fetchPreviousScans();

                if (status === 'FINISHED') {
                    // Auto-fetch results and display them
                    const resultsResponse = await fetch(`${BACKEND_URL}/api/profile-scan/${id}/results`);
                    if (resultsResponse.ok) {
                        const resultsData = await resultsResponse.json();
                        setResults(resultsData);

                        // Auto-save to timeline if investigation is selected
                        if (selectedInvestigation) {
                            saveToTimeline('spiderfoot', target.trim(), resultsData);
                        }
                    }
                }
            }
        } catch (err) {
            console.error('Status check failed:', err);
        }
    };

    // ========== STOP SCAN ==========
    const stopScan = async () => {
        if (!scanId) return;
        try {
            const response = await fetch(`${BACKEND_URL}/api/profile-scan/${scanId}/stop`, { method: 'POST' });
            const data = await response.json();
            if (data.success) {
                setScanStatus('ABORTED');
                setLoading(false);
                if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
                fetchPreviousScans();
            }
        } catch (err) {
            console.error('Failed to stop scan:', err);
        }
    };

    // ========== FETCH RESULTS ==========
    const fetchResults = async (id) => {
        try {
            setLoading(true);
            const response = await fetch(`${BACKEND_URL}/api/profile-scan/${id}/results`);
            if (!response.ok) {
                throw new Error(`Failed to fetch results (${response.status})`);
            }
            const data = await response.json();
            setResults(data);
            setLoading(false);
        } catch (err) {
            console.error('Failed to fetch results:', err);
            setError('Failed to fetch scan results. Make sure SpiderFoot is running.');
            setLoading(false);
        }
    };

    // ========== LOAD FROM HISTORY ==========
    const loadPreviousScan = async (scan) => {
        setScanId(scan.id);
        setTarget(scan.target || '');
        setScanStatus(scan.status);
        setScanMode('osint'); // Always set this so results render
        setActiveTab('new');
        setResults(null);
        setError(null);
        setSavedToTimeline(false);
        setPhishingResults(null);
        setBreachResults(null);

        if (scan.status === 'FINISHED') {
            await fetchResults(scan.id);
        } else if (scan.status === 'RUNNING' || scan.status === 'STARTING') {
            setLoading(true); // Will trigger polling via useEffect
        } else {
            // ABORTED or other status - still try to fetch partial results
            await fetchResults(scan.id);
        }
    };


    // ========== RENDER ==========
    return (
        <div className="social-mapping-page">
            {/* Header */}
            <div className="sf-header">
                <div className="sf-logo">🔍</div>
                <div className="sf-title">
                    <h1>Digital Recon</h1>
                    <p>Advanced OSINT Reconnaissance · Powered by SpiderFoot</p>
                </div>
            </div>

            {/* Investigation Selector */}
            <div className="sf-investigation-bar">
                <div className="sf-inv-selector">
                    <span className="sf-inv-label">📂 Save Results To:</span>
                    <select
                        value={selectedInvestigation}
                        onChange={(e) => {
                            if (e.target.value === 'new') {
                                setShowNewInv(true);
                            } else {
                                setSelectedInvestigation(e.target.value);
                            }
                        }}
                        className="sf-inv-select"
                        disabled={loading}
                    >
                        <option value="" disabled>Select Investigation...</option>
                        {investigations.map(inv => (
                            <option key={inv.id} value={inv.id}>{inv.name}</option>
                        ))}
                        <option value="new">+ Create New Investigation...</option>
                    </select>
                </div>

                {savingToTimeline && <span className="sf-saving-status">💾 Saving to Timeline...</span>}
                {savedToTimeline && <span className="sf-saved-status">✅ Saved to Timeline</span>}
            </div>

            {/* New Investigation Inline */}
            {showNewInv && (
                <div className="sf-inv-create-inline">
                    <input
                        type="text"
                        placeholder="Investigation Name"
                        value={newInvName}
                        onChange={(e) => setNewInvName(e.target.value)}
                        className="sf-inv-input"
                    />
                    <button onClick={createQuickInvestigation} className="sf-inv-btn create">Create</button>
                    <button onClick={() => setShowNewInv(false)} className="sf-inv-btn cancel">Cancel</button>
                </div>
            )}

            {/* Tabs */}
            <div className="sf-tabs">
                <button
                    className={`sf-tab ${activeTab === 'new' ? 'active' : ''}`}
                    onClick={() => setActiveTab('new')}
                >
                    ➕ New Scan
                </button>

                <button
                    className={`sf-tab ${activeTab === 'history' ? 'active' : ''}`}
                    onClick={() => setActiveTab('history')}
                >
                    📋 Scan History ({previousScans.length})
                </button>
            </div>

            {/* New Scan Tab */}
            {activeTab === 'new' && (
                <div className="sf-scan-form">
                    <div className="sf-input-section">
                        <label>Target (Enter what you want to investigate)</label>
                        <input
                            type="text"
                            value={target}
                            onChange={(e) => setTarget(e.target.value)}
                            placeholder='e.g., example.com, john@email.com, "John Smith"'
                            disabled={loading}
                        />
                        <div className="sf-format-hints">
                            <div className="sf-hint-title">Supported Formats:</div>
                            <div className="sf-hints-grid">
                                <span className="sf-hint"><strong>🔗 URL:</strong> https://example.com <em>(fast check)</em></span>
                                <span className="sf-hint"><strong>📧 Email:</strong> john@example.com</span>
                                <span className="sf-hint"><strong>🌐 Domain:</strong> example.com</span>
                                <span className="sf-hint"><strong>🔢 IPv4:</strong> 1.2.3.4</span>
                                <span className="sf-hint"><strong>👤 Name:</strong> "John Smith" <em>(in quotes)</em></span>
                                <span className="sf-hint"><strong>🏷️ Username:</strong> "jsmith2000" <em>(in quotes)</em></span>
                            </div>
                        </div>
                    </div>

                    {/* ========== SCAN PRESET ========== */}
                    <div className="sf-scan-config">
                        <div className="sf-preset-cards">
                            {PRESET_CATEGORIES.map(cat => (
                                <button
                                    key={cat.id}
                                    className={`sf-preset-card ${scanType === cat.id ? 'active' : ''}`}
                                    onClick={() => setScanType(cat.id)}
                                    disabled={loading}
                                >
                                    <span className="preset-icon">{cat.icon}</span>
                                    <div className="preset-info">
                                        <span className="preset-label">{cat.label}</span>
                                        <span className="preset-desc">{cat.desc}</span>
                                    </div>
                                    <span className={`preset-speed speed-${cat.speed.toLowerCase()}`}>{cat.speed}</span>
                                </button>
                            ))}
                        </div>
                    </div>

                    <button
                        className="sf-start-btn"
                        onClick={startScan}
                        disabled={loading || !target.trim()}
                    >
                        {loading ? '🔄 Scanning...' : '🚀 Start Scan'}
                    </button>

                    {error && (
                        <div className="sf-error">
                            ⚠️ {error}
                        </div>
                    )}
                </div>
            )}

            {/* Scan History Tab */}
            {activeTab === 'history' && (
                <div className="sf-history">
                    {previousScans.length === 0 ? (
                        <div className="sf-empty">No previous scans found</div>
                    ) : (
                        <div className="sf-scan-list">
                            {previousScans.map(scan => (
                                <div
                                    key={scan.id}
                                    className="sf-scan-item"
                                    onClick={() => loadPreviousScan(scan)}
                                >
                                    <div className="sf-scan-target">{scan.target || scan.name}</div>
                                    <div className="sf-scan-meta">
                                        <span className={`sf-status ${scan.status?.toLowerCase()}`}>
                                            {scan.status}
                                        </span>
                                        <span>{scan.startTime}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}



            {/* Loading State - Initial (No results yet) */}
            {loading && !results && (
                <div className="sf-loading">
                    <div className="sf-loading-spinner"></div>
                    <div className="sf-loading-text">
                        <h3>
                            {scanMode === 'url' ? '🔗 Analyzing URL' : '🔍 Scanning Target'}: {target}
                        </h3>
                        <p>Status: <span className={`sf-status ${scanStatus?.toLowerCase()}`}>{scanStatus || 'Initializing...'}</span></p>
                        <p className="sf-loading-hint">
                            {scanMode === 'url'
                                ? 'Checking with VirusTotal & URLhaus... (~5-10 seconds)'
                                : 'SpiderFoot is gathering intelligence... (5-15 minutes)'
                            }
                        </p>
                        {scanMode === 'osint' && (
                            <button className="sf-stop-btn" onClick={stopScan}>
                                ⏹️ Stop Scan
                            </button>
                        )}
                    </div>
                </div>
            )}

            {/* Loading State - Streaming (Results visible) */}
            {loading && results && (
                <div className="sf-loading-streaming" style={{
                    padding: '16px',
                    background: 'rgba(168, 85, 247, 0.1)',
                    border: '1px solid rgba(168, 85, 247, 0.3)',
                    borderRadius: '8px',
                    marginBottom: '20px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '12px'
                }}>
                    <div className="sf-loading-spinner" style={{ width: '24px', height: '24px', borderWidth: '3px' }}></div>
                    <div style={{ color: '#fff', fontSize: '1.1rem' }}>
                        <strong>Scan in Progress:</strong> Streaming new findings...
                    </div>
                    <button className="sf-stop-btn small" onClick={stopScan} style={{ padding: '5px 12px', fontSize: '0.8rem', marginLeft: '20px' }}>
                        ⏹️ Stop
                    </button>
                </div>
            )}

            {/* Breach Results (LeakCheck) */}
            {breachResults && (
                <div className="sf-results breach-results" style={{ marginBottom: '20px', borderLeft: breachResults.success ? '4px solid #ef4444' : '4px solid #22c55e' }}>
                    <div className="sf-results-header">
                        <div>
                            <h2>🔓 Credential Exposure Analysis</h2>
                            <div className="sf-results-meta">
                                Target: <strong>{target}</strong> |
                                Status: <span className={`sf-status ${breachResults.success ? 'danger' : 'safe'}`}>
                                    {breachResults.success ? 'COMPROMISED' : 'SAFE'}
                                </span>
                            </div>
                        </div>
                        <div className="sf-results-actions">
                            <button
                                className={`sf-sync-btn ${savedToTimeline ? 'saved' : ''}`}
                                onClick={() => saveToTimeline('manual', target, breachResults)}
                                disabled={savingToTimeline || savedToTimeline || !selectedInvestigation}
                            >
                                {savingToTimeline ? '💾 Saving...' : savedToTimeline ? '✅ Saved' : '📥 Sync to Timeline'}
                            </button>
                        </div>
                    </div>

                    <div className="sf-stats" style={{ marginTop: '15px' }}>
                        <div className={`sf-stat-card ${breachResults.success ? 'leak' : 'safe'}`}>
                            <div className="sf-stat-value">{breachResults.sources ? breachResults.sources.length : 0}</div>
                            <div className="sf-stat-label">Breaches Found</div>
                        </div>
                        <div className="sf-stat-description" style={{ flex: 2, padding: '10px', color: '#cbd5e1' }}>
                            {breachResults.success
                                ? '⚠️ This email address was found in known data breaches. Passwords and personal data may be exposed.'
                                : '✅ No public breach records found for this email address.'}
                        </div>
                    </div>

                    {breachResults.sources && breachResults.sources.length > 0 && (
                        <div className="sf-findings">
                            <div className="sf-category danger">
                                <h3>⚠️ Known Leaks</h3>
                                <div className="sf-items">
                                    {breachResults.sources.map((source, i) => (
                                        <div key={i} className="sf-item danger">
                                            <span className="sf-item-type">Source</span>
                                            <span className="sf-item-data">
                                                {typeof source === 'string' ? source : (source.name || JSON.stringify(source))}
                                                {source.date ? ` (${source.date})` : ''}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Phishing/URL Results */}
            {!loading && phishingResults && (
                <div className="sf-results phishing-results">
                    <div className="sf-results-header">
                        <h2>🔗 URL Security Analysis</h2>
                        <div className="sf-results-actions">
                            <button
                                className={`sf-sync-btn ${savedToTimeline ? 'saved' : ''}`}
                                onClick={() => saveToTimeline('phishing', target, phishingResults)}
                                disabled={savingToTimeline || savedToTimeline || !selectedInvestigation}
                            >
                                {savingToTimeline ? '💾 Saving...' : savedToTimeline ? '✅ Saved' : '📥 Sync to Timeline'}
                            </button>
                        </div>
                        <div className="sf-results-meta">
                            URL: <strong>{target}</strong> |
                            Status: <span className={`sf-status ${phishingResults.isPhishing ? 'danger' : 'finished'}`}>
                                {phishingResults.isPhishing ? 'POTENTIALLY DANGEROUS' : 'APPEARS SAFE'}
                            </span>
                        </div>
                    </div>

                    {/* Threat Score */}
                    <div className="phishing-score-section">
                        <div className={`threat-score ${phishingResults.isPhishing ? 'danger' : 'safe'}`}>
                            <div className="score-value">{phishingResults.confidence || 0}%</div>
                            <div className="score-label">Threat Confidence</div>
                        </div>
                        <div className="threat-verdict">
                            {phishingResults.isPhishing ? (
                                <div className="verdict danger">
                                    <span className="verdict-icon">⚠️</span>
                                    <span>This URL shows signs of being malicious or phishing</span>
                                </div>
                            ) : (
                                <div className="verdict safe">
                                    <span className="verdict-icon">✅</span>
                                    <span>No immediate threats detected</span>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Source Results */}
                    <div className="sf-findings">
                        {phishingResults.sources && phishingResults.sources.map((source, idx) => (
                            <div key={idx} className={`source-result ${source.found ? 'danger' : 'safe'}`}>
                                <div className="source-header">
                                    <span className="source-badge">{source.source}</span>
                                    <span className={`source-status ${source.found ? 'found' : 'clean'}`}>
                                        {source.found ? '⚠️ Found in database' : '✓ Clean'}
                                    </span>
                                </div>
                                {source.details && (
                                    <div className="source-details">
                                        {source.details.positives && (
                                            <span>Detections: {source.details.positives}/{source.details.total}</span>
                                        )}
                                        {source.details.threat && (
                                            <span>Threat: {source.details.threat}</span>
                                        )}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>

                    {/* New Scan Button */}
                    <div className="sf-external-link">
                        <button className="sf-start-btn" onClick={() => {
                            setPhishingResults(null);
                            setTarget('');
                            setScanMode(null);
                        }}>
                            🔍 Scan Another Target
                        </button>
                    </div>
                </div>
            )}

            {/* OSINT Results */}
            {results && (
                <div className="sf-results">
                    <div className="sf-results-header">
                        <div>
                            <h2>🎯 Scan Results</h2>
                            <div className="sf-results-meta">
                                Target: <strong>{target}</strong> |
                                Status: <span className={`sf-status ${String(scanStatus).toLowerCase()}`}>{scanStatus}</span>
                            </div>
                        </div>
                        <div className="sf-results-actions">
                            <button
                                className={`sf-sync-btn ${savedToTimeline ? 'saved' : ''}`}
                                onClick={() => saveToTimeline('spiderfoot', target, results)}
                                disabled={savingToTimeline || savedToTimeline || !selectedInvestigation || loading}
                            >
                                {savingToTimeline ? '💾 Saving...' : savedToTimeline ? '✅ Saved' : (loading ? '🚫 Wait for Scan' : '📥 Sync to Timeline')}
                            </button>
                        </div>
                    </div>

                    {/* Stats Summary */}
                    <div className="sf-stats">
                        <div className="sf-stat-card">
                            <div className="sf-stat-value">
                                {Math.max(0, (results.stats?.totalFindings || 0) - (results.stats?.personal || 0) - (results.stats?.emails || 0))}
                            </div>
                            <div className="sf-stat-label">Total Findings</div>
                        </div>
                        <div className="sf-stat-card accounts">
                            <div className="sf-stat-value">{results.stats?.accounts || 0}</div>
                            <div className="sf-stat-label">Accounts Found</div>
                        </div>
                        <div className="sf-stat-card domain">
                            <div className="sf-stat-value">{results.stats?.domains || 0}</div>
                            <div className="sf-stat-label">Domains</div>
                        </div>
                        <div className="sf-stat-card leak">
                            <div className="sf-stat-value">{results.stats?.leaks || 0}</div>
                            <div className="sf-stat-label">Data Leaks</div>
                        </div>
                    </div>

                    {/* Findings by Category */}
                    <div className="sf-findings">
                        {/* Accounts on External Sites */}
                        {results.findings?.accounts?.length > 0 && (
                            <div className="sf-category accounts">
                                <h3>👤 Accounts on External Sites</h3>
                                <div className="sf-items">
                                    {results.findings.accounts.map((item, i) => (
                                        <div key={i} className="sf-item">
                                            <span className="sf-item-type">{item.type}</span>
                                            <span className="sf-item-data">{item.data}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}



                        {/* Domains & Internet Names */}
                        {results.findings?.domains?.length > 0 && (
                            <div className="sf-category">
                                <h3>🌐 Domains & Internet Names</h3>
                                <div className="sf-items">
                                    {results.findings.domains.map((item, i) => (
                                        <div key={i} className="sf-item">
                                            <span className="sf-item-type">{item.type}</span>
                                            <span className="sf-item-data">{item.data}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* IP Addresses */}
                        {results.findings?.ipAddresses?.length > 0 && (
                            <div className="sf-category">
                                <h3>🔢 IP Addresses</h3>
                                <div className="sf-items">
                                    {results.findings.ipAddresses.map((item, i) => (
                                        <div key={i} className="sf-item">
                                            <span className="sf-item-type">{item.type}</span>
                                            <span className="sf-item-data">{item.data}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Infrastructure */}
                        {results.findings?.infrastructure?.length > 0 && (
                            <div className="sf-category">
                                <h3>🖥️ Infrastructure</h3>
                                <div className="sf-items">
                                    {results.findings.infrastructure.map((item, i) => (
                                        <div key={i} className="sf-item">
                                            <span className="sf-item-type">{item.type}</span>
                                            <span className="sf-item-data">{item.data}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Geo & Network Intelligence */}
                        {results.findings?.geoNetwork?.length > 0 && (
                            <div className="sf-category geo-network">
                                <h3>🌍 Geo & Network Intelligence</h3>
                                {results.findings.geoNetwork.map((item, i) => (
                                    <div key={i} className="sf-geo-card">
                                        {/* Location & Network Row */}
                                        {(item.city || item.country) && (
                                            <div className="sf-geo-row">
                                                <span className="sf-geo-label">📍 Location</span>
                                                <span className="sf-geo-value">
                                                    {[item.city, item.country].filter(Boolean).join(', ')}
                                                </span>
                                            </div>
                                        )}
                                        {item.bgpRoute && (
                                            <div className="sf-geo-row">
                                                <span className="sf-geo-label">🔀 BGP Route</span>
                                                <span className="sf-geo-value mono">{item.bgpRoute}</span>
                                            </div>
                                        )}
                                        {item.asn && (
                                            <div className="sf-geo-row">
                                                <span className="sf-geo-label">🏢 ASN</span>
                                                <span className="sf-geo-value">
                                                    AS{item.asn} — {item.asName || 'Unknown'}
                                                </span>
                                            </div>
                                        )}
                                        {item.asDesc && (
                                            <div className="sf-geo-row">
                                                <span className="sf-geo-label">📝 WHOIS</span>
                                                <span className="sf-geo-value">{item.asDesc}</span>
                                            </div>
                                        )}
                                        {/* Passive DNS Records */}
                                        {item.passiveDNS?.length > 0 && (
                                            <div className="sf-passive-dns">
                                                <span className="sf-geo-label">🔗 Passive DNS ({item.passiveDNS.length} records)</span>
                                                <div className="sf-dns-list">
                                                    {item.passiveDNS.map((dns, j) => (
                                                        <div key={j} className="sf-dns-item">
                                                            <span className="sf-dns-dot">●</span>
                                                            <span className="sf-dns-domain">{dns.domain}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                        {/* Fallback for non-JSON entries */}
                                        {!item.city && !item.asn && !item.passiveDNS?.length && item.data && (
                                            <div className="sf-geo-row">
                                                <span className="sf-geo-label">📋 Data</span>
                                                <span className="sf-geo-value">{item.data}</span>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                        {results.findings?.leaks?.length > 0 && (
                            <div className="sf-category danger">
                                <h3>⚠️ Data Leaks & Breaches</h3>
                                <div className="sf-items">
                                    {results.findings.leaks.map((item, i) => (
                                        <div key={i} className="sf-item danger">
                                            <span className="sf-item-type">{item.type}</span>
                                            <span className="sf-item-data">{item.data}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Other Findings */}
                        {results.findings?.other?.length > 0 && (
                            <div className="sf-category other">
                                <h3>📋 Other Findings</h3>
                                <div className="sf-items">
                                    {results.findings.other.slice(0, 30).map((item, i) => (
                                        <div key={i} className="sf-item">
                                            <span className="sf-item-type">{item.type}</span>
                                            <span className="sf-item-data">{item.data}</span>
                                        </div>
                                    ))}
                                    {results.findings.other.length > 30 && (
                                        <div className="sf-more">
                                            +{results.findings.other.length - 30} more findings
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {results.stats?.totalFindings === 0 && (
                            <div className="sf-empty-results">
                                <p>No findings yet. The scan may still be gathering data.</p>
                                <button onClick={() => fetchResults(scanId)}>
                                    🔄 Refresh Results
                                </button>
                            </div>
                        )}
                    </div>

                    {/* New Scan Button */}
                    <div className="sf-external-link">
                        <button className="sf-start-btn" onClick={() => {
                            setResults(null);
                            setTarget('');
                            setScanMode(null);
                            setScanId(null);
                        }}>
                            🔍 Scan Another Target
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default SocialMapping;
