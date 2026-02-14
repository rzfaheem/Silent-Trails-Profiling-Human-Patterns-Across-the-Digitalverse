/**
 * Silent Trails - Backend Server
 * Multi-source threat intelligence: VirusTotal + URLhaus + Supabase
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');
const fs = require('fs');
const { createClient } = require('@supabase/supabase-js');

const app = express();
const PORT = process.env.PORT || 3002;

// Supabase Client
const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_ANON_KEY
);

// Middleware
app.use(cors());
app.use(express.json());

// API Configuration
const VT_API_KEY = process.env.VIRUSTOTAL_API_KEY;
const VT_BASE_URL = 'https://www.virustotal.com/api/v3';

// URLhaus Configuration (FREE - get Auth-Key from https://auth.abuse.ch/)
const URLHAUS_API_URL = 'https://urlhaus-api.abuse.ch/v1/url/';
const URLHAUS_AUTH_KEY = process.env.URLHAUS_AUTH_KEY;

// SpiderFoot Configuration (Docker container)
const SPIDERFOOT_URL = process.env.SPIDERFOOT_URL || 'http://localhost:5001';

/**
 * Health check endpoint
 */
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        message: 'Silent Trails Backend is running',
        services: ['VirusTotal', 'URLhaus', 'SpiderFoot', 'Supabase']
    });
});

/**
 * Analyze URL for phishing/malware using multiple sources
 * POST /api/check-url
 * Body: { url: "https://example.com" }
 */
app.post('/api/check-url', async (req, res) => {
    try {
        const { url } = req.body;

        if (!url) {
            return res.status(400).json({ error: 'URL is required' });
        }

        console.log(`\n[Analysis] Starting multi-source analysis for: ${url}`);

        // Run both checks in parallel for speed
        const [virusTotalResult, urlhausResult] = await Promise.allSettled([
            checkVirusTotal(url),
            checkURLhaus(url)
        ]);

        // Extract results
        const vtData = virusTotalResult.status === 'fulfilled' ? virusTotalResult.value : null;
        const uhData = urlhausResult.status === 'fulfilled' ? urlhausResult.value : null;

        // Combine results
        const combinedResult = combineResults(url, vtData, uhData);

        console.log(`[Analysis] Complete - Phishing: ${combinedResult.isPhishing}, Confidence: ${combinedResult.confidence}%`);

        res.json(combinedResult);

    } catch (error) {
        console.error('[Analysis] Error:', error.message);
        res.status(500).json({
            error: 'Failed to analyze URL',
            message: error.message,
            isPhishing: null,
            confidence: 0,
            indicators: ['API Error occurred'],
            recommendation: 'Could not verify this URL. Please try again later.'
        });
    }
});

/**
 * Check URL with VirusTotal
 */
async function checkVirusTotal(url) {
    try {
        console.log('[VirusTotal] Checking URL...');

        const urlId = Buffer.from(url).toString('base64').replace(/=/g, '');

        // Try to get existing report
        try {
            const response = await axios.get(
                `${VT_BASE_URL}/urls/${urlId}`,
                { headers: { 'x-apikey': VT_API_KEY } }
            );

            const attributes = response.data.data.attributes;
            const stats = attributes.last_analysis_stats || {};

            console.log(`[VirusTotal] Found report - Malicious: ${stats.malicious}, Suspicious: ${stats.suspicious}`);

            return {
                source: 'VirusTotal',
                found: true,
                malicious: stats.malicious || 0,
                suspicious: stats.suspicious || 0,
                harmless: stats.harmless || 0,
                undetected: stats.undetected || 0,
                total: (stats.malicious || 0) + (stats.suspicious || 0) + (stats.harmless || 0) + (stats.undetected || 0),
                categories: attributes.categories || {}
            };

        } catch (err) {
            if (err.response && err.response.status === 404) {
                // Submit for new scan
                console.log('[VirusTotal] No existing report, submitting for scan...');

                const scanResponse = await axios.post(
                    `${VT_BASE_URL}/urls`,
                    `url=${encodeURIComponent(url)}`,
                    {
                        headers: {
                            'x-apikey': VT_API_KEY,
                            'Content-Type': 'application/x-www-form-urlencoded'
                        }
                    }
                );

                // Wait briefly for scan
                await new Promise(resolve => setTimeout(resolve, 3000));

                const analysisId = scanResponse.data.data.id;
                const analysisResponse = await axios.get(
                    `${VT_BASE_URL}/analyses/${analysisId}`,
                    { headers: { 'x-apikey': VT_API_KEY } }
                );

                const stats = analysisResponse.data.data.attributes.stats || {};

                return {
                    source: 'VirusTotal',
                    found: true,
                    newScan: true,
                    malicious: stats.malicious || 0,
                    suspicious: stats.suspicious || 0,
                    harmless: stats.harmless || 0,
                    undetected: stats.undetected || 0,
                    total: (stats.malicious || 0) + (stats.suspicious || 0) + (stats.harmless || 0) + (stats.undetected || 0)
                };
            }
            throw err;
        }

    } catch (error) {
        console.error('[VirusTotal] Error:', error.message);
        return { source: 'VirusTotal', found: false, error: error.message };
    }
}

/**
 * Check URL with URLhaus (abuse.ch)
 * Requires Auth-Key from https://auth.abuse.ch/
 */
async function checkURLhaus(url) {
    try {
        // Skip if no Auth-Key configured
        if (!URLHAUS_AUTH_KEY) {
            console.log('[URLhaus] Skipped - No Auth-Key configured');
            return {
                source: 'URLhaus',
                found: false,
                error: 'Not configured (add URLHAUS_AUTH_KEY to .env)'
            };
        }

        console.log('[URLhaus] Checking URL...');

        const response = await axios.post(
            URLHAUS_API_URL,
            `url=${encodeURIComponent(url)}`,
            {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Auth-Key': URLHAUS_AUTH_KEY
                }
            }
        );

        const data = response.data;

        if (data.query_status === 'ok') {
            console.log(`[URLhaus] FOUND in database - Threat: ${data.threat}, Status: ${data.url_status}`);

            return {
                source: 'URLhaus',
                found: true,
                inDatabase: true,
                threat: data.threat || 'unknown',
                urlStatus: data.url_status || 'unknown',
                dateAdded: data.date_added,
                tags: data.tags || [],
                payloads: data.payloads ? data.payloads.length : 0,
                reporter: data.reporter
            };
        } else if (data.query_status === 'no_results') {
            console.log('[URLhaus] Not found in database (clean)');
            return {
                source: 'URLhaus',
                found: true,
                inDatabase: false
            };
        } else {
            console.log(`[URLhaus] Query status: ${data.query_status}`);
            return {
                source: 'URLhaus',
                found: true,
                inDatabase: false
            };
        }

    } catch (error) {
        console.error('[URLhaus] Error:', error.message);
        return { source: 'URLhaus', found: false, error: error.message };
    }
}


/**
 * Combine results from all sources
 */
function combineResults(url, vtData, uhData) {
    const indicators = [];
    const sources = [];
    let threatScore = 0;
    let maxConfidence = 0;

    // Process VirusTotal results
    if (vtData && vtData.found && !vtData.error) {
        sources.push('VirusTotal');

        if (vtData.malicious > 0) {
            indicators.push(`🛡️ VirusTotal: ${vtData.malicious} security vendor(s) flagged as malicious`);
            threatScore += vtData.malicious * 15;
            maxConfidence = Math.max(maxConfidence, vtData.malicious >= 3 ? 90 : vtData.malicious >= 1 ? 70 : 50);
        }
        if (vtData.suspicious > 0) {
            indicators.push(`⚡ VirusTotal: ${vtData.suspicious} vendor(s) marked as suspicious`);
            threatScore += vtData.suspicious * 10;
            maxConfidence = Math.max(maxConfidence, 50);
        }
        if (vtData.malicious === 0 && vtData.suspicious === 0 && vtData.harmless > 0) {
            indicators.push(`✅ VirusTotal: ${vtData.harmless} vendors confirmed safe`);
        }
    } else if (vtData && vtData.error) {
        indicators.push(`⚠️ VirusTotal: Could not check (${vtData.error})`);
    }

    // Process URLhaus results
    if (uhData && uhData.found && !uhData.error) {
        sources.push('URLhaus');

        if (uhData.inDatabase) {
            indicators.push(`🦠 URLhaus: Found in malware database!`);
            indicators.push(`   └─ Threat type: ${uhData.threat}`);
            indicators.push(`   └─ Status: ${uhData.urlStatus}`);
            if (uhData.tags && uhData.tags.length > 0) {
                indicators.push(`   └─ Tags: ${uhData.tags.join(', ')}`);
            }
            if (uhData.payloads > 0) {
                indicators.push(`   └─ Known payloads: ${uhData.payloads}`);
            }
            threatScore += 50;
            maxConfidence = Math.max(maxConfidence, 95);
        } else {
            indicators.push(`✅ URLhaus: Not found in malware database`);
        }
    } else if (uhData && uhData.error) {
        indicators.push(`⚠️ URLhaus: Could not check (${uhData.error})`);
    }

    // Calculate final verdict
    const isPhishing = threatScore >= 15 || (vtData && vtData.malicious >= 1) || (uhData && uhData.inDatabase);
    const confidence = Math.min(95, Math.max(10, maxConfidence || (isPhishing ? 60 : 15)));
    const riskLevel = confidence > 70 ? 'high' : confidence > 40 ? 'medium' : 'low';

    // Generate recommendation
    let recommendation;
    if (uhData && uhData.inDatabase) {
        recommendation = '🚨 DANGER: This URL is in the URLhaus malware database. Do NOT visit this link!';
    } else if (vtData && vtData.malicious >= 3) {
        recommendation = '⛔ DANGER: Multiple security vendors have flagged this URL. Avoid at all costs!';
    } else if (vtData && vtData.malicious >= 1) {
        recommendation = '⚠️ WARNING: Security vendors have detected threats. Exercise extreme caution.';
    } else if (vtData && vtData.suspicious >= 2) {
        recommendation = '⚡ CAUTION: Some suspicious indicators detected. Verify before proceeding.';
    } else if (indicators.length === 0 || indicators.every(i => i.includes('✅'))) {
        recommendation = '✅ This URL appears to be safe based on current threat intelligence.';
    } else {
        recommendation = 'ℹ️ No major threats detected, but always exercise caution with unknown links.';
    }

    // Build stats object
    const stats = {
        malicious: vtData?.malicious || 0,
        suspicious: vtData?.suspicious || 0,
        harmless: vtData?.harmless || 0,
        undetected: vtData?.undetected || 0,
        total: vtData?.total || 0,
        urlhausListed: uhData?.inDatabase || false
    };

    return {
        input: url,
        isPhishing,
        confidence,
        riskLevel,
        indicators: indicators.length > 0 ? indicators : ['No threat data available'],
        recommendation,
        stats,
        sources,
        urlhausData: uhData?.inDatabase ? {
            threat: uhData.threat,
            status: uhData.urlStatus,
            tags: uhData.tags,
            dateAdded: uhData.dateAdded
        } : null,
        analyzedAt: new Date().toISOString()
    };
}

/**
 * Analyze message text for phishing indicators
 * POST /api/check-message
 */
app.post('/api/check-message', async (req, res) => {
    try {
        const { message } = req.body;

        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }

        // Extract URLs from message
        const urlRegex = /(https?:\/\/[^\s]+)/gi;
        const urls = message.match(urlRegex) || [];

        // Analyze keywords and patterns
        const result = analyzeMessagePatterns(message, urls);

        // If URLs found, check them with our multi-source API
        if (urls.length > 0) {
            console.log(`[Message] Found ${urls.length} URL(s), checking first one...`);

            const [vtResult, uhResult] = await Promise.allSettled([
                checkVirusTotal(urls[0]),
                checkURLhaus(urls[0])
            ]);

            const vtData = vtResult.status === 'fulfilled' ? vtResult.value : null;
            const uhData = uhResult.status === 'fulfilled' ? uhResult.value : null;

            // Add URL analysis results
            if (vtData && vtData.malicious > 0) {
                result.indicators.push(`🛡️ URL flagged by ${vtData.malicious} VirusTotal vendor(s)`);
                result.confidence = Math.max(result.confidence, 70);
                result.isPhishing = true;
            }
            if (uhData && uhData.inDatabase) {
                result.indicators.push(`🦠 URL found in URLhaus malware database`);
                result.confidence = Math.max(result.confidence, 95);
                result.isPhishing = true;
            }

            result.sources = ['Pattern Analysis', 'VirusTotal', 'URLhaus'];
        }

        res.json(result);

    } catch (error) {
        console.error('[Message Analysis] Error:', error.message);
        res.status(500).json({
            error: 'Failed to analyze message',
            message: error.message
        });
    }
});

/**
 * Analyze message text for phishing patterns
 */
function analyzeMessagePatterns(message, extractedUrls) {
    const lowerMessage = message.toLowerCase();
    const indicators = [];
    let score = 0;

    // Urgency patterns
    if (lowerMessage.match(/urgent|immediately|act now|limited time|expires/)) {
        indicators.push('⏰ Urgency language detected');
        score += 15;
    }

    // Credential requests
    if (lowerMessage.match(/password|verify your account|confirm your identity|update your information/)) {
        indicators.push('🔐 Credential request detected');
        score += 20;
    }

    // Threat patterns
    if (lowerMessage.match(/suspended|blocked|unauthorized|unusual activity|security alert/)) {
        indicators.push('⚠️ Account threat language detected');
        score += 15;
    }

    // Prize patterns
    if (lowerMessage.match(/winner|congratulations|selected|prize|reward|free gift/)) {
        indicators.push('🎁 Prize/reward claims detected');
        score += 15;
    }

    // URL patterns
    if (extractedUrls.length > 0) {
        indicators.push(`🔗 Contains ${extractedUrls.length} URL(s)`);
        extractedUrls.forEach(url => {
            if (!url.startsWith('https://')) {
                indicators.push('⚠️ Non-HTTPS link detected');
                score += 10;
            }
        });
    }

    score = Math.min(score, 95);
    const isPhishing = score >= 30;

    return {
        input: message.substring(0, 100) + (message.length > 100 ? '...' : ''),
        isPhishing,
        confidence: score,
        riskLevel: score > 60 ? 'high' : score > 30 ? 'medium' : 'low',
        indicators: indicators.length > 0 ? indicators : ['No suspicious patterns detected'],
        recommendation: isPhishing
            ? 'This message contains phishing indicators. Do not click any links or provide personal information.'
            : 'No obvious phishing patterns detected, but always verify sender identity.',
        extractedUrls,
        sources: ['Pattern Analysis'],
        analyzedAt: new Date().toISOString()
    };
}

// ==================== SPIDERFOOT OSINT INTEGRATION ====================

/**
 * Start a new SpiderFoot scan
 * POST /api/profile-scan
 * Body: { target: "username or email", scanType: "all" }
 */
app.post('/api/profile-scan', async (req, res) => {
    try {
        const { target, scanType, typelist } = req.body;

        if (!target) {
            return res.status(400).json({ error: 'Target is required' });
        }

        console.log(`\n[SpiderFoot] Starting OSINT scan for: ${target}`);

        // Create a unique scan name
        const scanName = `silenttrails_${Date.now()}`;

        // Determine scan mode: by data types or by preset
        let usecase = 'All';
        let typelistStr = '';

        if (typelist && typelist.length > 0) {
            typelistStr = Array.isArray(typelist) ? typelist.join(',') : typelist;
            console.log(`[SpiderFoot] Mode: By Required Data, types: ${typelistStr}`);
        } else {
            if (scanType === 'passive') usecase = 'Passive';
            else if (scanType === 'footprint') usecase = 'Footprint';
            else if (scanType === 'investigate') usecase = 'Investigate';
            else usecase = 'All';
            console.log(`[SpiderFoot] Mode: Preset, usecase: ${usecase}`);
        }

        // SpiderFoot v4 requires session cookies for scan creation
        const sessionResponse = await axios.get(`${SPIDERFOOT_URL}/newscan`, {
            withCredentials: true
        });

        const cookies = sessionResponse.headers['set-cookie'] || [];
        const cookieString = cookies.map(c => c.split(';')[0]).join('; ');

        // Start scan via SpiderFoot API with session cookies
        const formData = new URLSearchParams();
        formData.append('scanname', scanName);
        formData.append('scantarget', target);
        formData.append('usecase', usecase);
        formData.append('modulelist', '');
        formData.append('typelist', typelistStr);

        console.log(`[SpiderFoot] Starting scan with name: ${scanName}, target: ${target}, usecase: ${usecase}, typelist: ${typelistStr || '(none)'}`);

        const response = await axios.post(
            `${SPIDERFOOT_URL}/startscan`,
            formData.toString(),
            {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Cookie': cookieString,
                    'Referer': `${SPIDERFOOT_URL}/newscan`
                },
                maxRedirects: 5
            }
        );

        // SpiderFoot returns HTML redirect, wait and retry to get the new scan from list
        // Sometimes SpiderFoot takes a moment to register the new scan
        let ourScan = null;
        let retries = 5;

        while (!ourScan && retries > 0) {
            await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds

            const scanListResponse = await axios.get(`${SPIDERFOOT_URL}/scanlist`);
            const scans = scanListResponse.data || [];

            console.log(`[SpiderFoot] Looking for scan with name: "${scanName}" (retry ${6 - retries}/5)`);
            console.log(`[SpiderFoot] Scan list has ${scans.length} scans. Names: ${scans.slice(0, 5).map(s => `"${s[1]}"`).join(', ')}`);

            // SpiderFoot scan format: [scanId, scanName, target, startTime, endTime, finishTime, status, numResults, riskLevels]
            // Find our scan by matching the scan name (index 1) which we generated
            ourScan = scans.find(s => s[1] && s[1] === scanName);

            // Also try matching by target if name not found
            if (!ourScan) {
                // Try finding the most recent scan for this target
                ourScan = scans.find(s => s[2] && s[2] === target && s[1]?.startsWith('silenttrails_'));
            }

            // Last resort: find the newest scan (first in list) that wasn't there before
            if (!ourScan && scans.length > 0) {
                const newest = scans[0];
                console.log(`[SpiderFoot] Trying newest scan: ID=${newest[0]}, Name="${newest[1]}", Target="${newest[2]}"`);
                // If the newest scan's target matches, use it
                if (newest[2] === target) {
                    ourScan = newest;
                }
            }

            retries--;
        }

        if (!ourScan) {
            console.error(`[SpiderFoot] Could not find scan after retries. Name: ${scanName}, Target: ${target}`);
            throw new Error('Scan was submitted but could not be found in SpiderFoot. Please try again.');
        }

        const scanId = ourScan[0];
        console.log(`[SpiderFoot] Scan started with ID: ${scanId}, Name: ${scanName}, Target: ${target}`);

        res.json({
            success: true,
            scanId,
            scanName,
            target,
            usecase,
            message: 'Scan started successfully. SpiderFoot is now gathering intelligence - this takes time!',
            startedAt: new Date().toISOString()
        });

    } catch (error) {
        console.error('[SpiderFoot] Error starting scan:', error.message);
        res.status(500).json({
            error: 'Failed to start scan',
            message: error.message,
            suggestion: 'Make sure SpiderFoot is running on port 5001'
        });
    }
});

/**
 * Get scan status
 * GET /api/profile-scan/:id/status
 * SpiderFoot /scanstatus returns: [name, target, startTime, endTime, finishTime, status, {risks}]
 */
app.get('/api/profile-scan/:id/status', async (req, res) => {
    try {
        const { id } = req.params;

        const response = await axios.get(`${SPIDERFOOT_URL}/scanstatus?id=${id}`);
        const data = response.data;

        // SpiderFoot returns an array: [name, target, start, end, finish, STATUS, risks]
        let status = 'UNKNOWN';
        if (Array.isArray(data)) {
            status = data[5] || 'UNKNOWN';
        } else if (data && typeof data === 'object') {
            status = data.status || 'UNKNOWN';
        }

        console.log(`[SpiderFoot] Scan ${id} status: ${status}`);

        res.json({
            scanId: id,
            status: status,
            running: status === 'RUNNING' || status === 'STARTING',
            finished: status === 'FINISHED' || status === 'ABORTED' || status === 'ERROR-FAILED'
        });

    } catch (error) {
        console.error('[SpiderFoot] Error getting status:', error.message);
        res.status(500).json({ error: 'Failed to get scan status', message: error.message });
    }
});

/**
 * Stop a scan
 * POST /api/profile-scan/:id/stop
 */
app.post('/api/profile-scan/:id/stop', async (req, res) => {
    try {
        const { id } = req.params;
        console.log(`[SpiderFoot] Stopping scan: ${id}`);
        // SpiderFoot stop endpoint
        const response = await axios.get(`${SPIDERFOOT_URL}/stopscan?id=${id}`);

        if (response.data && response.data[0] === 'SUCCESS') {
            res.json({ success: true, message: 'Scan stopped successfully' });
        } else {
            res.json({ success: true, message: 'Scan stop signal sent' });
        }
    } catch (error) {
        console.error('[SpiderFoot] Error stopping scan:', error.message);
        res.status(500).json({ error: 'Failed to stop scan', message: error.message });
    }
});

/**
 * Get scan results summary
 * GET /api/profile-scan/:id/results
 */
app.get('/api/profile-scan/:id/results', async (req, res) => {
    try {
        const { id } = req.params;

        console.log(`[SpiderFoot] Fetching results for scan: ${id}`);

        // Get scan summary
        const summaryResponse = await axios.get(`${SPIDERFOOT_URL}/scansummary?id=${id}&by=type`);
        const summary = summaryResponse.data || [];

        // Get detailed events (limited)
        const eventsResponse = await axios.get(`${SPIDERFOOT_URL}/scaneventresults?id=${id}&eventType=ALL`);
        const events = eventsResponse.data || [];

        // SpiderFoot event type to user-friendly label mapping
        const typeLabels = {
            'ACCOUNT_EXTERNAL_OWNED': '👤 Account on External Site',
            'ACCOUNT_EXTERNAL_OWNED_OTHERS': '👥 Account Linked to Others',
            'USERNAME': '🏷️ Username',
            'EMAILADDR': '📧 Email Address',
            'EMAILADDR_GENERIC': '📧 Generic Email',
            'PHONE_NUMBER': '📱 Phone Number',
            'HUMAN_NAME': '👤 Name',
            'SOCIAL_MEDIA': '📱 Social Media Profile',
            'INTERNET_NAME': '🌐 Internet Name/Domain',
            'DOMAIN_NAME': '🌍 Domain Name',
            'IP_ADDRESS': '🔢 IP Address',
            'IPV6_ADDRESS': '🔢 IPv6 Address',
            'MALICIOUS_IPADDR': '⚠️ Malicious IP',
            'MALICIOUS_EMAILADDR': '⚠️ Malicious Email',
            'EMAILADDR_COMPROMISED': '🔓 Breached Email',
            'PASSWORD_COMPROMISED': '🔑 Password Exposed',
            'LEAK': '🔓 Data Leak',
            'DARKNET_MENTION': '🕵️ Darknet Mention',
            'GEOINFO': '📍 Geographic Info',
            'WEBSERVER_BANNER': '🖥️ Web Server',
            'WEBSERVER_TECHNOLOGY': '⚙️ Web Technology',
            'RAW_RIR_DATA': '📋 Registry Data',
            'PROVIDER_DNS': '🌐 DNS Provider',
            'PROVIDER_MAIL': '📧 Mail Provider',
            'ROOT': '🎯 Scan Target'
        };

        // Categorize findings with better labels
        const findings = {
            accounts: [],      // External accounts
            personal: [],      // Names, usernames, phones
            emails: [],        // Email addresses
            domains: [],       // Domains & internet names
            ipAddresses: [],   // IPs
            leaks: [],         // Breaches & leaks
            infrastructure: [], // Web servers, DNS
            geoNetwork: [],    // Geo/IP intelligence & registry data
            other: []
        };

        // Process events - SpiderFoot format: [timestamp, data, source, module, ?, ?, ?, hash, ?, ?, eventType]
        events.forEach(event => {
            const timestamp = event[0];
            const rawData = event[1];
            const source = event[2];
            const eventType = event[10] || 'UNKNOWN';

            // Clean up data (remove SFURL tags, decode HTML entities, etc)
            const cleanData = rawData
                .replace(/&lt;SFURL&gt;/g, '')
                .replace(/&lt;\/SFURL&gt;/g, '')
                .replace(/<SFURL>/g, '')
                .replace(/<\/SFURL>/g, '')
                .replace(/&quot;/g, '"')
                .replace(/&amp;/g, '&')
                .replace(/&lt;/g, '<')
                .replace(/&gt;/g, '>')
                .replace(/&#39;/g, "'")
                .replace(/\\n/g, ' - ');

            const friendlyType = typeLabels[eventType] || eventType.replace(/_/g, ' ');

            const finding = {
                type: friendlyType,
                rawType: eventType,
                data: cleanData,
                source,
                timestamp
            };

            // Categorize based on event type
            if (eventType.includes('ACCOUNT_EXTERNAL')) {
                findings.accounts.push(finding);
            } else if (['USERNAME', 'HUMAN_NAME', 'PHONE_NUMBER'].includes(eventType)) {
                findings.personal.push(finding);
            } else if (eventType.includes('EMAIL')) {
                findings.emails.push(finding);
            } else if (eventType.includes('DOMAIN') || eventType.includes('INTERNET_NAME')) {
                findings.domains.push(finding);
            } else if (eventType.includes('IP') || eventType.includes('IPV')) {
                findings.ipAddresses.push(finding);
            } else if (eventType.includes('COMPROMISED') || eventType.includes('LEAK') || eventType.includes('DARKNET') || eventType.includes('MALICIOUS')) {
                findings.leaks.push(finding);
            } else if (eventType.includes('WEBSERVER') || eventType.includes('PROVIDER') || eventType.includes('TECHNOLOGY')) {
                findings.infrastructure.push(finding);
            } else if (eventType === 'RAW_RIR_DATA' || eventType === 'GEOINFO' || eventType === 'RAW_DNS_RECORDS') {
                // Parse registry/geo JSON data into structured format
                try {
                    const parsed = JSON.parse(cleanData);
                    const geoEntry = {
                        type: friendlyType,
                        rawType: eventType,
                        source,
                        timestamp,
                        city: parsed.city || null,
                        country: parsed.country || null,
                        asn: parsed.as || null,
                        asName: parsed.asname || null,
                        asDesc: parsed.asdesc || parsed.whoisdesc || null,
                        bgpRoute: parsed.bgproute || null,
                        passiveDNS: (parsed.pas || []).map(p => ({
                            domain: p.o,
                            lastSeen: p.t ? new Date(p.t * 1000).toISOString() : null
                        })),
                        raw: cleanData
                    };
                    findings.geoNetwork.push(geoEntry);
                } catch (e) {
                    // Not JSON — push as regular finding
                    finding.data = cleanData;
                    findings.geoNetwork.push(finding);
                }
            } else if (eventType !== 'ROOT') {
                findings.other.push(finding);
            }
        });

        // Build summary stats
        const stats = {
            totalFindings: events.length,
            accounts: findings.accounts.length,
            personal: findings.personal.length,
            emails: findings.emails.length,
            domains: findings.domains.length,
            ipAddresses: findings.ipAddresses.length,
            leaks: findings.leaks.length,
            infrastructure: findings.infrastructure.length,
            geoNetwork: findings.geoNetwork.length,
            other: findings.other.length
        };

        console.log(`[SpiderFoot] Found ${stats.totalFindings} total findings`);

        res.json({
            scanId: id,
            stats,
            findings,
            summary: summary.slice(0, 20),
            analyzedAt: new Date().toISOString()
        });

    } catch (error) {
        console.error('[SpiderFoot] Error getting results:', error.message);
        res.status(500).json({ error: 'Failed to get scan results', message: error.message });
    }
});

/**
 * List all scans
 * GET /api/profile-scans
 */
app.get('/api/profile-scans', async (req, res) => {
    try {
        const response = await axios.get(`${SPIDERFOOT_URL}/scanlist`);
        const scans = response.data || [];

        // Format scan list
        // SpiderFoot format: [id, name, target, startTime, endTime, finishTime, status, numResults, riskLevels]
        const formattedScans = scans.map(scan => ({
            id: scan[0],
            name: scan[1],
            target: scan[2],
            startTime: scan[3],
            endTime: scan[4],
            finishTime: scan[5],
            status: scan[6] || 'UNKNOWN',
            numResults: scan[7] || 0
        }));

        res.json({ scans: formattedScans });

    } catch (error) {
        console.error('[SpiderFoot] Error listing scans:', error.message);
        res.status(500).json({ error: 'Failed to list scans', message: error.message });
    }
});

/**
 * Delete a scan
 * DELETE /api/profile-scan/:id
 */
app.delete('/api/profile-scan/:id', async (req, res) => {
    try {
        const { id } = req.params;

        await axios.get(`${SPIDERFOOT_URL}/scandelete?id=${id}&confirm=1`);

        res.json({ success: true, message: 'Scan deleted successfully' });

    } catch (error) {
        console.error('[SpiderFoot] Error deleting scan:', error.message);
        res.status(500).json({ error: 'Failed to delete scan', message: error.message });
    }
});

/**
 * Stop/Abort a running scan
 * POST /api/profile-scan/:id/stop
 */
app.post('/api/profile-scan/:id/stop', async (req, res) => {
    try {
        const { id } = req.params;

        console.log(`[SpiderFoot] Stopping scan: ${id}`);

        // SpiderFoot uses GET request with id parameter to stop scans
        await axios.get(`${SPIDERFOOT_URL}/scanstop?id=${id}`);

        res.json({
            success: true,
            scanId: id,
            message: 'Scan stopped successfully',
            status: 'ABORTED'
        });

    } catch (error) {
        console.error('[SpiderFoot] Error stopping scan:', error.message);
        res.status(500).json({ error: 'Failed to stop scan', message: error.message });
    }
});

// ==================== SPIDERFOOT EVENT TYPES ====================

app.get('/api/spiderfoot/eventtypes', (req, res) => {
    const eventTypes = {
        identity: {
            label: 'Identity & Social', icon: '👤',
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
            label: 'Infrastructure', icon: '🌐',
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
            label: 'Security & Threats', icon: '🔒',
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
    res.json(eventTypes);
});

// ==================== SUPABASE INTEGRATION ====================

// Helper to get request-scoped Supabase client
const getSupabaseClient = (req) => {
    const authHeader = req.headers.authorization;
    if (authHeader) {
        return createClient(
            process.env.SUPABASE_URL,
            process.env.SUPABASE_ANON_KEY,
            { global: { headers: { Authorization: authHeader } } }
        );
    }
    return supabase; // Fallback to anon client (will fail RLS)
};

/**
 * Save scan results to Supabase and extract timeline events
 * POST /api/save-scan
 * Body: { investigationId, scanType, target, results, userId }
 */
app.post('/api/save-scan', async (req, res) => {
    try {
        const { investigationId, scanType, target, results, userId } = req.body;

        // Use request-scoped client to respect RLS
        const scopedSupabase = getSupabaseClient(req);

        if (!investigationId || !scanType || !target) {
            return res.status(400).json({ error: 'investigationId, scanType, and target are required' });
        }

        console.log(`[Supabase] Saving ${scanType} scan for: ${target}`);

        // Save scan
        const { data: scan, error: scanError } = await scopedSupabase
            .from('scans')
            .insert({
                investigation_id: investigationId,
                scan_type: scanType,
                target: target,
                status: 'completed',
                raw_results: results || {}
            })
            .select()
            .single();

        if (scanError) throw scanError;

        // Extract events from scan results
        let events = [];

        if (scanType === 'spiderfoot' && results?.findings) {
            events = extractSpiderFootEvents(scan.id, results);
        } else if (scanType === 'phishing' && results) {
            events = extractPhishingEvents(scan.id, target, results);
        } else if (scanType === 'manual' && (results.sources || results.success)) {
            events = extractBreachEvents(scan.id, results);
        }

        // Insert events
        if (events.length > 0) {
            const { error: eventsError } = await scopedSupabase
                .from('events')
                .insert(events);

            if (eventsError) {
                console.error('[Supabase] Error inserting events:', eventsError);
            }
        }

        console.log(`[Supabase] Saved scan ${scan.id} with ${events.length} events`);

        res.json({
            success: true,
            scanId: scan.id,
            eventsCreated: events.length
        });

    } catch (error) {
        console.error('[Supabase] Error saving scan:', error.message);
        res.status(500).json({ error: 'Failed to save scan', message: error.message });
    }
});

/**
 * Extract timeline events from Breach Results (LeakCheck)
 */
function extractBreachEvents(scanId, breachResults) {
    const events = [];
    if (breachResults && breachResults.sources) {
        breachResults.sources.forEach(source => {
            events.push({
                scan_id: scanId,
                event_type: 'security',
                title: `Credential Leak: ${source.name || 'Unknown Source'}`,
                description: `Email found in data breach. Date: ${source.date || 'Unknown'}`,
                timestamp: source.date ? new Date(source.date) : new Date(),
                timestamp_estimated: !source.date,
                source: 'LeakCheck',
                confidence: 1.0,
                severity: 'high',
                metadata: { raw: source }
            });
        });
    }
    return events;
}

/**
 * Extract timeline events from SpiderFoot results
 */
function extractSpiderFootEvents(scanId, results) {
    const events = [];
    const findings = results.findings || {};

    // Process accounts found
    if (findings.accounts) {
        findings.accounts.forEach(account => {
            events.push({
                scan_id: scanId,
                event_type: 'identity',
                title: `Account discovered: ${account.data}`,
                description: `External account found via ${account.module || 'OSINT'}`,
                timestamp: account.timestamp ? new Date(account.timestamp) : new Date(),
                timestamp_estimated: !account.timestamp,
                source: 'SpiderFoot',
                confidence: 0.8,
                severity: 'medium',
                metadata: { raw: account }
            });
        });
    }

    // Process emails
    if (findings.emails) {
        findings.emails.forEach(email => {
            events.push({
                scan_id: scanId,
                event_type: 'identity',
                title: `Email address found: ${email.data}`,
                description: `Email discovered during OSINT scan`,
                timestamp: email.timestamp ? new Date(email.timestamp) : new Date(),
                timestamp_estimated: !email.timestamp,
                source: 'SpiderFoot',
                confidence: 0.9,
                severity: 'low',
                metadata: { raw: email }
            });
        });
    }

    // Process security-related findings (leaks, breaches)
    if (findings.leaks) {
        findings.leaks.forEach(leak => {
            events.push({
                scan_id: scanId,
                event_type: 'security',
                title: `Security incident: ${leak.data}`,
                description: `Potential data breach or leak detected`,
                timestamp: leak.timestamp ? new Date(leak.timestamp) : new Date(),
                timestamp_estimated: !leak.timestamp,
                source: 'SpiderFoot',
                confidence: 0.7,
                severity: 'high',
                metadata: { raw: leak }
            });
        });
    }

    // Process infrastructure
    if (findings.domains) {
        findings.domains.forEach(domain => {
            events.push({
                scan_id: scanId,
                event_type: 'infrastructure',
                title: `Domain linked: ${domain.data}`,
                description: `Internet domain associated with target`,
                timestamp: domain.timestamp ? new Date(domain.timestamp) : new Date(),
                timestamp_estimated: true,
                source: 'SpiderFoot',
                confidence: 0.6,
                severity: 'info',
                metadata: { raw: domain }
            });
        });
    }

    // Process IP addresses
    if (findings.ipAddresses) {
        findings.ipAddresses.forEach(ip => {
            events.push({
                scan_id: scanId,
                event_type: 'infrastructure',
                title: `IP address found: ${ip.data}`,
                description: `IP address associated with target infrastructure`,
                timestamp: ip.timestamp ? new Date(ip.timestamp) : new Date(),
                timestamp_estimated: true,
                source: 'SpiderFoot',
                confidence: 0.7,
                severity: 'info',
                metadata: { raw: ip }
            });
        });
    }

    // Process geo/network intelligence (RAW_RIR_DATA, GEOINFO)
    if (findings.geoNetwork) {
        findings.geoNetwork.forEach(geo => {
            // Build a human-readable title from structured data
            const parts = [];
            if (geo.city && geo.country) parts.push(`${geo.city}, ${geo.country}`);
            else if (geo.country) parts.push(geo.country);
            if (geo.asName) parts.push(`AS${geo.asn} (${geo.asName})`);
            if (geo.passiveDNS?.length > 0) parts.push(`${geo.passiveDNS.length} passive DNS records`);
            const title = parts.length > 0
                ? `Network intelligence: ${parts.join(' · ')}`
                : `Registry data discovered`;

            const descParts = [];
            if (geo.asDesc) descParts.push(geo.asDesc);
            if (geo.bgpRoute) descParts.push(`BGP: ${geo.bgpRoute}`);
            if (geo.passiveDNS?.length > 0) {
                descParts.push(`Related domains: ${geo.passiveDNS.map(d => d.domain).join(', ')}`);
            }

            events.push({
                scan_id: scanId,
                event_type: 'infrastructure',
                title: title,
                description: descParts.join(' | ') || 'Raw registry/network data from OSINT scan',
                timestamp: geo.timestamp ? new Date(geo.timestamp) : new Date(),
                timestamp_estimated: true,
                source: 'SpiderFoot',
                confidence: 0.8,
                severity: 'low',
                metadata: {
                    city: geo.city,
                    country: geo.country,
                    asn: geo.asn,
                    asName: geo.asName,
                    asDesc: geo.asDesc,
                    bgpRoute: geo.bgpRoute,
                    passiveDNS: geo.passiveDNS,
                    raw: geo.raw || geo.data
                }
            });
        });
    }

    // Process personal info (usernames, names, phone numbers)
    if (findings.personal) {
        findings.personal.forEach(item => {
            events.push({
                scan_id: scanId,
                event_type: 'identity',
                title: `Personal info: ${item.data}`,
                description: `${item.type} identified during OSINT scan`,
                timestamp: item.timestamp ? new Date(item.timestamp) : new Date(),
                timestamp_estimated: !item.timestamp,
                source: 'SpiderFoot',
                confidence: 0.7,
                severity: 'medium',
                metadata: { raw: item }
            });
        });
    }

    // Process infrastructure (web servers, DNS providers, technologies)
    if (findings.infrastructure) {
        findings.infrastructure.forEach(item => {
            events.push({
                scan_id: scanId,
                event_type: 'infrastructure',
                title: `Infrastructure: ${item.data}`,
                description: `${item.type} detected via OSINT scan`,
                timestamp: item.timestamp ? new Date(item.timestamp) : new Date(),
                timestamp_estimated: true,
                source: 'SpiderFoot',
                confidence: 0.6,
                severity: 'info',
                metadata: { raw: item }
            });
        });
    }

    return events;
}

/**
 * Extract timeline events from phishing scan results
 */
function extractPhishingEvents(scanId, target, results) {
    const events = [];

    events.push({
        scan_id: scanId,
        event_type: 'security',
        title: `URL scanned: ${target}`,
        description: `Phishing analysis — ${results.isPhishing ? 'MALICIOUS' : 'SAFE'} (${results.confidence || 0}% confidence)`,
        timestamp: new Date(),
        timestamp_estimated: false,
        source: results.sources?.join(', ') || 'VirusTotal + URLhaus',
        confidence: (results.confidence || 0) / 100,
        severity: results.isPhishing ? 'critical' : 'info',
        metadata: {
            isPhishing: results.isPhishing,
            confidence: results.confidence,
            indicators: results.indicators,
            recommendation: results.recommendation
        }
    });

    return events;
}

/**
 * Get all events for an investigation (for timeline)
 * GET /api/investigations/:id/events
 */
app.get('/api/investigations/:id/events', async (req, res) => {
    try {
        const { id } = req.params;

        // Get all scans for this investigation
        const { data: scans, error: scansError } = await supabase
            .from('scans')
            .select('id')
            .eq('investigation_id', id);

        if (scansError) throw scansError;

        if (!scans || scans.length === 0) {
            return res.json({ events: [], count: 0 });
        }

        const scanIds = scans.map(s => s.id);

        // Get all events for those scans
        const { data: events, error: eventsError } = await supabase
            .from('events')
            .select('*')
            .in('scan_id', scanIds)
            .order('timestamp', { ascending: true });

        if (eventsError) throw eventsError;

        res.json({
            events: events || [],
            count: (events || []).length
        });

    } catch (error) {
        console.error('[Supabase] Error fetching events:', error.message);
        res.status(500).json({ error: 'Failed to fetch events', message: error.message });
    }
});

/**
 * Check for data breaches using LeakCheck Public API
 * GET /api/check-leak/:query
 */
app.get('/api/check-leak/:query', async (req, res) => {
    try {
        const { query } = req.params;
        if (!query) return res.status(400).json({ error: 'Query is required' });

        console.log(`[LeakCheck] Checking: ${query}`);

        // LeakCheck Public API
        const response = await axios.get(`https://leakcheck.io/api/public?check=${encodeURIComponent(query)}`);

        if (response.data.success) {
            res.json(response.data);
        } else {
            // Even if success is false, it might return a reason or empty result
            res.json({ success: false, sources: [], message: response.data.error || 'No data found' });
        }
    } catch (error) {
        console.error('[LeakCheck] Error:', error.message);
        // Public API might rate limit or block 
        res.status(500).json({ error: 'LeakCheck lookup failed', details: error.message });
    }
});


const multer = require('multer');
const { spawn } = require('child_process');

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, 'uploads');
        if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const uniqueName = `df_${Date.now()}_${file.originalname}`;
        cb(null, uniqueName);
    }
});

const upload = multer({
    storage,
    limits: { fileSize: 20 * 1024 * 1024 }, // 20MB max
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed'), false);
        }
    }
});

/**
 * Analyze an uploaded image for deepfake/manipulation indicators
 * POST /api/analyze-deepfake
 * Body: multipart/form-data with 'file' field
 * 
 * Pipeline:
 * 1. Upload saved to backend/uploads/
 * 2. Python script runs ELA + EXIF + Noise analysis
 * 3. Returns full forensic report with ELA heatmap
 */
/*
// DEEPFAKE ANALYSIS ENDPOINT (DISABLED PER USER REQUEST - REVERTED TO SIMULATION)
app.post('/api/analyze-deepfake', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }
 
        const filePath = req.file.path;
        console.log(`\n[Deepfake] Analyzing: ${req.file.originalname} (${(req.file.size / 1024).toFixed(1)} KB)`);
 
        // Spawn Python analyzer
        const pythonScript = path.join(__dirname, 'deepfake_analyzer.py');
 
        const result = await new Promise((resolve, reject) => {
            const python = spawn('python', [pythonScript, filePath]);
            let stdout = '';
            let stderr = '';
 
            python.stdout.on('data', (data) => {
                stdout += data.toString();
            });
 
            python.stderr.on('data', (data) => {
                stderr += data.toString();
                // Log progress messages
                const lines = data.toString().split('\n').filter(l => l.trim());
                lines.forEach(line => console.log(`  ${line}`));
            });
 
            python.on('close', (code) => {
                if (code !== 0) {
                    reject(new Error(`Python exited with code ${code}: ${stderr}`));
                } else {
                    try {
                        resolve(JSON.parse(stdout));
                    } catch (e) {
                        reject(new Error(`Failed to parse Python output: ${stdout.substring(0, 200)}`));
                    }
                }
            });
 
            // Timeout after 30 seconds
            setTimeout(() => {
                python.kill();
                reject(new Error('Analysis timed out after 30s'));
            }, 30000);
        });
 
        // Clean up
        fs.unlink(filePath, () => {});
 
        res.json(result);
 
    } catch (error) {
        console.error(`[Deepfake] Error: ${error.message}`);
        // Clean up
        if (req.file) fs.unlink(req.file.path, () => {});
        res.status(500).json({ error: error.message });
    }
});
*/

app.listen(PORT, '127.0.0.1', () => {
    console.log(`
╔══════════════════════════════════════════════════════════════╗
║           Silent Trails Backend Server v5.0                  ║
║    Threat Intelligence + OSINT Reconnaissance                ║
╠══════════════════════════════════════════════════════════════╣
║  🌐 Server:     http://localhost:${PORT}                        ║
║  ❤️  Health:     http://localhost:${PORT}/api/health             ║
╠══════════════════════════════════════════════════════════════╣
║  📡 Active Services:                                         ║
║     • VirusTotal (70+ security vendors)                      ║
║     • URLhaus (Malware URL database)                         ║
║     • SpiderFoot (200+ OSINT modules)                        ║
║     • Supabase (Database + Auth)                             ║
╚══════════════════════════════════════════════════════════════╝
  `);
});


