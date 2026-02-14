// Silent Trails - API Service
// Handles both mock responses and real API calls

const BACKEND_URL = 'http://localhost:3001';

// Check if backend is available
const checkBackendHealth = async () => {
  try {
    const response = await fetch(`${BACKEND_URL}/api/health`);
    return response.ok;
  } catch {
    return false;
  }
};

// Simulate network delay for mock responses
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
const randomDelay = (min = 800, max = 1500) => {
  return delay(Math.floor(Math.random() * (max - min + 1)) + min);
};

/**
 * Email Leak & Exposure Analysis (Mock - no real API yet)
 * Can be replaced with HaveIBeenPwned API in future
 */
export const analyzeEmail = async (email) => {
  await randomDelay(1000, 2000);

  const mockBreaches = [
    { name: "LinkedIn", date: "2021-04-15", dataTypes: ["email", "password", "username"], severity: "high" },
    { name: "Adobe", date: "2019-10-26", dataTypes: ["email", "username", "encrypted password"], severity: "medium" },
    { name: "Dropbox", date: "2016-08-31", dataTypes: ["email", "password"], severity: "high" },
    { name: "MyFitnessPal", date: "2018-02-01", dataTypes: ["email", "username", "IP address"], severity: "medium" },
    { name: "Canva", date: "2019-05-24", dataTypes: ["email", "username", "location"], severity: "low" }
  ];

  let breachCount;
  if (email.includes("admin") || email.includes("test")) {
    breachCount = Math.floor(Math.random() * 3) + 3;
  } else if (email.includes("secure") || email.includes("safe")) {
    breachCount = 0;
  } else {
    breachCount = Math.floor(Math.random() * 4);
  }

  const selectedBreaches = mockBreaches.slice(0, breachCount);
  let riskLevel = breachCount === 0 ? "low" : breachCount <= 2 ? "medium" : "high";

  return {
    email,
    breachCount,
    breaches: selectedBreaches,
    riskLevel,
    lastChecked: new Date().toISOString(),
    recommendations: breachCount > 0 ? [
      "Change passwords for affected accounts immediately",
      "Enable two-factor authentication where available",
      "Monitor your accounts for suspicious activity",
      "Consider using a password manager"
    ] : [
      "Continue using strong, unique passwords",
      "Keep two-factor authentication enabled",
      "Regularly monitor your digital footprint"
    ]
  };
};

/**
 * Phishing URL & Message Detection
 * Uses real VirusTotal API via backend
 */
export const analyzePhishing = async (input) => {
  // Check if backend is available
  const backendAvailable = await checkBackendHealth();

  if (backendAvailable) {
    // Use real VirusTotal API
    return analyzePhishingReal(input);
  } else {
    // Fallback to mock response
    console.warn('Backend not available, using mock response');
    return analyzePhishingMock(input);
  }
};

/**
 * Real VirusTotal API call via backend
 */
const analyzePhishingReal = async (input) => {
  try {
    // Determine if input is URL or message
    const isUrl = input.startsWith('http://') || input.startsWith('https://') ||
      input.match(/^[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,}/);

    const endpoint = isUrl ? '/api/check-url' : '/api/check-message';
    const body = isUrl ? { url: input } : { message: input };

    const response = await fetch(`${BACKEND_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      throw new Error('API request failed');
    }

    const data = await response.json();
    return data;

  } catch (error) {
    console.error('VirusTotal API error:', error);
    // Return error state
    return {
      input: input.substring(0, 100) + (input.length > 100 ? '...' : ''),
      isPhishing: null,
      confidence: 0,
      riskLevel: 'unknown',
      indicators: ['API Error: ' + error.message],
      recommendation: 'Could not analyze. Please ensure the backend server is running.',
      analyzedAt: new Date().toISOString(),
      source: 'Error'
    };
  }
};

/**
 * Mock phishing analysis (fallback)
 */
const analyzePhishingMock = async (input) => {
  await randomDelay(1200, 2500);

  const lowerInput = input.toLowerCase();

  const suspiciousPatterns = [
    { pattern: "bit.ly", indicator: "URL shortening service detected" },
    { pattern: "tinyurl", indicator: "URL shortening service detected" },
    { pattern: "login", indicator: "Login-related keywords found" },
    { pattern: "password", indicator: "Password-related keywords found" },
    { pattern: "verify", indicator: "Verification request detected" },
    { pattern: "urgent", indicator: "Urgency language detected" },
    { pattern: "suspended", indicator: "Account suspension threat detected" },
    { pattern: "click here", indicator: "Generic call-to-action detected" },
    { pattern: "winner", indicator: "Prize/winner claim detected" },
    { pattern: "http://", indicator: "Non-secure HTTP protocol" }
  ];

  const foundIndicators = suspiciousPatterns
    .filter(p => lowerInput.includes(p.pattern))
    .map(p => p.indicator);

  const uniqueIndicators = [...new Set(foundIndicators)];
  const baseConfidence = uniqueIndicators.length * 15;
  const confidence = Math.min(95, Math.max(10, baseConfidence + Math.random() * 20));
  const isPhishing = uniqueIndicators.length >= 2 || confidence > 50;

  return {
    input: input.substring(0, 100) + (input.length > 100 ? "..." : ""),
    isPhishing,
    confidence: Math.round(confidence),
    riskLevel: confidence > 70 ? "high" : confidence > 40 ? "medium" : "low",
    indicators: uniqueIndicators.length > 0 ? uniqueIndicators : ["No suspicious patterns detected"],
    recommendation: isPhishing
      ? "This appears to be a phishing attempt. Do not click any links."
      : "This content appears to be safe, but always exercise caution.",
    analyzedAt: new Date().toISOString(),
    source: 'Mock (Backend Offline)'
  };
};

/**
 * Social Media Presence Mapping
 * Uses SpiderFoot OSINT via backend when available
 */
export const analyzeSocialMedia = async (username) => {
  // Check if backend is available
  const backendAvailable = await checkBackendHealth();

  if (backendAvailable) {
    // Use real SpiderFoot API
    return analyzeSocialMediaReal(username);
  } else {
    // Fallback to mock response
    console.warn('Backend not available, using mock response');
    return analyzeSocialMediaMock(username);
  }
};

/**
 * Real SpiderFoot OSINT scan via backend
 */
const analyzeSocialMediaReal = async (username) => {
  try {
    // Start a SpiderFoot scan
    const startResponse = await fetch(`${BACKEND_URL}/api/profile-scan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target: username, scanType: 'footprint' })
    });

    if (!startResponse.ok) {
      throw new Error('Failed to start scan');
    }

    const startData = await startResponse.json();
    const scanId = startData.scanId;

    // Poll for results (SpiderFoot scans take time, we'll get partial results)
    // Wait a few seconds for initial results
    await new Promise(resolve => setTimeout(resolve, 5000));

    // Get scan status
    const statusResponse = await fetch(`${BACKEND_URL}/api/profile-scan/${scanId}/status`);
    const statusData = await statusResponse.json();

    // Get available results
    const resultsResponse = await fetch(`${BACKEND_URL}/api/profile-scan/${scanId}/results`);
    const resultsData = await resultsResponse.json();

    // Transform SpiderFoot results to our format
    const platforms = transformSpiderFootResults(resultsData, username);

    return {
      username,
      scanId,
      platforms,
      totalFound: platforms.filter(p => p.found).length,
      totalSearched: platforms.length,
      digitalFootprintScore: platforms.filter(p => p.found).length * 10,
      searchedAt: new Date().toISOString(),
      source: 'SpiderFoot',
      scanStatus: statusData.status,
      rawFindings: resultsData.stats
    };

  } catch (error) {
    console.error('SpiderFoot API error:', error);
    // Return error state with fallback to mock
    return analyzeSocialMediaMock(username);
  }
};

/**
 * Transform SpiderFoot findings to platform format
 */
const transformSpiderFootResults = (results, username) => {
  const platformMap = [
    { name: 'Twitter/X', icon: '🐦', patterns: ['twitter', 'x.com'] },
    { name: 'Instagram', icon: '📸', patterns: ['instagram'] },
    { name: 'Facebook', icon: '👤', patterns: ['facebook'] },
    { name: 'LinkedIn', icon: '💼', patterns: ['linkedin'] },
    { name: 'GitHub', icon: '💻', patterns: ['github'] },
    { name: 'Reddit', icon: '🤖', patterns: ['reddit'] },
    { name: 'TikTok', icon: '🎵', patterns: ['tiktok'] },
    { name: 'YouTube', icon: '▶️', patterns: ['youtube'] },
    { name: 'Pinterest', icon: '📌', patterns: ['pinterest'] },
    { name: 'Twitch', icon: '🎮', patterns: ['twitch'] }
  ];

  // Check each platform in SpiderFoot findings
  return platformMap.map(platform => {
    const found = results.findings?.socialMedia?.some(item =>
      platform.patterns.some(p => item.data?.toLowerCase().includes(p))
    ) ||
      results.findings?.other?.some(item =>
        platform.patterns.some(p => item.data?.toLowerCase().includes(p))
      );

    // Find the matching URL if found
    const matchingItem = results.findings?.socialMedia?.find(item =>
      platform.patterns.some(p => item.data?.toLowerCase().includes(p))
    );

    return {
      name: platform.name,
      icon: platform.icon,
      found,
      url: matchingItem?.data || null,
      confidence: found ? 'Verified by SpiderFoot' : null,
      lastActive: null
    };
  });
};

/**
 * Mock Social Media Analysis (fallback)
 */
const analyzeSocialMediaMock = async (username) => {
  await delay(2000);

  const platforms = [
    { name: 'Twitter/X', icon: '🐦', baseUrl: 'https://twitter.com/', probability: 0.7 },
    { name: 'Instagram', icon: '📸', baseUrl: 'https://instagram.com/', probability: 0.75 },
    { name: 'Facebook', icon: '👤', baseUrl: 'https://facebook.com/', probability: 0.6 },
    { name: 'LinkedIn', icon: '💼', baseUrl: 'https://linkedin.com/in/', probability: 0.5 },
    { name: 'GitHub', icon: '💻', baseUrl: 'https://github.com/', probability: 0.4 },
    { name: 'Reddit', icon: '🤖', baseUrl: 'https://reddit.com/user/', probability: 0.35 },
    { name: 'TikTok', icon: '🎵', baseUrl: 'https://tiktok.com/@', probability: 0.45 },
    { name: 'Pinterest', icon: '📌', baseUrl: 'https://pinterest.com/', probability: 0.3 },
    { name: 'YouTube', icon: '▶️', baseUrl: 'https://youtube.com/@', probability: 0.4 },
    { name: 'Twitch', icon: '🎮', baseUrl: 'https://twitch.tv/', probability: 0.25 }
  ];

  const results = platforms.map(platform => {
    const found = Math.random() < platform.probability;
    return {
      name: platform.name,
      icon: platform.icon,
      found,
      url: found ? `${platform.baseUrl}${username}` : null,
      followers: found ? Math.floor(Math.random() * 10000) : null,
      lastActive: found ? getRandomPastDate() : null
    };
  });

  const totalFound = results.filter(r => r.found).length;

  return {
    username,
    platforms: results,
    totalFound,
    totalSearched: platforms.length,
    digitalFootprintScore: Math.round((totalFound / platforms.length) * 100),
    searchedAt: new Date().toISOString(),
    source: 'Mock (Backend Offline)'
  };
};

const getRandomPastDate = () => {
  const daysAgo = Math.floor(Math.random() * 30) + 1;
  const date = new Date();
  date.setDate(date.getDate() - daysAgo);
  return date.toISOString().split('T')[0];
};

export default {
  analyzeEmail,
  analyzePhishing,
  analyzeSocialMedia
};

