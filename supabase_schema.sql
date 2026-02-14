-- ==========================================
-- Silent Trails Database Schema
-- Run this in Supabase SQL Editor
-- ==========================================

-- 1. Investigations Table
-- Each investigation is a case/target being analyzed
CREATE TABLE investigations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    target TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'archived')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Scans Table
-- Each scan performed under an investigation
CREATE TABLE scans (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    investigation_id UUID REFERENCES investigations(id) ON DELETE CASCADE NOT NULL,
    scan_type TEXT NOT NULL CHECK (scan_type IN ('spiderfoot', 'phishing', 'deepfake', 'manual')),
    target TEXT NOT NULL,
    status TEXT DEFAULT 'completed' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    raw_results JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Events Table (Core for Timeline Engine)
-- Normalized events extracted from scan results
CREATE TABLE events (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    scan_id UUID REFERENCES scans(id) ON DELETE CASCADE NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('identity', 'security', 'infrastructure')),
    title TEXT NOT NULL,
    description TEXT,
    timestamp TIMESTAMPTZ,
    timestamp_estimated BOOLEAN DEFAULT FALSE,
    source TEXT NOT NULL,
    confidence REAL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    severity TEXT DEFAULT 'info' CHECK (severity IN ('critical', 'high', 'medium', 'low', 'info')),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ==========================================
-- Row Level Security (RLS)
-- Users can only see their OWN data
-- ==========================================

ALTER TABLE investigations ENABLE ROW LEVEL SECURITY;
ALTER TABLE scans ENABLE ROW LEVEL SECURITY;
ALTER TABLE events ENABLE ROW LEVEL SECURITY;

-- Investigation policies
CREATE POLICY "Users can view own investigations"
    ON investigations FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can create own investigations"
    ON investigations FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own investigations"
    ON investigations FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own investigations"
    ON investigations FOR DELETE
    USING (auth.uid() = user_id);

-- Scan policies (via investigation ownership)
CREATE POLICY "Users can view own scans"
    ON scans FOR SELECT
    USING (
        investigation_id IN (
            SELECT id FROM investigations WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can create scans in own investigations"
    ON scans FOR INSERT
    WITH CHECK (
        investigation_id IN (
            SELECT id FROM investigations WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can delete own scans"
    ON scans FOR DELETE
    USING (
        investigation_id IN (
            SELECT id FROM investigations WHERE user_id = auth.uid()
        )
    );

-- Event policies (via scan → investigation ownership)
CREATE POLICY "Users can view own events"
    ON events FOR SELECT
    USING (
        scan_id IN (
            SELECT s.id FROM scans s
            JOIN investigations i ON s.investigation_id = i.id
            WHERE i.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can create events in own scans"
    ON events FOR INSERT
    WITH CHECK (
        scan_id IN (
            SELECT s.id FROM scans s
            JOIN investigations i ON s.investigation_id = i.id
            WHERE i.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can delete own events"
    ON events FOR DELETE
    USING (
        scan_id IN (
            SELECT s.id FROM scans s
            JOIN investigations i ON s.investigation_id = i.id
            WHERE i.user_id = auth.uid()
        )
    );

-- ==========================================
-- Indexes for Performance
-- ==========================================
CREATE INDEX idx_investigations_user ON investigations(user_id);
CREATE INDEX idx_scans_investigation ON scans(investigation_id);
CREATE INDEX idx_events_scan ON events(scan_id);
CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_events_type ON events(event_type);
