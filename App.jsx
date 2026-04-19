import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

import IntelligenceScore from './components/IntelligenceScore';
import ScoreBreakdown from './components/ScoreBreakdown';
import SentimentTimeline from './components/SentimentTimeline';
import SpeakerAnalysis from './components/SpeakerAnalysis';
import TopicCoverage from './components/TopicCoverage';
import LiveTranscription from './components/LiveTranscription';
import VideoEngagement from './components/VideoEngagement';
import SarcasmAlerts from './components/SarcasmAlerts';
import UploadPanel from './components/UploadPanel';
import { fetchDemo, uploadAudio, uploadFull } from './services/api';

const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'transcript', label: 'Transcript' },
    { id: 'analysis', label: 'Linguistic Analysis' },
    { id: 'video', label: 'Visual Engagement' },
    { id: 'alerts', label: 'Behavioral Alerts' },
];

export default function App() {
    const [report, setReport] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('overview');
    const [sessionId, setSessionId] = useState(null);

    const handleAnalyze = useCallback(async (audioFile, videoFile, isDemo = false) => {
        setLoading(true);
        setError(null);
        try {
            let result;
            if (isDemo) {
                result = await fetchDemo();
            } else if (videoFile && audioFile) {
                // Multimodal analysis (usually the same MP4 file)
                result = await uploadFull(audioFile, videoFile);
            } else if (audioFile) {
                result = await uploadAudio(audioFile);
            } else if (videoFile) {
                result = await uploadFull(videoFile, videoFile);
            } else {
                throw new Error('No valid recording provided.');
            }
            setReport(result.report);
            setSessionId(result.session_id);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, []);

    const handleReset = () => {
        setReport(null);
        setSessionId(null);
        setError(null);
        setActiveTab('overview');
    };

    // Data extraction
    const segments = report?.segments || [];
    const scores = report?.candidate_scores || {};
    const engagement = report?.engagement_score || 0;
    const topicCoverage = report?.topic_coverage || null;
    const clarifications = report?.clarifications || [];
    const sarcasmEvents = report?.sarcasm_events || [];
    const videoMetrics = report?.video_metrics || null;
    const visualScore = report?.visual_engagement_score || 0;
    const intelligence = report?.intelligence_score || null;

    return (
        <div className="app">
            <header className="header">
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <h1>Session Intelligence</h1>
                    <p>Deep behavioral and linguistic analysis for professional recordings.</p>
                </motion.div>
                
                {sessionId && (
                    <div className="session-tag-container">
                        <span className="token">ID: {sessionId}</span>
                        <button className="reset-link" onClick={handleReset}>New Analysis</button>
                    </div>
                )}
            </header>

            {!report && !loading && (
                <UploadPanel onAnalyze={handleAnalyze} loading={loading} />
            )}

            {loading && (
                <div className="loading">
                    <div className="spinner" />
                    <p className="loading-text">Generating Intelligence Report...</p>
                </div>
            )}

            {error && !loading && (
                <div className="error-banner">
                    <div className="error-icon">⚠️</div>
                    <div className="error-content">
                        <strong>Analysis Failed</strong>
                        <p>{error}</p>
                    </div>
                    <button onClick={() => setError(null)}>Dismiss</button>
                </div>
            )}

            {report && (
                <div className="dashboard-layout">
                    <nav className="tabs">
                        {TABS.map(tab => (
                            <button
                                key={tab.id}
                                className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                                onClick={() => setActiveTab(tab.id)}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </nav>

                    <AnimatePresence mode="wait">
                        {activeTab === 'overview' && (
                            <motion.div key="overview" initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -10 }}>
                                <div className="metrics-grid">
                                    <MetricCard label="Intelligence" value={intelligence?.composite_score || 0} sub={intelligence?.label || 'Score'} color="blue" />
                                    <MetricCard label="Candidate Performance" value={scores.final_candidate_score || 0} sub="/100" color="purple" />
                                    <MetricCard label="Interaction depth" value={`${engagement}%`} sub="Q/A Ratio" color="green" />
                                    <MetricCard label="Content Coverage" value={`${topicCoverage?.coverage_pct || 0}%`} sub="Topics hit" color="cyan" />
                                </div>

                                <div className="grid-2">
                                    <IntelligenceScore data={intelligence} />
                                    <ScoreBreakdown scores={scores} />
                                </div>

                                <SentimentTimeline segments={segments} />
                            </motion.div>
                        )}

                        {activeTab === 'transcript' && (
                            <motion.div key="transcript" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                <LiveTranscription segments={segments} />
                            </motion.div>
                        )}

                        {activeTab === 'analysis' && (
                            <motion.div key="analysis" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                <div className="grid-2">
                                    <SpeakerAnalysis segments={segments} />
                                    <TopicCoverage coverage={topicCoverage} />
                                </div>
                                <div style={{ height: '24px' }} />
                                <ScoreBreakdown scores={scores} />
                            </motion.div>
                        )}

                        {activeTab === 'video' && (
                            <motion.div key="video" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                <VideoEngagement videoMetrics={videoMetrics} visualScore={visualScore} />
                            </motion.div>
                        )}

                        {activeTab === 'alerts' && (
                            <motion.div key="alerts" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                <SarcasmAlerts events={sarcasmEvents} />
                                {clarifications.length > 0 && (
                                    <div className="card" style={{ marginTop: '24px' }}>
                                        <div className="card-title">Clarification Requests</div>
                                        <div className="alert-feed">
                                            {clarifications.map((c, i) => (
                                                <div key={i} className={`alert-item ${c.resolved ? 'resolved' : 'unresolved'}`}>
                                                    <div className="alert-header">
                                                        <span className="speaker">{c.speaker}</span>
                                                        <span className="status">{c.resolved ? '✓ Resolved' : '!? Pending'}</span>
                                                    </div>
                                                    <p className="alert-text">"{c.confusion_text}"</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            )}
        </div>
    );
}

function MetricCard({ label, value, sub, color }) {
    return (
        <div className={`metric-card metric-${color}`}>
            <div className="label">{label}</div>
            <div className="value">{value}</div>
            <div className="sub">{sub}</div>
        </div>
    );
}
