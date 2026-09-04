import React, { useState, useEffect, useRef, useMemo } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend, LineChart, Line, XAxis, YAxis, CartesianGrid, AreaChart, Area } from 'recharts';
import { Upload, CloudUpload, FileText, BarChart3, Zap, BrainCircuit, Star, AlertTriangle, CheckCircle, Search, Compass, Network, MessageSquare, ChevronLeft, TrendingUp, Smile, Heart, ThumbsDown, Info, Database, Brain, Copy, Check, Clock, X } from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const COLORS = ['#10b981', '#f59e0b', '#ef4444']; // Pos, Neu, Neg
const EMOTION_COLORS = {
  joy:     '#10b981', // emerald
  neutral: '#64748b', // slate
  sadness: '#6366f1', // indigo
  anger:   '#ef4444', // red
  fear:    '#f59e0b', // amber
  disgust: '#8b5cf6', // violet
};

// Animates a number counting up to `target` whenever it changes.
function useCountUp(target, duration = 900) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (typeof target !== 'number') return;
    const start = performance.now();
    const from = 0;
    let frame;
    const tick = (now) => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.round(from + (target - from) * eased));
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, duration]);
  return value;
}

function App() {
  const [activeTab, setActiveTab] = useState('ingestion'); // ingestion, portal, owner

  // -- TOASTS --
  const [toasts, setToasts] = useState([]);
  const pushToast = (type, message) => {
    const id = `${Date.now()}-${Math.random()}`;
    setToasts((t) => [...t, { id, type, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000);
  };

  // -- INGESTION STATE --
  const [file, setFile] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [stats, setStats] = useState(null);
  const [emotionDistribution, setEmotionDistribution] = useState([]);
  const [emotionTrajectory, setEmotionTrajectory] = useState([]);

  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphDims, setGraphDims] = useState({ width: 1120, height: 450 });
  const [hoverNode, setHoverNode] = useState(null);
  const graphContainerRef = useRef(null);
  const forceGraphRef = useRef(null);

  // -- PORTAL STATE --
  const [review, setReview] = useState('');
  const [rating, setRating] = useState(3);
  const [prediction, setPrediction] = useState('');
  const [portalLoading, setPortalLoading] = useState(false);
  const [portalResult, setPortalResult] = useState(null);
  const [portalHistory, setPortalHistory] = useState([]);
  const [copied, setCopied] = useState(false);
  const typingTimeoutRef = useRef(null);

  // -- OWNER STATE --
  const [ownerInsights, setOwnerInsights] = useState(null);
  const [ownerLoading, setOwnerLoading] = useState(false);
  const [ownerError, setOwnerError] = useState(null);

  // -- EXECUTIVE SLM CHAT STATE --
  const [activeChatInsight, setActiveChatInsight] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [suggestedLoading, setSuggestedLoading] = useState(false);
  const chatScrollRef = useRef(null);

  useEffect(() => {
    if (chatScrollRef.current) {
       chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages, isChatLoading]);

  // === HANDLERS ===

  // 1. Data Ingestion
  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setUploadLoading(true);
    setUploadStatus('Parsing CSV and running VADER Sentiment Analysis...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/api/upload-dataset`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (!response.ok || data.status === 'error') {
        setUploadStatus(data.detail || data.message || 'Failed to upload file.');
        return;
      }
      setUploadStatus('success');

      if (data.sentiment_distribution) {
        const chartData = [
          { name: 'Positive', value: data.sentiment_distribution.Positive || 0 },
          { name: 'Neutral', value: data.sentiment_distribution.Neutral || 0 },
          { name: 'Negative', value: data.sentiment_distribution.Negative || 0 },
        ];
        setStats({ count: data.vectors_embedded, chartData });
      }

      if (data.emotion_distribution) {
        const emoData = Object.entries(data.emotion_distribution)
          .filter(([, v]) => v > 0)
          .map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value }));
        setEmotionDistribution(emoData);
      }

      await fetchGraphData();
      await fetchEmotionAnalytics();

    } catch {
      setUploadStatus('Failed to upload file.');
    } finally {
      setUploadLoading(false);
    }
  };

  const fetchGraphData = async () => {
    setGraphLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/graph-data`);
      const data = await res.json();
      if (data.nodes && data.nodes.length > 0) {
         setGraphData(data);
      }
    } catch {
      pushToast('error', 'Could not reach the graph service. Is the backend running?');
    } finally {
      setGraphLoading(false);
    }
  };

  const fetchEmotionAnalytics = async () => {
    try {
      const res = await fetch(`${API_URL}/api/emotion-analytics`);
      const data = await res.json();
      if (data.trajectory) {
        setEmotionTrajectory(data.trajectory);
      }
    } catch {
      console.error("Failed to fetch emotion analytics");
    }
  };

  // Keep the force-graph canvas sized to its container instead of a fixed box.
  useEffect(() => {
    const el = graphContainerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setGraphDims({ width, height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // node id -> Set of neighbor ids, rebuilt whenever the graph data changes (for hover highlighting)
  const neighborMap = useMemo(() => {
    const map = new Map();
    graphData.nodes.forEach((n) => map.set(n.id, new Set()));
    graphData.links.forEach((l) => {
      const a = typeof l.source === 'object' ? l.source.id : l.source;
      const b = typeof l.target === 'object' ? l.target.id : l.target;
      map.get(a)?.add(b);
      map.get(b)?.add(a);
    });
    return map;
  }, [graphData]);

  useEffect(() => {
    if (activeTab === 'ingestion') {
      if (graphData.nodes.length === 0) fetchGraphData();
      if (emotionTrajectory.length === 0) fetchEmotionAnalytics();
    }
  }, [activeTab, emotionTrajectory.length, graphData.nodes.length]);

  // 2. Portal & Autocomplete
  const handleReviewChange = (e) => {
    const val = e.target.value;
    setReview(val);
    setPrediction('');

    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);

    if (val.trim().length > 10) {
      typingTimeoutRef.current = setTimeout(async () => {
        try {
          const res = await fetch(`${API_URL}/api/autocomplete`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ text: val })
          });
          const data = await res.json();
          if (data.suggestion) {
            setPrediction(data.suggestion);
          }
        } catch { console.error("Autocomplete failed")}
      }, 800); // 800ms debounce
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Tab' && prediction) {
      e.preventDefault();
      setReview(prev => prev + prediction);
      setPrediction('');
    }
  };

  const handlePortalSubmit = async (e) => {
    e.preventDefault();
    setPortalLoading(true);
    setPrediction('');
    setPortalResult(null);
    try {
      const response = await fetch(`${API_URL}/api/analyze-review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_text: review, rating: parseFloat(rating) })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Analysis failed.');
      }
      setPortalResult(data);
      setPortalHistory((h) => [{ ...data, review_text: review, rating }, ...h].slice(0, 6));
    } catch (err) {
      console.error(err);
      setPortalResult({ error: err.message });
    } finally {
      setPortalLoading(false);
    }
  };

  const handleCopyOutput = async () => {
    if (!portalResult?.final_output) return;
    try {
      await navigator.clipboard.writeText(portalResult.final_output);
      setCopied(true);
      pushToast('success', 'Forge output copied to clipboard.');
      setTimeout(() => setCopied(false), 2000);
    } catch {
      pushToast('error', 'Could not copy — clipboard access was denied.');
    }
  };

  // 3. Owner Insights
  const fetchOwnerInsights = async () => {
    setOwnerLoading(true);
    setOwnerError(null);
    try {
      const response = await fetch(`${API_URL}/api/owner-insights`);
      const data = await response.json();
      if (data.error) {
        setOwnerError(data.error);
        setOwnerInsights([]);
      } else {
        setOwnerInsights(data.insights || []);
      }
    } catch (err) {
      setOwnerError(err.message);
      console.error(err);
    } finally {
      setOwnerLoading(false);
    }
  };

  // 4. Executive Chat Handlers
  const handleOpenChat = (insight) => {
    setActiveChatInsight(insight);
    setChatMessages([
      { role: 'assistant', content: `**Gemma 2 SLM Engine Active.**\n\nI am analyzing the specific context for: \n> "${insight.question || insight.Question}"\n\nWhat disruptive, forward-thinking strategy do you want to explore?` }
    ]);
    setActiveTab('advisor');
  };

  const handleSendChat = async (e, contentOverride = null) => {
    if (e) e.preventDefault();
    const query = contentOverride || chatInput;
    if (!query.trim()) return;

    const newMsg = { role: 'user', content: query };
    const msgs = [...chatMessages, newMsg];
    setChatMessages(msgs);
    setChatInput('');
    setIsChatLoading(true);

    try {
      const resp = await fetch(`${API_URL}/api/owner-chat`, {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({
            query: newMsg.content,
            topic_context: activeChatInsight || {},
            messages: msgs.slice(0, -1)
         })
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || 'The advisor failed to respond.');
      }
      setChatMessages([...msgs, { role: 'assistant', content: data.response }]);
    } catch(err) {
      console.error(err);
      setChatMessages([...msgs, { role: 'assistant', content: `⚠️ ${err.message}` }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const fetchSuggestedQuestions = async () => {
    setSuggestedLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/suggested-questions`);
      const data = await res.json();
      setSuggestedQuestions(data.questions || []);
    } catch (err) {
      console.error(err);
    } finally {
      setSuggestedLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'owner' && !ownerInsights) {
      fetchOwnerInsights();
    }
    if (activeTab === 'advisor' && suggestedQuestions.length === 0) {
      fetchSuggestedQuestions();
    }
  }, [activeTab, ownerInsights, suggestedQuestions.length]);

  const animatedCount = useCountUp(stats?.count ?? 0);

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans flex flex-col">
      {/* Navbar */}
      <nav className="bg-[#1e293b] border-b border-slate-700/50 p-4 shadow-sm relative z-20">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Compass className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-cyan-400">
              FeedbackForge Enterprise
            </h1>
          </div>
          <div className="text-xs font-semibold tracking-wider text-emerald-400 border border-emerald-500/30 px-4 py-1.5 rounded-full bg-emerald-500/10 tabular-nums">
            {stats ? `ACTIVE VECTORS: ${animatedCount}` : 'AWAITING DATA'}
          </div>
        </div>
      </nav>

      {/* Tabs */}
      <div className="bg-[#1e293b]/50 border-b border-slate-700/50">
        <div className="max-w-7xl mx-auto flex space-x-8 px-6">
          <button onClick={() => setActiveTab('ingestion')} className={`flex items-center gap-2 py-4 px-2 font-medium transition-colors ${activeTab === 'ingestion' ? 'text-emerald-400 border-b-2 border-emerald-400' : 'text-slate-400 hover:text-slate-200'}`}>
            <CloudUpload className="w-4 h-4" /> Data Ingestion
          </button>
          <button onClick={() => setActiveTab('portal')} className={`flex items-center gap-2 py-4 px-2 font-medium transition-colors ${activeTab === 'portal' ? 'text-emerald-400 border-b-2 border-emerald-400' : 'text-slate-400 hover:text-slate-200'}`}>
            <BrainCircuit className="w-4 h-4" /> Agent Portal
          </button>
          <button onClick={() => setActiveTab('owner')} className={`flex items-center gap-2 py-4 px-2 font-medium transition-colors ${activeTab === 'owner' ? 'text-emerald-400 border-b-2 border-emerald-400' : 'text-slate-400 hover:text-slate-200'}`}>
            <BarChart3 className="w-4 h-4" /> Owner Insights
          </button>
          <button onClick={() => setActiveTab('advisor')} className={`flex items-center gap-2 py-4 px-2 font-medium transition-colors ${activeTab === 'advisor' ? 'text-emerald-400 border-b-2 border-emerald-400' : 'text-slate-400 hover:text-slate-200'}`}>
            <Zap className="w-4 h-4" /> Executive Advisor
          </button>
        </div>
      </div>

      <main className="flex-1 max-w-7xl w-full mx-auto p-6 mt-4">

        {/* === TAB 1: DATA INGESTION === */}
        {activeTab === 'ingestion' && (
          <div className="flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500 w-full mb-12">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Platform Import Block */}
              <div className="bg-[#1e293b] rounded-2xl p-8 border border-slate-700 shadow-xl">
                <h2 className="text-xl font-bold mb-6 flex items-center gap-2 text-slate-100">
                  <CloudUpload className="text-emerald-400" /> Platform Import
                </h2>
                <form onSubmit={handleUpload} className="space-y-6">
                  <div className="border-2 border-dashed border-slate-600 hover:border-emerald-500 transition-colors rounded-xl p-8 text-center bg-slate-900/50 group relative">
                    <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
                    <Upload className="w-10 h-10 text-slate-500 group-hover:text-emerald-400 mx-auto mb-3 transition-colors" />
                    <p className="text-slate-300 font-medium">Drag & Drop or Click to Upload</p>
                    <p className="text-sm text-slate-500 mt-1">{file ? file.name : 'Must be a .csv containing "Review" and "Rating" fields'}</p>
                  </div>

                  <button
                    type="submit"
                    disabled={uploadLoading || !file}
                    className="w-full py-3 bg-gradient-to-r from-emerald-500 to-cyan-600 hover:from-emerald-400 hover:to-cyan-500 disabled:opacity-50 text-white rounded-xl font-semibold shadow-lg transition-all flex justify-center items-center gap-2"
                  >
                    {uploadLoading ? <><Zap className="w-5 h-5 animate-pulse" /> Processing Vector Embeddings...</> : 'Initialize Memory Bank'}
                  </button>
                </form>

                {uploadStatus && uploadStatus !== 'success' && (
                  <div className="mt-6 p-4 bg-slate-800/80 rounded-lg border border-slate-700 text-sm flex gap-3 text-emerald-400 items-center">
                    <div className="w-5 h-5 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin flex-shrink-0"></div>
                    {uploadStatus}
                  </div>
                )}
              </div>

              {/* Memory Bank Status Block */}
              <div className="bg-[#1e293b] rounded-2xl p-8 border border-slate-700 shadow-xl flex flex-col items-center justify-center text-center">
                 {!stats ? (
                    <div className="text-slate-500">
                       <Database className="w-12 h-12 mb-4 opacity-30 mx-auto" />
                       <p className="text-lg font-medium text-slate-400">Awaiting Memory Initialization</p>
                       <p className="text-sm max-w-xs">The RAG pipeline requires a vectorized dataset to perform intelligence extraction.</p>
                    </div>
                 ) : (
                    <div className="space-y-6">
                       <div className="w-20 h-20 rounded-2xl bg-gradient-to-tr from-emerald-500/20 to-cyan-500/20 flex items-center justify-center mx-auto border border-emerald-500/30 rotate-12">
                          <Brain className="w-10 h-10 text-emerald-400" />
                       </div>
                       <div>
                          <h3 className="text-3xl font-black text-slate-100 tabular-nums">{animatedCount}</h3>
                          <p className="text-xs font-bold text-emerald-500 uppercase tracking-widest mt-1">Linguistic Vectors Synced</p>
                       </div>
                       <div className="p-4 bg-slate-900/50 rounded-xl border border-slate-700/50 text-sm text-slate-400 text-left">
                          <div className="flex items-center gap-2 mb-2">
                             <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                             Semantic index reconstructed
                          </div>
                           <div className="flex items-center gap-2">
                              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                              GraphRAG Ontology Active
                           </div>
                       </div>
                    </div>
                 )}
              </div>
            </div>

             {/* Analysis Row: Sentiment + Emotion Pie Charts */}
             {stats && (
                <div className="flex flex-col gap-8 animate-in fade-in duration-700">
                   <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      {/* Coarse Sentiment Map */}
                      <div className="bg-[#1e293b] rounded-2xl p-8 border border-slate-700 shadow-xl transition-all hover:border-slate-600">
                         <div className="flex items-center justify-between mb-6">
                            <h2 className="text-xl font-bold flex items-center gap-2 text-slate-100">
                               <TrendingUp className="text-emerald-400" /> Coarse Sentiment Map
                            </h2>
                            <div className="px-3 py-1 bg-slate-800 rounded-full text-[10px] font-bold text-slate-400 border border-slate-700 uppercase tracking-wider">
                               VADER-Powered
                            </div>
                         </div>
                         <div className="h-72">
                            <ResponsiveContainer width="100%" height="100%">
                               <PieChart>
                                  <Pie data={stats.chartData} innerRadius={70} outerRadius={95} paddingAngle={8} dataKey="value">
                                     {stats.chartData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                     ))}
                                  </Pie>
                                  <Tooltip
                                     contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '12px', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.5)' }}
                                     itemStyle={{ color: '#fff', fontSize: '12px' }}
                                  />
                                  <Legend iconType="circle" />
                               </PieChart>
                            </ResponsiveContainer>
                         </div>
                      </div>

                      {/* Emotion Intelligence Map */}
                      <div className="bg-[#1e293b] rounded-2xl p-8 border border-slate-700 shadow-xl transition-all hover:border-slate-600">
                         <div className="flex items-center justify-between mb-6">
                            <h2 className="text-xl font-bold flex items-center gap-2 text-slate-100">
                               <Smile className="text-pink-400" /> Emotion Intelligence
                            </h2>
                            <div className="px-3 py-1 bg-slate-800 rounded-full text-[10px] font-bold text-slate-400 border border-slate-700 uppercase tracking-wider">
                               Heuristic Engine
                            </div>
                         </div>
                         <div className="h-72">
                            <ResponsiveContainer width="100%" height="100%">
                               <PieChart>
                                  <Pie data={emotionDistribution} innerRadius={70} outerRadius={95} paddingAngle={4} dataKey="value">
                                     {emotionDistribution.map((entry, index) => (
                                        <Cell key={`cell-emo-${index}`} fill={EMOTION_COLORS[entry.name.toLowerCase()] || '#94a3b8'} />
                                     ))}
                                  </Pie>
                                  <Tooltip
                                     contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '12px', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.5)' }}
                                     itemStyle={{ color: '#fff', fontSize: '12px' }}
                                  />
                                  <Legend iconType="circle" />
                               </PieChart>
                            </ResponsiveContainer>
                         </div>
                      </div>
                   </div>
                 </div>
              )}

              {/* Longitudinal Emotion Trajectory Row */}
              {emotionTrajectory.length > 0 && (
                <div className="bg-[#1e293b] rounded-2xl p-8 border border-slate-700 shadow-xl w-full animate-in fade-in duration-700 mt-8 mb-8">
                   <div className="flex items-center justify-between mb-6">
                      <h2 className="text-xl font-bold flex items-center gap-2 text-slate-100">
                         <TrendingUp className="text-pink-400" /> Longitudinal Emotion Trajectory
                      </h2>
                      <div className="px-3 py-1 bg-slate-800 rounded-full text-[10px] font-bold text-slate-400 border border-slate-700 uppercase tracking-wider">
                         Temporal Analytics
                      </div>
                   </div>
                   <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                         <AreaChart data={emotionTrajectory} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                            <defs>
                              <linearGradient id="colorJoy" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={EMOTION_COLORS.joy} stopOpacity={0.4}/>
                                <stop offset="95%" stopColor={EMOTION_COLORS.joy} stopOpacity={0}/>
                              </linearGradient>
                              <linearGradient id="colorSadness" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={EMOTION_COLORS.sadness} stopOpacity={0.4}/>
                                <stop offset="95%" stopColor={EMOTION_COLORS.sadness} stopOpacity={0}/>
                              </linearGradient>
                              <linearGradient id="colorAnger" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={EMOTION_COLORS.anger} stopOpacity={0.4}/>
                                <stop offset="95%" stopColor={EMOTION_COLORS.anger} stopOpacity={0}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                            <XAxis dataKey="date" stroke="#94a3b8" tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                            <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                            <Tooltip
                               contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '12px', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.5)' }}
                               itemStyle={{ fontSize: '14px' }}
                               labelStyle={{ color: '#94a3b8', marginBottom: '8px' }}
                            />
                            <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px' }} />
                            <Area type="monotone" dataKey="joy" stroke={EMOTION_COLORS.joy} strokeWidth={3} fillOpacity={1} fill="url(#colorJoy)" />
                            <Area type="monotone" dataKey="sadness" stroke={EMOTION_COLORS.sadness} strokeWidth={3} fillOpacity={1} fill="url(#colorSadness)" />
                            <Area type="monotone" dataKey="anger" stroke={EMOTION_COLORS.anger} strokeWidth={3} fillOpacity={1} fill="url(#colorAnger)" />
                            <Area type="monotone" dataKey="fear" stroke={EMOTION_COLORS.fear} strokeWidth={2} fillOpacity={0} fill="none" />
                            <Area type="monotone" dataKey="disgust" stroke={EMOTION_COLORS.disgust} strokeWidth={2} fillOpacity={0} fill="none" />
                            <Area type="monotone" dataKey="neutral" stroke={EMOTION_COLORS.neutral} strokeWidth={2} fillOpacity={0} fill="none" strokeDasharray="5 5" />
                         </AreaChart>
                      </ResponsiveContainer>
                   </div>
                </div>
              )}

            {/* Knowledge Graph Row */}
            <div className="bg-[#1e293b] rounded-2xl p-8 border border-slate-700 shadow-xl h-[640px] flex flex-col w-full relative">
               <div className="flex justify-between items-end mb-4">
                 <div>
                   <h2 className="text-xl font-bold flex items-center gap-2 text-slate-100">
                     <Network className="text-emerald-400" /> GraphRAG Co-Occurrence Ontology
                   </h2>
                   <p className="text-slate-400 text-sm mt-1">Force-directed map extracting noun-adjective frequencies via spaCy. Hover a node to trace its connections.</p>
                 </div>
                 <div className="flex items-center gap-4">
                    <div className="hidden sm:flex items-center gap-4 text-xs text-slate-400">
                       <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-emerald-400 inline-block" /> Noun</span>
                       <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block" /> Adjective</span>
                    </div>
                    <button onClick={fetchGraphData} className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm rounded-lg border border-slate-700 transition-colors whitespace-nowrap">
                      Refresh Map
                    </button>
                 </div>
               </div>

               <div ref={graphContainerRef} className="flex-1 rounded-xl border border-slate-800 bg-[#0f172a] relative overflow-hidden mt-2">
                  {graphLoading && (
                     <div className="absolute inset-0 flex items-center justify-center bg-[#1e293b]/80 z-20">
                        <div className="w-10 h-10 border-4 border-slate-800 border-t-emerald-500 rounded-full animate-spin"></div>
                     </div>
                  )}

                  {!graphLoading && graphData.nodes.length > 0 ? (
                      <ForceGraph2D
                         ref={forceGraphRef}
                         graphData={graphData}
                         width={graphDims.width}
                         height={graphDims.height}
                         backgroundColor="#0f172a"
                         nodeRelSize={4}
                         nodeVal={node => (node.val || 1)}
                         cooldownTime={4000}
                         onEngineStop={() => forceGraphRef.current?.zoomToFit(500, 60)}
                         onNodeHover={(node) => setHoverNode(node)}
                         nodeLabel="id"
                         linkWidth={link => {
                            const w = Math.max(0.6, Math.min(5, (link.weight || 1) * 0.6));
                            if (!hoverNode) return w;
                            const a = typeof link.source === 'object' ? link.source.id : link.source;
                            const b = typeof link.target === 'object' ? link.target.id : link.target;
                            return (a === hoverNode.id || b === hoverNode.id) ? w + 1.5 : w * 0.4;
                         }}
                         linkColor={link => {
                            if (!hoverNode) return 'rgba(52, 211, 153, 0.35)';
                            const a = typeof link.source === 'object' ? link.source.id : link.source;
                            const b = typeof link.target === 'object' ? link.target.id : link.target;
                            return (a === hoverNode.id || b === hoverNode.id) ? 'rgba(103, 232, 249, 0.85)' : 'rgba(100, 116, 139, 0.12)';
                         }}
                         linkDirectionalParticles={link => {
                            if (!hoverNode) return 0;
                            const a = typeof link.source === 'object' ? link.source.id : link.source;
                            const b = typeof link.target === 'object' ? link.target.id : link.target;
                            return (a === hoverNode.id || b === hoverNode.id) ? 3 : 0;
                         }}
                         linkDirectionalParticleWidth={2.4}
                         linkDirectionalParticleColor={() => '#67e8f9'}
                         linkDirectionalParticleSpeed={0.006}
                         nodeCanvasObject={(node, ctx, globalScale) => {
                            const isNoun = node.group === 'noun';
                            const baseColor = isNoun ? '#34d399' : '#22d3ee';
                            const dimmed = hoverNode && node.id !== hoverNode.id && !neighborMap.get(hoverNode.id)?.has(node.id);
                            const radius = Math.max(2.2, Math.sqrt(node.val || 1) * 1.8);

                            ctx.globalAlpha = dimmed ? 0.25 : 1;
                            ctx.beginPath();
                            ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
                            ctx.fillStyle = baseColor;
                            ctx.fill();
                            if (hoverNode && node.id === hoverNode.id) {
                               ctx.lineWidth = 2 / globalScale;
                               ctx.strokeStyle = '#f8fafc';
                               ctx.stroke();
                            }

                            const fontSize = Math.max(3.2, 11 / globalScale);
                            ctx.font = `${node.id === hoverNode?.id ? 'bold ' : ''}${fontSize}px system-ui, sans-serif`;
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'middle';
                            ctx.fillStyle = dimmed ? 'rgba(226, 232, 240, 0.25)' : '#e2e8f0';
                            ctx.fillText(node.id, node.x + radius + 2, node.y);
                         }}
                      />
                  ) : (
                     <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
                        <Network className="w-12 h-12 mb-3 opacity-30" />
                        <p>No knowledge graph data extracted yet.</p>
                     </div>
                  )}
                </div>
             </div>
          </div>
        )}

        {/* === TAB 2: PORTAL === */}
        {activeTab === 'portal' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
             <div className="lg:col-span-4 bg-[#1e293b] rounded-2xl p-6 border border-slate-700 shadow-xl relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-400 to-cyan-500" />
                <h2 className="text-xl font-bold mb-6 flex items-center gap-2"><Zap className="w-5 h-5 text-emerald-400" /> Multi-Agent Intake</h2>

                <form onSubmit={handlePortalSubmit} className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-slate-400 mb-3">Priority / Rating</label>
                    <div className="flex gap-2">
                        {[1, 2, 3, 4, 5].map((star) => (
                        <button key={star} type="button" onClick={() => setRating(star)} className={`w-10 h-10 rounded-full flex items-center justify-center transition-all ${rating >= star ? 'bg-amber-500/20 text-amber-400 border border-amber-500/50 shadow-[0_0_15px_rgba(251,191,36,0.2)]' : 'bg-slate-800 text-slate-600 border border-slate-700'}`}>
                          ★
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="flex justify-between items-end text-sm font-medium text-slate-400 mb-2">
                      Review Transcript
                      {prediction && <span className="text-[10px] uppercase tracking-wider text-emerald-500 bg-emerald-900/40 px-2 py-0.5 rounded">Tab to Auto-Complete</span>}
                    </label>

                    <div className="relative group text-base font-sans">
                      <div className="absolute inset-0 p-4 text-base font-sans leading-normal whitespace-pre-wrap break-words bg-slate-900 border border-slate-700 rounded-xl pointer-events-none overflow-hidden">
                        <span className="opacity-0 flex-shrink-0">{review}</span>
                        <span className="text-emerald-500/60 font-medium">{prediction}</span>
                      </div>
                      <textarea
                        value={review}
                        onChange={handleReviewChange}
                        onKeyDown={handleKeyDown}
                        placeholder="Type to trigger local ML auto-complete..."
                        className="relative z-10 w-full h-48 bg-transparent border border-transparent rounded-xl p-4 text-slate-100 font-sans text-base leading-normal focus:outline-none focus:ring-2 focus:ring-emerald-500/50 transition-all resize-none shadow-inner m-0 block"
                        required
                      />
                    </div>
                  </div>

                  <button type="submit" disabled={portalLoading} className="w-full py-3 bg-slate-800 border border-slate-700 hover:border-emerald-500 hover:text-emerald-400 text-slate-300 rounded-xl font-medium transition-all flex items-center justify-center gap-2 disabled:opacity-50">
                    {portalLoading ? 'Orchestrating Agents...' : <><BrainCircuit className="w-5 h-5"/> Analyze Context</>}
                  </button>
                </form>

                {portalHistory.length > 0 && (
                  <div className="mt-6 pt-5 border-t border-slate-800">
                     <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-1.5"><Clock className="w-3.5 h-3.5" /> Session History</h3>
                     <div className="space-y-2 max-h-56 overflow-y-auto custom-scrollbar pr-1">
                        {portalHistory.map((item, i) => (
                           <button
                              key={i}
                              onClick={() => setPortalResult(item)}
                              className={`w-full text-left px-3 py-2.5 rounded-lg border transition-colors ${portalResult === item ? 'bg-emerald-500/10 border-emerald-500/40' : 'bg-slate-900/50 border-slate-800 hover:border-slate-700'}`}
                           >
                              <div className="flex items-center justify-between gap-2 mb-1">
                                 <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400">{item.category}</span>
                                 <span className={`text-[10px] font-bold ${item.urgency_score > 3.5 ? 'text-red-400' : 'text-amber-400'}`}>{item.urgency_score}/10</span>
                              </div>
                              <p className="text-xs text-slate-400 truncate">{item.review_text}</p>
                           </button>
                        ))}
                     </div>
                  </div>
                )}
             </div>

             <div className="lg:col-span-8 bg-[#1e293b] rounded-2xl border border-slate-700 shadow-xl overflow-hidden flex flex-col">
                {!portalResult && !portalLoading && (
                  <div className="m-auto text-center text-slate-500 flex flex-col items-center">
                    <Search className="w-12 h-12 mb-4 opacity-30" />
                    Input review to retrieve historical vectors and forge PRD plans.
                  </div>
                )}

                {portalLoading && (
                   <div className="m-auto flex flex-col items-center">
                     <div className="w-16 h-16 border-4 border-slate-800 border-t-emerald-500 rounded-full animate-spin mb-4" />
                     <p className="text-emerald-400 animate-pulse tracking-widest font-mono text-sm uppercase">Querying Chroma DB...</p>
                   </div>
                )}

                {portalResult && !portalLoading && portalResult.error && (
                  <div className="m-auto text-center text-red-400 flex flex-col items-center px-8">
                    <AlertTriangle className="w-12 h-12 mb-4 opacity-60" />
                    <p className="font-semibold mb-1">Analysis Failed</p>
                    <p className="text-sm text-slate-400">{portalResult.error}</p>
                  </div>
                )}

                {portalResult && !portalLoading && !portalResult.error && (
                  <div className="p-8 pb-4 h-full flex flex-col gap-6 overflow-y-auto">
                    <div className="flex flex-wrap gap-4 mb-2">
                       <div className="bg-slate-900 border border-slate-800 px-4 py-2 rounded-lg"><span className="text-xs text-slate-500 uppercase block">Category</span> <span className="text-cyan-400 font-semibold">{portalResult.category}</span></div>
                       <div className="bg-slate-900 border border-slate-800 px-4 py-2 rounded-lg"><span className="text-xs text-slate-500 uppercase block">Urgency</span> <span className={portalResult.urgency_score > 3.5 ? 'text-red-400 font-bold' : 'text-amber-400 font-semibold'}>{portalResult.urgency_score}/10</span></div>
                       <div className="bg-slate-900 border border-slate-800 px-4 py-2 rounded-lg">
                          <span className="text-xs text-slate-500 uppercase block">Sentiment</span>
                          <span className={`${portalResult.sentiment === 'Positive' ? 'text-emerald-400' : portalResult.sentiment === 'Negative' ? 'text-red-400' : 'text-amber-400'} font-bold`}>
                             {portalResult.sentiment}
                          </span>
                       </div>
                    </div>

                    <div className="bg-slate-900 rounded-xl border border-slate-800 p-5">
                       <h3 className="text-sm font-semibold text-slate-300 mb-2 uppercase tracking-wide flex items-center gap-2"><Search className="w-4 h-4 text-emerald-500"/> Synthesized Root Cause</h3>
                       <p className="text-slate-400 text-sm leading-relaxed">{portalResult.root_cause}</p>
                    </div>

                    <div className="bg-slate-800/50 rounded-xl border border-emerald-900/30 p-5 shadow-inner">
                       <div className="flex items-center justify-between mb-3">
                          <h3 className="text-sm font-semibold text-emerald-400 uppercase tracking-wide">Forge Output Directive</h3>
                          <button
                             onClick={handleCopyOutput}
                             className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 hover:text-emerald-400 bg-slate-900/60 hover:bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 transition-colors"
                          >
                             {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                             {copied ? 'Copied' : 'Copy'}
                          </button>
                       </div>
                       <pre className="text-sm text-slate-300 whitespace-pre-wrap font-sans">{portalResult.final_output}</pre>
                    </div>
                  </div>
                )}
             </div>
          </div>
        )}

        {/* === TAB 3: OWNER === */}
        {activeTab === 'owner' && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
             <div className="flex justify-between items-end mb-6">
               <div>
                  <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-3"><AlertTriangle className="text-amber-500 w-7 h-7" /> Aggregate Priority Issues</h2>
                  <p className="text-slate-400 text-sm mt-1">Macro-analysis of the worst historical vectors across your active dataset.</p>
               </div>
               {!activeChatInsight && (
                 <button onClick={fetchOwnerInsights} className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm rounded-lg border border-slate-700 transition-colors">
                    Refresh Insights
                 </button>
               )}
             </div>

             {ownerLoading ? (
               <div className="bg-[#1e293b] rounded-2xl border border-slate-700 p-24 text-center">
                  <div className="w-12 h-12 border-4 border-slate-700 border-t-amber-500 rounded-full animate-spin mx-auto mb-4" />
                  <p className="text-amber-500 font-medium">Executive Agents synthesizing systemic failures...</p>
               </div>
             ) : ownerInsights && ownerInsights.length > 0 ? (
               <div className="grid grid-cols-1 gap-6">
                 {ownerInsights.map((insight, idx) => (
                    <div key={idx} className="bg-[#1e293b] border-l-4 border-amber-500 rounded-xl shadow-xl overflow-hidden flex flex-col md:flex-row shadow-[0_4px_30px_rgba(0,0,0,0.1)] hover:shadow-[0_4px_30px_rgba(245,158,11,0.05)] transition-shadow">
                       <div className="p-6 md:w-1/3 bg-slate-900/30 border-r border-slate-800 relative flex flex-col justify-between">
                         <div>
                            <div className="absolute top-4 left-4 text-6xl text-slate-800 font-black opacity-30 z-0">0{idx+1}</div>
                            <h3 className="text-lg font-bold text-slate-100 relative z-10 leading-snug pt-2">{insight.question || insight.Question}</h3>
                         </div>
                         {(insight.urgency_score || insight.Urgency_score || insight.Urgency) && (
                            <div className="mt-6 relative z-10 flex flex-col gap-4 border-t border-slate-800 pt-4">
                               <div className="flex items-center justify-between">
                                 <span className="text-xs uppercase tracking-wider text-slate-500 font-semibold gap-2 flex items-center"><AlertTriangle className="w-3 h-3"/> Urgency Index</span>
                                 <span className={`px-2 py-1 rounded text-sm font-bold ${parseFloat(insight.urgency_score || insight.Urgency) > 8.0 ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'}`}>
                                   {insight.urgency_score || insight.Urgency_score || insight.Urgency} / 10
                                 </span>
                               </div>
                               <button onClick={() => handleOpenChat(insight)} className="w-full py-2 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:border-emerald-500/50 rounded-lg text-sm font-semibold transition-colors flex justify-center items-center gap-2">
                                  <MessageSquare className="w-4 h-4" /> Discuss with Gemma
                               </button>
                            </div>
                         )}
                       </div>
                       <div className="p-6 md:w-2/3 flex flex-col gap-4">
                         <div>
                            <h4 className="text-xs uppercase font-bold text-slate-500 mb-1 border-b border-slate-800 pb-1">Root Cause Explanation</h4>
                            <p className="text-sm text-slate-400">{insight.explanation || insight.Explanation}</p>
                         </div>
                         <div className="bg-emerald-950/20 p-4 rounded-lg border border-emerald-900/30">
                            <h4 className="text-xs uppercase font-bold text-emerald-500 mb-3 flex items-center gap-2"><CheckCircle className="w-3 h-3"/> Detailed Action Plan</h4>
                            {Array.isArray(insight.solution || insight.Solution) ? (
                              <ul className="space-y-3">
                                {(insight.solution || insight.Solution).map((stepObj, i) => (
                                  <li key={i} className="text-sm text-slate-300 relative pl-4">
                                    <div className="absolute left-0 top-1.5 w-1.5 h-1.5 rounded-full bg-emerald-500"></div>
                                    <span className="font-semibold text-emerald-400">{stepObj.step || `Step ${i+1}`}:</span> {stepObj.description || stepObj}
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">{insight.solution || insight.Solution}</p>
                            )}
                         </div>
                       </div>
                    </div>
                  ))}
               </div>
             ) : ownerError ? (
                <div className="bg-[#1e293b] rounded-2xl border border-red-500/30 p-12 text-center bg-red-500/5">
                   <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4 opacity-50" />
                   <h3 className="text-xl font-bold text-slate-100 mb-2">Service Temporarily Unavailable</h3>
                   <p className="text-slate-400 mb-6 max-w-md mx-auto">{ownerError}</p>
                   <button onClick={fetchOwnerInsights} className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 transition-all font-bold">
                      Retry Analysis
                   </button>
                </div>
             ) : (
                <div className="bg-[#1e293b] rounded-2xl border border-slate-700 p-12 text-center text-slate-400">
                  No critical aggregated insights found. Either the dataset is small or ratings are strictly positive.
                </div>
             )}
          </div>
        )}

        {/* === TAB 4: EXECUTIVE ADVISOR (SLM) === */}
        {activeTab === 'advisor' && (
          <div className="max-w-6xl mx-auto animate-in fade-in slide-in-from-bottom-6 duration-700 flex flex-col h-[750px] relative">
             {/* Decorative Background Glows */}
             <div className="absolute top-1/4 -left-20 w-64 h-64 bg-emerald-500/10 rounded-full blur-[100px] pointer-events-none" />
             <div className="absolute bottom-1/4 -right-20 w-64 h-64 bg-cyan-500/10 rounded-full blur-[100px] pointer-events-none" />

             <div className="bg-[#1e293b]/80 backdrop-blur-xl rounded-3xl border border-slate-700/50 shadow-2xl overflow-hidden flex flex-col h-full relative z-10 transition-all duration-500">
                <div className="border-b border-slate-700/50 bg-[#1e293b]/90 p-6 flex items-center justify-between backdrop-blur-md">
                   <div className="flex items-center gap-5">
                      <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
                        <Zap className="w-7 h-7 text-white" />
                      </div>
                      <div>
                         <h3 className="font-extrabold text-slate-100 text-xl tracking-tight flex items-center gap-2">Executive Strategy Copilot</h3>
                         <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                            <p className="text-[10px] text-emerald-400/80 uppercase tracking-[0.2em] font-black">Gemma 2 SLM • High Fidelity Mode</p>
                         </div>
                      </div>
                   </div>
                   <div className="flex gap-3">
                      <button
                         onClick={fetchSuggestedQuestions}
                         className="p-2.5 bg-slate-800/50 hover:bg-slate-700/50 text-slate-400 hover:text-slate-200 rounded-xl border border-slate-700/50 transition-all shadow-inner"
                         title="Refresh Suggested Questions"
                      >
                         <BrainCircuit className="w-5 h-5" />
                      </button>
                      {activeChatInsight && (
                        <button
                          onClick={() => {
                            setActiveChatInsight(null);
                            setChatMessages([{ role: 'assistant', content: "Context cleared. I am ready for general strategic inquiries. How can we disrupt the status quo today?" }]);
                          }}
                          className="px-5 py-2.5 bg-slate-800/80 hover:bg-slate-700 text-slate-100 text-sm rounded-xl border border-slate-600/50 transition-all font-bold shadow-lg"
                        >
                           Reset Context
                        </button>
                      )}
                   </div>
                </div>

                {activeChatInsight && (
                  <div className="bg-gradient-to-r from-emerald-500/10 via-emerald-500/5 to-transparent border-b border-emerald-500/20 p-5 flex items-center gap-4">
                    <AlertTriangle className="w-5 h-5 text-emerald-400 animate-pulse" />
                    <div className="flex-1">
                      <span className="text-[10px] uppercase font-black text-emerald-500/70 tracking-widest block mb-0.5">Deep-Dive Context Layer Injected</span>
                      <p className="text-sm text-slate-100 font-semibold italic truncate">"{activeChatInsight.question || activeChatInsight.Question}"</p>
                    </div>
                  </div>
                )}

                <div className="flex-1 overflow-y-auto p-8 space-y-8 bg-[#0b101e]/60 scrollbar-hide" ref={chatScrollRef}>
                   {chatMessages.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto">
                        <div className="w-20 h-20 bg-slate-800/30 rounded-full flex items-center justify-center mb-8 border border-slate-700/50 animate-bounce duration-[3000ms]">
                           <MessageSquare className="w-10 h-10 text-slate-600" />
                        </div>
                        <h4 className="text-2xl font-black text-slate-100 mb-2 uppercase tracking-tight">Initiate Executive Inquiry</h4>
                        <p className="text-slate-400 text-center mb-12 text-lg">Select a RAG-analyzed theme from your historical dataset to begin a disruptive strategy forge.</p>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 w-full">
                           {suggestedLoading ? (
                             [1,2,3].map(i => <div key={i} className="h-40 bg-slate-800/40 animate-pulse rounded-2xl border border-slate-700/30" />)
                           ) : suggestedQuestions.map((q, i) => (
                             <button
                                key={i}
                                onClick={() => {
                                   setChatMessages([{ role: 'user', content: q }]);
                                   handleSendChat(null, q);
                                }}
                                className="group p-6 bg-slate-800/40 hover:bg-emerald-500/10 border border-slate-700/50 hover:border-emerald-500/50 rounded-2xl text-left transition-all hover:scale-[1.02] hover:shadow-xl hover:shadow-emerald-500/5"
                             >
                                <div className="p-2 bg-slate-800 group-hover:bg-emerald-500/20 rounded-lg w-fit mb-4 transition-colors">
                                   <Star className="w-4 h-4 text-slate-400 group-hover:text-emerald-400" />
                                </div>
                                <p className="text-sm font-bold text-slate-300 group-hover:text-white leading-relaxed">{q}</p>
                             </button>
                           ))}
                        </div>
                      </div>
                   ) : (
                     chatMessages.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
                           <div className={`max-w-[75%] rounded-[2rem] p-6 shadow-xl ${msg.role === 'user' ? 'bg-gradient-to-tr from-emerald-600 to-emerald-800 border-2 border-emerald-400/20 text-white rounded-tr-sm shadow-emerald-500/5' : 'bg-[#1e293b] border border-slate-700/50 text-slate-300 rounded-tl-sm whitespace-pre-wrap leading-relaxed shadow-black/20'}`}>
                               {msg.content}
                           </div>
                        </div>
                     ))
                   )}
                   {isChatLoading && (
                      <div className="flex justify-start">
                         <div className="bg-[#1e293b] border border-slate-700/50 rounded-[2rem] rounded-tl-sm p-6 flex items-center gap-4 shadow-xl">
                            <div className="flex gap-1.5">
                               <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce delay-0" />
                               <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce delay-150" />
                               <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce delay-300" />
                            </div>
                            <span className="text-emerald-400/80 text-xs font-black uppercase tracking-widest">Gemma 2 RAG Analysis in progress...</span>
                         </div>
                      </div>
                   )}
                </div>

                <form
                  onSubmit={handleSendChat}
                  className="p-6 bg-slate-900/90 border-t border-slate-700/50 backdrop-blur-md"
                >
                   <div className="relative max-w-4xl mx-auto">
                      <input
                         type="text"
                         value={chatInput}
                         onChange={(e) => setChatInput(e.target.value)}
                         disabled={isChatLoading}
                         placeholder={activeChatInsight ? "Ask about this systemic failure..." : "Forge a new disruptive strategy..."}
                         className="w-full bg-[#0b101e] border border-slate-700/50 rounded-2xl py-5 pl-7 pr-16 text-slate-100 text-lg focus:outline-none focus:ring-4 focus:ring-emerald-500/10 placeholder-slate-600 disabled:opacity-50 transition-all shadow-2xl"
                      />
                      <button
                        type="submit"
                        disabled={isChatLoading || !chatInput.trim()}
                        className="absolute right-3 top-3 w-12 h-12 bg-gradient-to-tr from-emerald-500 to-cyan-500 text-white hover:scale-105 active:scale-95 rounded-xl transition-all flex items-center justify-center shadow-lg shadow-emerald-500/20 disabled:opacity-30"
                      >
                         <MessageSquare className="w-6 h-6"/>
                      </button>
                   </div>
                </form>
             </div>
           </div>
        )}

      </main>

      {/* Toast Stack */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3 w-full max-w-sm px-4 sm:px-0">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`animate-toast-in flex items-start gap-3 p-4 rounded-xl border shadow-2xl backdrop-blur-md ${
              t.type === 'error'
                ? 'bg-red-950/90 border-red-500/30 text-red-200'
                : 'bg-[#1e293b]/95 border-emerald-500/30 text-slate-100'
            }`}
          >
            {t.type === 'error' ? (
              <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
            ) : (
              <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            )}
            <p className="text-sm flex-1 leading-snug">{t.message}</p>
            <button onClick={() => setToasts((ts) => ts.filter((x) => x.id !== t.id))} className="text-slate-500 hover:text-slate-300">
              <X className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
