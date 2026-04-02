import React, { useState, useEffect, useRef } from 'react';
import { Crosshair, Power, Search, AlertTriangle, ShieldAlert, Cpu, Upload } from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  
  // Telemetry state
  const [stats, setStats] = useState({ uptime: 0, subjects: 0, anomalies: 0 });
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);
  
  const handleUpload = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    if (files.length > 4) {
      alert("Maximum 4 cameras supported for Grid View.");
      return;
    }
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
    }
    setUploading(true);
    try {
        await fetch(`${API_BASE}/upload_cameras`, { method: 'POST', body: formData });
        setIsRunning(true);
    } catch (err) {}
    setUploading(false);
    e.target.value = null;
  };

  useEffect(() => {
    // Stats Poller
    const statInterval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/stats`);
        const data = await res.json();
        setStats({
          uptime: data.uptime,
          subjects: data.subjects_known,
          anomalies: data.anomalies_detected
        });
        setIsRunning(data.is_running);
      } catch(e) {}
    }, 2000);

    return () => clearInterval(statInterval);
  }, []);

  useEffect(() => {
    let eventSource;
    if (isRunning) {
      eventSource = new EventSource(`${API_BASE}/api/live_alerts`);
      eventSource.onmessage = (event) => {
        const text = event.data;
        if (text && text !== ': keep-alive') {
          const newAlert = { id: Date.now(), text, time: new Date().toLocaleTimeString() };
          setAlerts(prev => [newAlert, ...prev].slice(0, 50));
        }
      };
    }
    return () => { if (eventSource) eventSource.close(); };
  }, [isRunning]);

  const toggleSystem = async () => {
    try {
      if (isRunning) {
        await fetch(`${API_BASE}/stop`, { method: 'POST' });
        setIsRunning(false);
      } else {
        await fetch(`${API_BASE}/start`, { method: 'POST' });
        setIsRunning(true);
      }
    } catch (err) {}
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    try {
      const res = await fetch(`${API_BASE}/search?query=${encodeURIComponent(searchQuery)}`);
      const data = await res.json();
      setSearchResults(data.results);
    } catch (err) {}
  };

  const formatUptime = (sec) => {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <>
      <div className="cyber-grid" />
      <div className="app-container">
        
        {/* Header FUI */}
        <header className="header glass-panel fui-bracket">
          <div className="brand">
            <Crosshair size={32} color="var(--accent-cyan)" />
            <div>
              <h1>CHRONOS.OS</h1>
              <div className="sys-text">SYS.VER: 1.4 // PROTOCOL: OMEGA // STATUS: {isRunning ? 'SECURE' : 'STANDBY'}</div>
            </div>
          </div>
          <div className="control-panel" style={{display: 'flex', gap: '12px'}}>
            <input 
              type="file" 
              multiple 
              accept="video/mp4" 
              style={{display: 'none'}} 
              ref={fileInputRef} 
              onChange={handleUpload}
            />
            <button 
              className="start-btn" 
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              style={{borderColor: 'var(--accent-green)', color: 'var(--accent-green)'}}
            >
              <Upload size={18} />
              {uploading ? 'DECRYPTING...' : 'UPLOAD ARCHIVES'}
            </button>
            <button 
              className={isRunning ? "stop-btn" : "start-btn"} 
              onClick={toggleSystem}
            >
              <Power size={18} />
              {isRunning ? 'OVERRIDE : TERMINATE' : 'INITIALIZE UPLINK'}
            </button>
          </div>
        </header>

        {/* Telemetry Bar */}
        <div className="telemetry-bar">
          <div className="stat-box fui-bracket glass-panel">
            <span className="stat-label">SESSION UPTIME</span>
            <span className="stat-value">{formatUptime(stats.uptime)}</span>
          </div>
          <div className="stat-box fui-bracket glass-panel">
            <span className="stat-label">KNOWN SUBJECTS (RE-ID DB)</span>
            <span className="stat-value">{stats.subjects} ENTITIES</span>
          </div>
          <div className="stat-box fui-bracket glass-panel" style={{ borderColor: stats.anomalies > 0 ? 'var(--danger)' : '' }}>
            <span className="stat-label">ANOMALIES LOGGED</span>
            <span className="stat-value" style={{ color: stats.anomalies > 0 ? 'var(--danger)' : '' }}>{stats.anomalies} SIGS</span>
          </div>
          <div className="stat-box fui-bracket glass-panel">
            <span className="stat-label">AI COGNITIVE LOAD</span>
            <span className="stat-value" style={{ color: 'var(--accent-green)'}}><Cpu size={16} /> NOMINAL</span>
          </div>
        </div>

        {/* Main Content */}
        <main className="main-content">
          {/* Left: Live Video Feed */}
          <section className="video-section fui-bracket">
            <div className="section-title">OPTICAL SURVEILLANCE FEED [SECTOR ALPHA]</div>
            <div className="video-feed">
              {isRunning && <div className="radar-sweep" />}
              {isRunning ? (
                <>
                  <div style={{position: 'absolute', top: 10, left: 10, zIndex: 20, color: 'var(--danger)', fontFamily: 'Orbitron', fontWeight: 900, textShadow: '0 0 5px red'}}>REC <div style={{display: 'inline-block', width: 10, height: 10, background: 'red', borderRadius: '50%', animation: 'glitch 1s infinite'}}></div></div>
                  <img 
                    src={`${API_BASE}/api/video_feed?time=${Date.now()}`} 
                    alt="Live Camera Feed" 
                    style={{ width: '100%', height: '100%', objectFit: 'contain', filter: 'contrast(1.2)' }}
                  />
                  <div style={{position: 'absolute', bottom: 10, left: 10, zIndex: 20, color: 'var(--accent-cyan)', fontFamily: 'Share Tech Mono', fontSize: '0.8rem'}}>MULTI-CAM COMPOSITE | FPS: 30 / MJPEG</div>
                </>
              ) : (
                <div style={{textAlign: 'center', color: 'var(--text-muted)', fontFamily: 'Orbitron'}}>
                  <ShieldAlert size={64} opacity={0.2} style={{margin: '0 auto'}} />
                  <p style={{marginTop: '20px'}}>SYSTEM OFFLINE. SECURE UPLINK REQUIRED.</p>
                </div>
              )}
            </div>
          </section>

          {/* Right: Anomaly Timeline & Search */}
          <section className="alerts-section glass-panel fui-bracket">
            <div className="section-title">THREAT LOGS & TERMINAL</div>
            
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px', paddingRight: '4px' }}>
              {alerts.length === 0 ? (
                <p style={{color: 'var(--text-muted)', fontFamily: 'Share Tech Mono', fontSize: '0.9rem', marginTop: '20px', textAlign: 'center'}}>Monitoring for irregular acoustic/visual signatures...</p>
              ) : (
                alerts.map((alert, i) => {
                  let alertType = 'SYS_WARNING';
                  let msgText = alert.text.toUpperCase();
                  if (msgText.includes('[SYSTEM]')) alertType = 'SYSTEM_LOG';
                  if (msgText.includes('[ACTION]')) alertType = 'ACTION_LOG';
                  
                  return (
                  <div key={alert.id} className="alert-card">
                    <div className="alert-time">[{alert.time}] {alertType}</div>
                    <div className="alert-msg">
                      <AlertTriangle size={14} style={{display: 'inline', marginRight: '6px', opacity: alertType === 'SYS_WARNING' ? 1 : 0.5}} />
                      {msgText.replace('[SYSTEM]', '').replace('[ACTION]', '').trim()}
                    </div>
                  </div>
                  );
                })
              )}
            </div>

            <div style={{ position: 'relative', marginTop: 'auto', paddingTop: '16px', borderTop: '1px solid var(--glass-border)' }}>
              <form onSubmit={handleSearch} style={{display: 'flex', gap: '8px', alignItems: 'center'}}>
                <span style={{color: 'var(--accent-green)', fontFamily: 'Orbitron'}}>&gt;</span>
                <input 
                  type="text" 
                  className="search-input" 
                  placeholder="EXECUTE VECTOR QUERY..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                />
              </form>

              {searchResults && (
                <div className="search-results fui-bracket">
                  <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '8px'}}>
                    <h3 style={{fontSize: '0.9rem', color: 'var(--accent-green)'}}>MATCHING RECORDS:</h3>
                    <span style={{cursor: 'pointer', color: 'var(--danger)'}} onClick={() => setSearchResults(null)}>[X] CLOSE</span>
                  </div>
                  {searchResults.documents?.[0]?.length > 0 ? (
                    searchResults.documents[0].map((doc, idx) => {
                      const meta = searchResults.metadatas[0][idx];
                      const ts = meta?.timestamp ? new Date(meta.timestamp * 1000).toLocaleString() : 'Unknown';
                      return (
                        <div key={idx} className="result-card">
                          <div style={{color: 'var(--accent-cyan)', fontSize: '0.8rem', marginBottom: '4px'}}>TIMESTAMP: {ts}</div>
                          <div style={{color: '#fff', fontSize: '0.9rem'}}>&gt; {doc}</div>
                        </div>
                      );
                    })
                  ) : (
                    <p style={{color: 'var(--danger)'}}>ERR: NO MATCHING SIGNATURES FOUND.</p>
                  )}
                </div>
              )}
            </div>
          </section>
        </main>
      </div>
    </>
  );
}

export default App;
