'use client';
import { useState, useEffect } from 'react';

export default function ConnectionStatus() {
  const [status, setStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [info, setInfo] = useState<any>(null);

  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(d => { setStatus('connected'); setInfo(d); })
      .catch(() => setStatus('disconnected'));
  }, []);

  const colors = { checking: 'text-gray-400', connected: 'text-risk-low', disconnected: 'text-risk-high' };
  const labels = { checking: '⏳ Connecting...', connected: '🟢 Connected', disconnected: '🔴 Disconnected' };

  return (
    <div className="text-center">
      <p className={`text-sm font-medium ${colors[status]}`}>{labels[status]}</p>
      {info && (
        <p className="text-xs text-gray-400 mt-1">
          {info.mock_mode ? 'Mock Mode' : 'Live'} • {info.gpu !== 'N/A' ? info.gpu : info.platform}
        </p>
      )}
    </div>
  );
}
