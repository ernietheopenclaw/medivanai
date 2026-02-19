'use client';
import { useState, useEffect } from 'react';

export default function ModelStatus() {
  const [models, setModels] = useState<any[]>([]);
  const [show, setShow] = useState(false);

  useEffect(() => {
    fetch('/api/models').then(r => r.json()).then(setModels).catch(() => {});
  }, []);

  if (!models.length) return null;

  return (
    <div>
      <button onClick={() => setShow(!show)} className="text-xs text-gray-400 underline">
        {show ? 'Hide' : 'Show'} Model Status
      </button>
      {show && (
        <div className="mt-2 space-y-1">
          {models.map((m, i) => (
            <div key={i} className="flex items-center justify-between text-xs px-3 py-1.5 bg-gray-50 rounded">
              <span className="font-medium">{m.name}</span>
              <span className={m.status === 'loaded' ? 'text-risk-low' : m.status === 'mock' ? 'text-risk-moderate' : 'text-gray-400'}>
                {m.status}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

