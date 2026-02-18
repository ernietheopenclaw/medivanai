'use client';
import { Suspense, useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import CameraCapture from '@/components/CameraCapture';
import ImageUpload from '@/components/ImageUpload';
import AnalysisCard from '@/components/AnalysisCard';

function SessionContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionId = searchParams.get('id') || '';
  const [findings, setFindings] = useState<any[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [mode, setMode] = useState<'menu' | 'camera' | 'upload'>('menu');

  useEffect(() => {
    if (sessionId) {
      fetch(`/api/session/${sessionId}`).then(r => r.json()).then(d => setFindings(d.findings || [])).catch(() => {});
    }
  }, [sessionId]);

  const handleImage = async (file: File) => {
    setAnalyzing(true);
    setMode('menu');
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`/api/session/${sessionId}/analyze`, { method: 'POST', body: form });
      const data = await res.json();
      setFindings(prev => [...prev, data]);
    } catch (e) {
      alert('Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const generateReport = async () => {
    try {
      await fetch(`/api/session/${sessionId}/report`, { method: 'POST' });
      router.push(`/report/?id=${sessionId}`);
    } catch (e) {
      alert('Report generation failed');
    }
  };

  return (
    <div className="min-h-screen p-4 max-w-lg mx-auto">
      <div className="flex items-center justify-between mb-6">
        <button onClick={() => router.push('/')} className="text-primary text-sm">← Home</button>
        <h1 className="text-lg font-bold text-primary">Session {sessionId}</h1>
        <div className="w-16" />
      </div>

      {mode === 'menu' && (
        <div className="grid grid-cols-2 gap-3 mb-6">
          <button onClick={() => setMode('camera')} className="py-4 bg-primary text-white rounded-xl font-medium text-sm min-h-[56px]">📷 Capture</button>
          <button onClick={() => setMode('upload')} className="py-4 bg-gray-100 text-gray-700 rounded-xl font-medium text-sm min-h-[56px]">📁 Upload</button>
        </div>
      )}

      {mode === 'camera' && (
        <div className="mb-6">
          <button onClick={() => setMode('menu')} className="text-sm text-gray-500 mb-2">← Back</button>
          <CameraCapture onCapture={handleImage} />
        </div>
      )}

      {mode === 'upload' && (
        <div className="mb-6">
          <button onClick={() => setMode('menu')} className="text-sm text-gray-500 mb-2">← Back</button>
          <ImageUpload onUpload={handleImage} />
        </div>
      )}

      {analyzing && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-4 text-center">
          <div className="animate-spin text-2xl mb-2">⚙️</div>
          <p className="text-primary font-medium">Analyzing image...</p>
        </div>
      )}

      <div className="space-y-3 mb-6">
        {findings.map((f, i) => <AnalysisCard key={i} finding={f} index={i + 1} />)}
      </div>

      {findings.length > 0 && (
        <button onClick={generateReport} className="w-full py-4 bg-green-600 text-white rounded-xl font-semibold text-lg min-h-[56px] shadow-lg">
          📋 Generate Holistic Report
        </button>
      )}
    </div>
  );
}

export default function SessionPage() {
  return <Suspense fallback={<div className="min-h-screen flex items-center justify-center">Loading...</div>}><SessionContent /></Suspense>;
}
