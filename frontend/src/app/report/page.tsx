'use client';
import { Suspense, useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

function ReportContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionId = searchParams.get('id') || '';
  const [session, setSession] = useState<any>(null);

  useEffect(() => {
    if (sessionId) {
      fetch(`/api/session/${sessionId}`).then(r => r.json()).then(setSession).catch(() => {});
    }
  }, [sessionId]);

  if (!session) return <div className="min-h-screen flex items-center justify-center"><p>Loading...</p></div>;

  return (
    <div className="min-h-screen p-4 max-w-2xl mx-auto">
      <div className="no-print flex items-center justify-between mb-4">
        <button onClick={() => router.push(`/session/?id=${sessionId}`)} className="text-primary text-sm">← Back</button>
        <button onClick={() => window.print()} className="px-4 py-2 bg-gray-100 rounded-lg text-sm">🖨️ Print</button>
      </div>
      <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
        <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-gray-800">{session.report || 'No report generated yet.'}</pre>
      </div>
      <div className="no-print mt-6 text-center">
        <button onClick={() => router.push('/')} className="px-6 py-3 bg-primary text-white rounded-xl font-medium">New Session</button>
      </div>
    </div>
  );
}

export default function ReportPage() {
  return <Suspense fallback={<div className="min-h-screen flex items-center justify-center">Loading...</div>}><ReportContent /></Suspense>;
}
