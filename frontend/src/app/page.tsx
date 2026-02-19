'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import ConnectionStatus from '@/components/ConnectionStatus';
import ModelStatus from '@/components/ModelStatus';

export default function Home() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  const startSession = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/session/start', { method: 'POST' });
      const data = await res.json();
      router.push(`/session/?id=${data.id}`);
    } catch (e) {
      alert('Failed to connect to MediVan server');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      <div className="max-w-md w-full text-center space-y-8">
        {/* Logo */}
        <div className="space-y-2">
          <div className="text-6xl">🏥</div>
          <h1 className="text-3xl font-bold text-primary">MediVan</h1>
          <p className="text-gray-500 text-lg">Mobile Diagnostic Hub</p>
        </div>

        {/* Status */}
        <ConnectionStatus />

        {/* Start Button */}
        <button
          onClick={startSession}
          disabled={loading}
          className="w-full py-4 px-6 bg-primary text-white text-lg font-semibold rounded-2xl
                     hover:bg-blue-700 active:bg-blue-800 transition-colors
                     disabled:opacity-50 disabled:cursor-not-allowed
                     min-h-[56px] shadow-lg"
        >
          {loading ? 'Starting...' : 'Start New Screening Session'}
        </button>

        {/* Model Status */}
        <ModelStatus />

        {/* Footer */}
        <p className="text-xs text-gray-400">
          All data processed locally on-device. No patient data leaves this system.
        </p>
      </div>
    </div>
  );
}

