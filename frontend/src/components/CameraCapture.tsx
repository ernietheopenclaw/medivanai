'use client';
import { useRef, useState, useEffect } from 'react';

export default function CameraCapture({ onCapture }: { onCapture: (file: File) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: 1280, height: 960 } })
      .then(s => { setStream(s); if (videoRef.current) videoRef.current.srcObject = s; })
      .catch(() => setError('Camera access denied. Use HTTPS or Tailscale connection.'));
    return () => { stream?.getTracks().forEach(t => t.stop()); };
  }, []);

  const capture = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d')!.drawImage(videoRef.current, 0, 0);
    canvas.toBlob(blob => {
      if (blob) onCapture(new File([blob], 'capture.jpg', { type: 'image/jpeg' }));
    }, 'image/jpeg', 0.9);
  };

  if (error) return <div className="bg-red-50 p-4 rounded-xl text-red-600 text-sm">{error}</div>;

  return (
    <div className="space-y-3">
      <video ref={videoRef} autoPlay playsInline className="w-full rounded-xl bg-black" />
      <button onClick={capture} className="w-full py-4 bg-primary text-white rounded-xl font-semibold min-h-[56px]">
        📸 Capture Image
      </button>
    </div>
  );
}

