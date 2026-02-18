'use client';
import { useRef, useState } from 'react';

export default function ImageUpload({ onUpload }: { onUpload: (file: File) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = (file: File) => {
    if (file.type.startsWith('image/')) onUpload(file);
    else alert('Please select an image file.');
  };

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={e => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); }}
      onClick={() => inputRef.current?.click()}
      className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors min-h-[160px] flex flex-col items-center justify-center
        ${dragOver ? 'border-primary bg-blue-50' : 'border-gray-300 hover:border-primary'}`}
    >
      <div className="text-4xl mb-3">📁</div>
      <p className="font-medium text-gray-700">Tap to select or drag image here</p>
      <p className="text-sm text-gray-400 mt-1">JPEG, PNG up to 10MB</p>
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={e => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }} />
    </div>
  );
}
