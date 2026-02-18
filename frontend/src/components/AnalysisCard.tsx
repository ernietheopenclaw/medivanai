'use client';
import RiskGauge from './RiskGauge';

const TYPE_LABELS: Record<string, string> = {
  skin_lesion: '🔬 Skin Lesion',
  chest_xray: '🫁 Chest X-ray',
  fundus: '👁️ Fundus',
  unknown: '❓ Unknown',
};

const TYPE_COLORS: Record<string, string> = {
  skin_lesion: 'bg-purple-100 text-purple-700',
  chest_xray: 'bg-blue-100 text-blue-700',
  fundus: 'bg-teal-100 text-teal-700',
  unknown: 'bg-gray-100 text-gray-500',
};

export default function AnalysisCard({ finding, index }: { finding: any; index: number }) {
  const type = finding.image_type || 'unknown';
  const conf = (finding.confidence || 0) * 100;

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${TYPE_COLORS[type] || TYPE_COLORS.unknown}`}>
          {TYPE_LABELS[type] || type}
        </span>
        <span className="text-xs text-gray-400">#{index}</span>
      </div>

      {/* Classification */}
      <h3 className="font-semibold text-lg capitalize mb-1">{finding.classification || 'N/A'}</h3>

      {/* Confidence Bar */}
      <div className="flex items-center gap-2 mb-3">
        <div className="flex-1 bg-gray-100 rounded-full h-2">
          <div className="bg-primary h-2 rounded-full transition-all" style={{ width: `${conf}%` }} />
        </div>
        <span className="text-sm font-medium text-gray-600">{conf.toFixed(1)}%</span>
      </div>

      {/* Risk */}
      <RiskGauge level={finding.risk_level || 'low'} />

      {/* Recommendation */}
      {finding.recommendation && (
        <p className="text-sm text-gray-600 mt-3 leading-relaxed">{finding.recommendation}</p>
      )}
    </div>
  );
}
