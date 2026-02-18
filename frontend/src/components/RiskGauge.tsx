'use client';

const RISK_CONFIG: Record<string, { emoji: string; label: string; color: string; bg: string }> = {
  low: { emoji: '🟢', label: 'Low Risk', color: 'text-risk-low', bg: 'bg-green-50' },
  moderate: { emoji: '🟡', label: 'Moderate Risk', color: 'text-risk-moderate', bg: 'bg-amber-50' },
  high: { emoji: '🔴', label: 'High Risk', color: 'text-risk-high', bg: 'bg-red-50' },
  urgent: { emoji: '🔴', label: 'URGENT', color: 'text-risk-urgent', bg: 'bg-red-100' },
};

export default function RiskGauge({ level }: { level: string }) {
  const cfg = RISK_CONFIG[level] || RISK_CONFIG.low;
  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg ${cfg.bg}`}>
      <span>{cfg.emoji}</span>
      <span className={`text-sm font-semibold ${cfg.color}`}>{cfg.label}</span>
    </div>
  );
}
