export default function AverageScoresPanel({ stats }) {
  const getStat = (statName) => {
    const value = stats?.[statName];
    if (value === null || value === undefined || isNaN(value)) {
      return 0;
    }
    return Number(value);
  };

  const scoreItems = [
    {
      label: "Climate Words",
      value: (getStat("avg_score_climate") || 0).toFixed(3),
    },
    {
      label: "General Words",
      value: (getStat("avg_score_general") || 0).toFixed(3),
    },
    {
      label: "Non-Climate",
      value: getStat("non_climate").toLocaleString(),
    },
    {
      label: "Neutral",
      value: getStat("neutral_words").toLocaleString(),
    },
  ];

  return (
    <div className="mb-6 p-5 bg-white/50 border border-bluish-gray rounded-xl">
      <h3 className="font-semibold text-black mb-3">
        Average Sentiment Scores
      </h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        {scoreItems.map((item, idx) => (
          <div key={idx} className="flex flex-col">
            <span className="text-gray-mid text-xs mb-1">{item.label}</span>
            <span className="font-mono text-lg font-bold text-black">
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
