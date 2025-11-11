export default function StatisticsCards({ stats }) {
  const getStat = (statName) => {
    const value = stats?.[statName];
    if (value === null || value === undefined || isNaN(value)) {
      return 0;
    }
    return Number(value);
  };

  const cards = [
    {
      label: "Total Words",
      value: getStat("total_words"),
      color: "primary",
    },
    {
      label: "Climate Related",
      value: getStat("climate_related"),
      color: "primary",
    },
    {
      label: "Positive",
      value: getStat("positive_words"),
      color: "blue",
    },
    {
      label: "Negative",
      value: getStat("negative_words"),
      color: "red",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      {cards.map((card, idx) => (
        <div
          key={idx}
          className="p-5 bg-gradient-to-br from-bluish-white to-bluish-gray rounded-xl"
        >
          <p
            className={`text-sm font-semibold tracking-wide text-${card.color} mb-1`}
          >
            {card.label}
          </p>
          <p
            className={`text-3xl font-bold text-${card.color} tracking-wide pl-6`}
          >
            {card.value.toLocaleString()}
          </p>
        </div>
      ))}
    </div>
  );
}
