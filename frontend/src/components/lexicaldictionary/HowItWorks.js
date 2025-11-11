import ProcessStep from "./ProcessStep";

export default function HowItWorks() {
  const steps = [
    {
      number: 1,
      title: "Word Cleaning",
      description:
        "The word is converted to lowercase and hyphens are removed for simpler form.",
      example: {
        input: '"Init"',
        output: '"init"',
      },
    },
    {
      number: 2,
      title: "Climate Keyword Detection",
      description:
        "The definition is checked against climate keywords (weather, temperature, climate, etc.) to determine if it's climate-related.",
      example: {
        definition: "Hot weather condition",
        result: '✓ Climate-related: Contains "weather"',
      },
    },
    {
      number: 3,
      title: "Base Sentiment Score",
      description:
        "Assign base score from sentiment label. Climate-related words get stronger scores.",
      example: {
        type: "scores",
        nonClimate: { positive: "+2.0", negative: "-2.0" },
        climate: { positive: "+3.0", negative: "-3.0" },
        result: 'For "init" (negative, climate): -3.0',
      },
    },
    {
      number: 4,
      title: "FastText Semantic Intensity",
      description:
        "Calculate semantic similarity to anchor words using FastText embeddings to adjust intensity (0.5x to 1.5x).",
      example: {
        anchors: "bad, terrible, crisis, disaster, destruction",
        similarity: "0.65",
        multiplier: "1.0 + (0.65 × 0.5) = 1.325",
      },
    },
    {
      number: 5,
      title: "Final Score Calculation",
      description:
        "Multiply base score by intensity multiplier for the final sentiment score.",
      example: {
        formula: "Base score × Intensity =",
        result: "-3.0 × 1.325 = -3.975",
      },
    },
  ];

  return (
    <div className="bg-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl">
      <div className="flex flex-row items-center gap-4 mb-6">
        <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
        <p className="font-medium text-xl text-[#1E293B]">HOW IT WORKS</p>
      </div>

      <div className="space-y-6">
        <p className="text-gray-mid">
          This tool processes Filipino/Cebuano dictionaries to generate
          VADER-compatible sentiment lexicons using FastText embeddings.
        </p>

        <div className="bg-bluish-white p-6 rounded-xl border-l-4 border-primary">
          <h3 className="font-semibold text-primary-dark mb-4">
            Sample Processing: Word "init"
          </h3>

          <div className="space-y-4 text-sm">
            {steps.map((step) => (
              <ProcessStep key={step.number} {...step} />
            ))}
          </div>
        </div>

        <div className="bg-blue/10 p-4 rounded-xl border border-blue">
          <p className="text-sm text-blue flex items-start gap-2">
            <span className="text-xl">💡</span>
            <span>
              <strong>Duplicate Handling:</strong> If multiple definitions exist
              for the same word, the system prioritizes climate-related
              definitions and keeps simpler word forms (without hyphens).
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}
