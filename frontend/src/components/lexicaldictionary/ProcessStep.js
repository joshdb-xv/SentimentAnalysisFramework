export default function ProcessStep({ number, title, description, example }) {
  const renderExample = () => {
    if (example.type === "scores") {
      return (
        <div className="mt-2 p-3 bg-white rounded-lg border border-bluish-gray">
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <p className="text-gray-mid mb-1">Non-climate:</p>
              <p className="font-mono">
                Positive:{" "}
                <span className="text-blue font-semibold">
                  {example.nonClimate.positive}
                </span>
              </p>
              <p className="font-mono">
                Negative:{" "}
                <span className="text-red font-semibold">
                  {example.nonClimate.negative}
                </span>
              </p>
            </div>
            <div>
              <p className="text-gray-mid mb-1">Climate-related:</p>
              <p className="font-mono">
                Positive:{" "}
                <span className="text-blue font-semibold">
                  {example.climate.positive}
                </span>
              </p>
              <p className="font-mono">
                Negative:{" "}
                <span className="text-red font-semibold">
                  {example.climate.negative}
                </span>
              </p>
            </div>
          </div>
          <p className="text-xs text-primary font-semibold mt-2">
            {example.result}
          </p>
        </div>
      );
    }

    if (example.input && example.output) {
      return (
        <div className="mt-2 p-3 bg-white rounded-lg border border-bluish-gray font-mono text-xs">
          <span className="text-gray">{example.input}</span> →{" "}
          <span className="text-primary font-semibold">{example.output}</span>
        </div>
      );
    }

    if (example.definition) {
      return (
        <div className="mt-2 p-3 bg-white rounded-lg border border-bluish-gray">
          <p className="text-xs text-gray-mid mb-1">Definition:</p>
          <p className="font-mono text-xs">{example.definition}</p>
          <p className="text-xs text-primary font-semibold mt-2">
            {example.result}
          </p>
        </div>
      );
    }

    if (example.anchors) {
      return (
        <div className="mt-2 p-3 bg-white rounded-lg border border-bluish-gray">
          <div className="text-xs space-y-2">
            <p className="text-gray-mid">Negative anchors: {example.anchors}</p>
            <p className="font-mono">
              Avg similarity:{" "}
              <span className="font-semibold">{example.similarity}</span>
            </p>
            <p className="font-mono">
              Intensity multiplier:{" "}
              <span className="font-semibold text-primary">
                {example.multiplier}
              </span>
            </p>
          </div>
        </div>
      );
    }

    if (example.formula) {
      return (
        <div className="mt-2 p-3 bg-gradient-to-r from-primary/10 to-primary/5 rounded-lg border-2 border-primary">
          <p className="font-mono text-sm">
            <span className="text-gray-mid">{example.formula}</span>
            <span className="font-bold text-primary ml-2">
              {example.result}
            </span>
          </p>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="flex gap-4">
      <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-bold">
        {number}
      </div>
      <div className="flex-1">
        <p className="font-semibold text-primary-dark mb-1">{title}</p>
        <p className="text-gray-mid">{description}</p>
        {renderExample()}
      </div>
    </div>
  );
}
