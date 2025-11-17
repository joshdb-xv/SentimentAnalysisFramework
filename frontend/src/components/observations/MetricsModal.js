import { MdClose } from "react-icons/md";

// Helper function to format values (handles both decimal and percentage formats)
const formatValue = (value) => {
  if (value === null || value === undefined) return 0;
  // If value is less than 2, it's probably a decimal (e.g., 0.75)
  // If value is 2 or greater, it's definitely a percentage (e.g., 75.0)
  return value < 2 ? value * 100 : value;
};

// Helper function to extract metrics from benchmark data
const getMetrics = (benchmarks, model) => {
  if (!benchmarks) return null;

  if (model === "vader") {
    // Use dynamic data from vader_multiple_runs if available
    if (benchmarks?.vader_multiple_runs) {
      const stats = benchmarks.vader_multiple_runs.statistics;

      return {
        name: "VADER - Tweet Sentiment Identifier",
        accuracy: formatValue(stats.accuracy.mean),
        std: formatValue(stats.accuracy.std),
        precision: formatValue(stats.precision.mean),
        recall: formatValue(stats.recall.mean),
        f1: formatValue(stats.f1.mean),
        runs: benchmarks.vader_multiple_runs.number_of_runs,
        baseline: 72,
      };
    }

    // Fallback to single identifier value
    if (benchmarks?.vader_sentiment_identifier) {
      return {
        name: "VADER - Tweet Sentiment Identifier",
        accuracy: formatValue(benchmarks.vader_sentiment_identifier),
        precision: null,
        recall: null,
        f1: null,
        runs: null,
        baseline: 72,
      };
    }

    // No data available
    return null;
  }

  if (model === "climateChecker" && benchmarks?.climate_checker_multiple_runs) {
    const stats = benchmarks.climate_checker_multiple_runs.statistics;

    return {
      name: "Naive Bayes - Climate Related Checker",
      accuracy: formatValue(stats.accuracy.mean),
      std: formatValue(stats.accuracy.std),
      precision: formatValue(stats.precision.mean),
      recall: formatValue(stats.recall.mean),
      f1: formatValue(stats.f1.mean),
      runs: benchmarks.climate_checker_multiple_runs.number_of_runs,
      baseline: 81,
    };
  }

  if (model === "domainIdentifier" && benchmarks?.multiple_runs) {
    const stats = benchmarks.multiple_runs.statistics;

    return {
      name: "Naive Bayes - Climate Domain Identifier",
      accuracy: formatValue(stats.accuracy.mean),
      std: formatValue(stats.accuracy.std),
      precision: formatValue(stats.precision.mean),
      recall: formatValue(stats.recall.mean),
      f1: formatValue(stats.f1.mean),
      runs: benchmarks.multiple_runs.number_of_runs,
      baseline: 81,
    };
  }

  return null;
};

// Metric Card Component
const MetricCard = ({ metrics }) => {
  if (!metrics) return null;

  const difference = metrics.accuracy - metrics.baseline;
  const differenceText =
    difference >= 0
      ? `+${difference.toFixed(2)}%`
      : `${difference.toFixed(2)}%`;
  const differenceColor = difference >= 0 ? "text-green-600" : "text-red-600";

  return (
    <div className="bg-white border-2 border-gray-200 rounded-xl p-6 flex-1">
      {/* Model Name */}
      <h3 className="font-bold text-[#0A3D91] text-lg mb-4 pb-3 border-b-2 border-gray-100">
        {metrics.name}
      </h3>

      {/* Metrics */}
      <div className="space-y-3">
        {/* Accuracy */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600 font-medium">Accuracy:</span>
          <span className="text-base font-bold text-[#1E293B]">
            {metrics.accuracy.toFixed(2)}%
            {metrics.std && (
              <span className="text-sm text-gray-500 font-normal ml-1">
                ±{metrics.std.toFixed(2)}%
              </span>
            )}
          </span>
        </div>

        {/* Precision */}
        {metrics.precision !== null && (
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600 font-medium">
              Precision:
            </span>
            <span className="text-base font-bold text-[#1E293B]">
              {metrics.precision.toFixed(2)}%
            </span>
          </div>
        )}

        {/* Recall */}
        {metrics.recall !== null && (
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600 font-medium">Recall:</span>
            <span className="text-base font-bold text-[#1E293B]">
              {metrics.recall.toFixed(2)}%
            </span>
          </div>
        )}

        {/* F1-Score */}
        {metrics.f1 !== null && (
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600 font-medium">F1-Score:</span>
            <span className="text-base font-bold text-[#1E293B]">
              {metrics.f1.toFixed(2)}%
            </span>
          </div>
        )}

        {/* Baseline Comparison */}
        <div className="border-t-2 border-gray-100 pt-3 mt-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600 font-medium">
              vs Baseline ({metrics.baseline}%):
            </span>
            <span className={`text-base font-bold ${differenceColor}`}>
              {differenceText}
            </span>
          </div>
        </div>

        {/* Number of runs if available */}
        {metrics.runs && (
          <div className="text-xs text-gray-500 text-center mt-3 pt-3 border-t border-gray-100">
            Based on {metrics.runs} runs
          </div>
        )}
      </div>
    </div>
  );
};

// Main Modal Component
export default function MetricsModal({ isOpen, onClose, benchmarks }) {
  if (!isOpen) return null;

  const vaderMetrics = getMetrics(benchmarks, "vader");
  const climateCheckerMetrics = getMetrics(benchmarks, "climateChecker");
  const domainIdentifierMetrics = getMetrics(benchmarks, "domainIdentifier");

  return (
    <div
      className="bg-[#F8FAFC] rounded-2xl shadow-2xl max-w-6xl w-[90%] max-h-[85vh] overflow-y-auto mx-auto"
      onClick={(e) => e.stopPropagation()}
    >
      {/* Modal Header */}
      <div className="sticky top-0 bg-[#F8FAFC] border-b-2 border-gray-200 p-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
          <h2 className="font-bold text-2xl  bg-gradient-to-r from-[#222222] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent tracking-wide">
            FRAMEWORK ACCURACY - DETAILED METRICS
          </h2>
        </div>
        <button
          onClick={onClose}
          className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
        >
          <MdClose className="text-3xl text-gray-600" />
        </button>
      </div>

      {/* Modal Content */}
      <div className="p-6">
        <div className="flex flex-col lg:flex-row gap-6">
          <MetricCard metrics={vaderMetrics} />
          <MetricCard metrics={climateCheckerMetrics} />
          <MetricCard metrics={domainIdentifierMetrics} />
        </div>

        {/* Additional Info */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <p className="text-sm text-blue-800">
            <span className="font-semibold">Note:</span> These metrics represent
            the performance of each model in the sentiment analysis framework.
            Higher values indicate better performance. The baseline comparison
            shows how much each model exceeds or falls short of the expected
            minimum accuracy.
          </p>
        </div>
      </div>
    </div>
  );
}
