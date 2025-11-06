export default function FrameworkAccuracyCard({
  benchmarks,
  loading,
  error,
  onRetry,
  onViewDetails,
}) {
  return (
    <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-full h-full p-4 rounded-2xl">
      <div className="flex items-center gap-3">
        <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
        <p className="font-medium text-[#1E293B] tracking-wide text-lg">
          FRAMEWORK ACCURACY
        </p>
      </div>

      {/* ACCURACY SCORES */}
      <div className="flex flex-col items-center justify-center gap-1 mt-4">
        {/* VADER - Sentiment Identifier - STATIC */}
        <div className="flex flex-col justify-center items-center gap-2">
          <p className="text-3xl font-extrabold bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent py-1">
            81.58%
          </p>
          <p className="text-base text-[#1E293B] text-center font-semibold">
            VADER - Tweet Sentiment Identifier
          </p>
        </div>

        {/* Naive Bayes - Climate Related Checker - DYNAMIC */}
        <div className="flex flex-col justify-center items-center gap-2">
          {loading ? (
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#0A3D91]"></div>
          ) : error ? (
            <div className="flex flex-col items-center gap-1">
              <p className="text-xl text-red-500">Error</p>
              <button
                onClick={onRetry}
                className="text-xs text-[#0A3D91] hover:underline"
              >
                Retry
              </button>
            </div>
          ) : benchmarks?.naive_bayes_climate_checker ? (
            <div className="flex flex-col items-center">
              {/* Main Accuracy */}
              <p className="text-3xl font-extrabold bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent py-1">
                {benchmarks.climate_checker_multiple_runs
                  ? benchmarks.climate_checker_multiple_runs.statistics.accuracy.mean.toFixed(
                      2
                    )
                  : benchmarks.naive_bayes_climate_checker.toFixed(2)}
                %
              </p>

              {/* Show ± std if multiple runs available */}
              {benchmarks.climate_checker_multiple_runs && (
                <div className="flex items-center gap-1">
                  <p className="text-base text-[#6B7280] font-medium">
                    ±{" "}
                    {benchmarks.climate_checker_multiple_runs.statistics.accuracy.std.toFixed(
                      2
                    )}
                    %
                  </p>
                  <span className="text-xs text-[#9CA3AF]">
                    ({benchmarks.climate_checker_multiple_runs.number_of_runs}{" "}
                    runs)
                  </span>
                </div>
              )}
            </div>
          ) : (
            <p className="text-xl text-[#9CA3AF]">N/A</p>
          )}
          <p className="text-base text-[#1E293B] text-center font-semibold">
            Naive Bayes - Climate Related Checker
          </p>
        </div>

        {/* Naive Bayes - Climate Domain Identifier */}
        <div className="flex flex-col justify-center items-center gap-2">
          {loading ? (
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#0A3D91]"></div>
          ) : error ? (
            <div className="flex flex-col items-center gap-1">
              <p className="text-xl text-red-500">Error</p>
              <button
                onClick={onRetry}
                className="text-xs text-[#0A3D91] hover:underline"
              >
                Retry
              </button>
            </div>
          ) : benchmarks?.multiple_runs ||
            benchmarks?.naive_bayes_domain_identifier ? (
            <div className="flex flex-col items-center">
              {/* Main Accuracy */}
              <p className="text-3xl font-extrabold bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent py-1">
                {benchmarks.multiple_runs
                  ? benchmarks.multiple_runs.statistics.accuracy.mean.toFixed(2)
                  : benchmarks.naive_bayes_domain_identifier.toFixed(2)}
                %
              </p>

              {/* Show ± std if multiple runs available */}
              {benchmarks.multiple_runs && (
                <div className="flex items-center gap-1">
                  <p className="text-base text-[#6B7280] font-medium">
                    ±{" "}
                    {benchmarks.multiple_runs.statistics.accuracy.std.toFixed(
                      2
                    )}
                    %
                  </p>
                  <span className="text-xs text-[#9CA3AF]">
                    ({benchmarks.multiple_runs.number_of_runs} runs)
                  </span>
                </div>
              )}
            </div>
          ) : (
            <p className="text-xl text-[#9CA3AF]">N/A</p>
          )}

          <p className="text-base text-[#1E293B] text-center font-semibold">
            Naive Bayes - Climate Domain Identifier
          </p>
        </div>
      </div>

      {/* View Details Button */}
      <div className="mt-4 flex justify-center">
        <button
          onClick={onViewDetails}
          className="text-[#0A3D91] font-bold text-base hover:text-[#1E293B] underline transition-all cursor-pointer"
        >
          View Details
        </button>
      </div>
    </div>
  );
}
