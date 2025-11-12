import { MdSearch } from "react-icons/md";

export default function SearchResultModal({
  searchResult,
  showSearchModal,
  setShowSearchModal,
  setSearchWord,
}) {
  if (!showSearchModal || !searchResult) return null;

  const getScoreColor = (score) => {
    if (score > 0) return "text-blue";
    if (score < 0) return "text-red";
    return "text-gray-mid";
  };

  const getScoreBgColor = (score) => {
    if (score > 0) return "bg-blue/10 border-blue";
    if (score < 0) return "bg-red/10 border-red";
    return "bg-gray/10 border-gray";
  };

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
      onClick={() => setShowSearchModal(false)}
    >
      <div
        className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-6 border-b border-bluish-gray bg-gradient-to-r from-primary/5 to-primary/10">
          <h3 className="text-xl font-bold text-primary-dark flex items-center gap-2">
            <MdSearch className="text-2xl" />
            Search Result
          </h3>
          <button
            onClick={() => setShowSearchModal(false)}
            className="text-gray-light hover:text-gray-mid transition"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          {searchResult.found ? (
            <div className="space-y-6">
              {/* Word Header */}
              <div className="text-center pb-4 border-b border-bluish-gray">
                <p className="text-4xl font-bold text-primary-dark font-mono mb-2">
                  {searchResult.word}
                </p>
                <div className="flex items-center justify-center gap-3">
                  <span
                    className={`px-4 py-1.5 rounded-full text-sm font-semibold ${
                      searchResult.polarity === "positive"
                        ? "bg-blue/20 text-blue"
                        : searchResult.polarity === "negative"
                        ? "bg-red/20 text-red"
                        : "bg-gray/20 text-gray-mid"
                    }`}
                  >
                    {searchResult.polarity.toUpperCase()}
                  </span>
                  <span className="px-4 py-1.5 bg-primary/10 text-primary rounded-full text-sm font-semibold">
                    {searchResult.intensity.toUpperCase()}
                  </span>
                </div>
              </div>

              {/* Sentiment Score */}
              <div
                className={`p-6 rounded-xl border-2 ${getScoreBgColor(
                  searchResult.sentiment_score
                )}`}
              >
                <p className="text-sm font-semibold text-gray-mid mb-2 text-center">
                  SENTIMENT SCORE
                </p>
                <p
                  className={`text-5xl font-bold text-center font-mono ${getScoreColor(
                    searchResult.sentiment_score
                  )}`}
                >
                  {searchResult.sentiment_score > 0 ? "+" : ""}
                  {searchResult.sentiment_score.toFixed(3)}
                </p>
              </div>

              {/* Interpretation */}
              <div className="bg-bluish-white p-4 rounded-xl border border-bluish-gray">
                <p className="text-sm font-semibold text-primary-dark mb-2">
                  INTERPRETATION
                </p>
                <p className="text-gray-mid">{searchResult.interpretation}</p>
              </div>

              {/* Detailed Breakdown */}
              {searchResult.detailed_breakdown && (
                <div className="bg-white p-6 rounded-xl border-2 border-primary/30">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-3 h-3 bg-gradient-to-r from-primary to-primary-dark rounded-full"></div>
                    <p className="text-lg font-bold text-primary-dark">
                      HOW THE SCORE WAS CALCULATED
                    </p>
                  </div>

                  {/* Stage 1: Base Polarity */}
                  {searchResult.detailed_breakdown.stages?.stage_1_base && (
                    <div className="mb-4 p-4 bg-blue/5 rounded-xl border border-blue/20">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-6 h-6 bg-blue text-white rounded-full flex items-center justify-center text-xs font-bold">
                          1
                        </div>
                        <p className="font-semibold text-blue">
                          Base Polarity (Manual Labels)
                        </p>
                      </div>
                      <p className="text-sm text-gray-mid mb-2">
                        {
                          searchResult.detailed_breakdown.stages.stage_1_base
                            .explanation
                        }
                      </p>
                      <div className="bg-white p-3 rounded-lg border border-blue/20">
                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <span className="text-gray-mid">Polarity:</span>
                            <span className="ml-2 font-mono font-bold text-blue">
                              {
                                searchResult.detailed_breakdown.stages
                                  .stage_1_base.polarity
                              }
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-mid">
                              Base Magnitude:
                            </span>
                            <span className="ml-2 font-mono font-bold text-blue">
                              {
                                searchResult.detailed_breakdown.stages
                                  .stage_1_base.base_magnitude
                              }
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Stage 2: Semantic Intensity */}
                  {searchResult.detailed_breakdown.stages
                    ?.stage_2_intensity && (
                    <div className="mb-4 p-4 bg-purple-500/5 rounded-xl border border-purple-500/20">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                          2
                        </div>
                        <p className="font-semibold text-purple-700">
                          Semantic Intensity (FastText Embeddings)
                        </p>
                      </div>
                      <p className="text-sm text-gray-mid mb-3">
                        {
                          searchResult.detailed_breakdown.stages
                            .stage_2_intensity.explanation
                        }
                      </p>

                      {searchResult.detailed_breakdown.stages.stage_2_intensity
                        .details && (
                        <div className="bg-white p-3 rounded-lg border border-purple-500/20 space-y-2">
                          <div className="grid grid-cols-2 gap-3 text-sm">
                            <div>
                              <span className="text-gray-mid">
                                Similarity to Positive:
                              </span>
                              <span className="ml-2 font-mono font-bold text-blue">
                                {searchResult.detailed_breakdown.stages.stage_2_intensity.details.similarity_to_positive?.toFixed(
                                  4
                                ) || "N/A"}
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-mid">
                                Similarity to Negative:
                              </span>
                              <span className="ml-2 font-mono font-bold text-red">
                                {searchResult.detailed_breakdown.stages.stage_2_intensity.details.similarity_to_negative?.toFixed(
                                  4
                                ) || "N/A"}
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-mid">
                                Semantic Strength:
                              </span>
                              <span className="ml-2 font-mono font-bold text-purple-700">
                                {searchResult.detailed_breakdown.stages.stage_2_intensity.details.semantic_strength?.toFixed(
                                  4
                                ) || "N/A"}
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-mid">
                                Semantic Clarity:
                              </span>
                              <span className="ml-2 font-mono font-bold text-purple-700">
                                {searchResult.detailed_breakdown.stages.stage_2_intensity.details.semantic_clarity?.toFixed(
                                  4
                                ) || "N/A"}
                              </span>
                            </div>
                          </div>
                          <div className="pt-2 border-t border-purple-500/10">
                            <span className="text-gray-mid text-sm">
                              Intensity Multiplier:
                            </span>
                            <span className="ml-2 font-mono font-bold text-purple-700 text-lg">
                              {
                                searchResult.detailed_breakdown.stages
                                  .stage_2_intensity.intensity_multiplier
                              }
                            </span>
                          </div>
                          {searchResult.detailed_breakdown.stages.stage_2_intensity.details.notes?.map(
                            (note, idx) => (
                              <p
                                key={idx}
                                className="text-xs text-gray-mid italic"
                              >
                                • {note}
                              </p>
                            )
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Stage 3: Domain Weighting */}
                  {searchResult.detailed_breakdown.stages?.stage_3_domain && (
                    <div className="mb-4 p-4 bg-green-500/5 rounded-xl border border-green-500/20">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                          3
                        </div>
                        <p className="font-semibold text-green-700">
                          Domain Weighting (Climate-Specific)
                        </p>
                      </div>
                      <p className="text-sm text-gray-mid mb-2">
                        {
                          searchResult.detailed_breakdown.stages.stage_3_domain
                            .explanation
                        }
                      </p>
                      <div className="bg-white p-3 rounded-lg border border-green-500/20">
                        <div className="flex items-center justify-between">
                          <span className="text-gray-mid">Domain Weight:</span>
                          <span className="font-mono font-bold text-green-700 text-lg">
                            {
                              searchResult.detailed_breakdown.stages
                                .stage_3_domain.domain_weight
                            }
                            x
                          </span>
                        </div>
                        <p className="text-xs text-gray-mid mt-2 italic">
                          {
                            searchResult.detailed_breakdown.stages
                              .stage_3_domain.reason
                          }
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Final Calculation */}
                  {searchResult.detailed_breakdown.calculation && (
                    <div className="p-5 bg-gradient-to-r from-primary/10 to-primary/5 rounded-xl border-2 border-primary">
                      <p className="font-semibold text-primary-dark mb-3 flex items-center gap-2">
                        <span className="text-2xl">🧮</span>
                        Final Calculation
                      </p>
                      <div className="bg-white p-4 rounded-lg space-y-2">
                        <div className="text-sm text-gray-mid">
                          <span className="font-semibold">Formula:</span>
                          <p className="font-mono text-primary-dark mt-1">
                            {
                              searchResult.detailed_breakdown.calculation
                                .formula
                            }
                          </p>
                        </div>
                        <div className="text-sm text-gray-mid">
                          <span className="font-semibold">
                            Substituted Values:
                          </span>
                          <p className="font-mono text-primary-dark mt-1">
                            {
                              searchResult.detailed_breakdown.calculation
                                .substituted
                            }
                          </p>
                        </div>
                        <div className="pt-3 border-t border-primary/20">
                          <span className="text-gray-mid">Final Score:</span>
                          <span
                            className={`ml-3 font-mono font-bold text-2xl ${getScoreColor(
                              searchResult.detailed_breakdown.calculation.result
                            )}`}
                          >
                            {searchResult.detailed_breakdown.calculation
                              .result > 0
                              ? "+"
                              : ""}
                            {searchResult.detailed_breakdown.calculation.result.toFixed(
                              3
                            )}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Word Metadata */}
                  <div className="mt-4 p-4 bg-bluish-white rounded-xl border border-bluish-gray">
                    <p className="text-sm font-semibold text-primary-dark mb-2">
                      WORD INFORMATION
                    </p>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <span className="text-gray-mid">Dialect:</span>
                        <span className="ml-2 font-semibold text-black capitalize">
                          {searchResult.detailed_breakdown.dialect || "N/A"}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-mid">Climate-Related:</span>
                        <span className="ml-2 font-semibold text-black">
                          {searchResult.detailed_breakdown.is_climate_related
                            ? "Yes"
                            : "No"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="w-20 h-20 bg-gray/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <MdSearch className="text-4xl text-gray" />
              </div>
              <p className="text-xl font-semibold text-gray-dark mb-2">
                Word Not Found
              </p>
              <p className="text-gray-mid mb-6">
                The word "
                <span className="font-mono font-semibold">
                  {searchResult.word}
                </span>
                " was not found in the lexical dictionary.
              </p>

              {searchResult.suggestions &&
                searchResult.suggestions.length > 0 && (
                  <div className="bg-bluish-white p-4 rounded-xl border border-bluish-gray">
                    <p className="text-sm font-semibold text-primary-dark mb-3">
                      Did you mean one of these?
                    </p>
                    <div className="flex flex-wrap gap-2 justify-center">
                      {searchResult.suggestions.map((suggestion, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            setSearchWord(suggestion);
                            setShowSearchModal(false);
                          }}
                          className="px-3 py-1.5 bg-white border border-bluish-gray rounded-lg hover:border-primary hover:bg-primary/5 transition text-sm font-mono text-primary-dark"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
            </div>
          )}
        </div>

        <div className="p-4 border-t border-bluish-gray bg-bluish-white flex justify-end">
          <button
            onClick={() => setShowSearchModal(false)}
            className="px-6 py-2.5 bg-primary text-white rounded-xl hover:bg-primary-dark transition font-medium"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
