import { MdDownload, MdRefresh } from "react-icons/md";

export default function ResultsSection({
  results,
  downloadCSV,
  resetAll,
  setShowJsonModal,
}) {
  const getStat = (statName) => {
    const value = results?.stats?.[statName];
    if (value === null || value === undefined || isNaN(value)) {
      return 0;
    }
    return Number(value);
  };

  return (
    <div className="bg-bluish-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl">
      <div className="flex justify-between items-center mb-6">
        <div className="flex flex-row items-center gap-4">
          <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
          <p className="font-medium text-xl text-[#1E293B]">RESULTS</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={downloadCSV}
            className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-primary to-primary-dark text-white rounded-xl hover:from-primary-dark hover:to-black transition shadow-sm hover:shadow-md"
          >
            <MdDownload className="text-xl" />
            Download CSV
          </button>
          <button
            onClick={resetAll}
            className="flex items-center gap-2 px-4 py-2.5 bg-gray-mid text-white rounded-xl hover:bg-primary-dark transition shadow-sm hover:shadow-md"
          >
            <MdRefresh className="text-xl" />
            Start Again
          </button>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="p-5 bg-gradient-to-br from-bluish-white to-bluish-gray rounded-xl">
          <p className="text-sm font-semibold tracking-wide text-primary mb-1">
            Total Words
          </p>
          <p className="text-3xl font-bold text-primary tracking-wide pl-6">
            {getStat("total_words").toLocaleString()}
          </p>
        </div>
        <div className="p-5 bg-gradient-to-br from-bluish-white to-bluish-gray rounded-xl">
          <p className="text-sm font-semibold tracking-wide text-primary mb-1">
            Climate Related
          </p>
          <p className="text-3xl font-bold text-primary tracking-wide pl-6">
            {getStat("climate_related").toLocaleString()}
          </p>
        </div>
        <div className="p-5 bg-gradient-to-br from-bluish-white to-bluish-gray rounded-xl">
          <p className="text-sm font-semibold tracking-wide text-blue mb-1">
            Positive
          </p>
          <p className="text-3xl font-bold text-blue tracking-wide pl-6">
            {getStat("positive_words").toLocaleString()}
          </p>
        </div>
        <div className="p-5 bg-gradient-to-br from-bluish-white to-bluish-gray rounded-xl">
          <p className="text-sm font-semibold tracking-wide text-red mb-1">
            Negative
          </p>
          <p className="text-3xl font-bold text-red tracking-wide pl-6">
            {getStat("negative_words").toLocaleString()}
          </p>
        </div>
      </div>

      {/* Additional Stats */}
      {results.stats && (
        <div className="mb-6 p-5 bg-white/50 border border-bluish-gray rounded-xl">
          <h3 className="font-semibold text-black mb-3">
            Average Sentiment Scores
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="flex flex-col">
              <span className="text-gray-mid text-xs mb-1">Climate Words</span>
              <span className="font-mono text-lg font-bold text-black">
                {(getStat("avg_score_climate") || 0).toFixed(3)}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-gray-mid text-xs mb-1">General Words</span>
              <span className="font-mono text-lg font-bold text-black">
                {(getStat("avg_score_general") || 0).toFixed(3)}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-gray-mid text-xs mb-1">Non-Climate</span>
              <span className="font-mono text-lg font-bold text-black">
                {getStat("non_climate").toLocaleString()}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-gray-mid text-xs mb-1">Neutral</span>
              <span className="font-mono text-lg font-bold text-black">
                {getStat("neutral_words").toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Preview Table */}
      <div>
        <h3 className="text-base font-semibold text-black mb-3">
          Preview (First 50 words)
        </h3>
        {results.preview && results.preview.length > 0 ? (
          <div className="overflow-x-auto border border-bluish-gray rounded-xl max-h-96 overflow-y-auto">
            <table className="min-w-full text-sm divide-y divide-bluish-gray">
              <thead className="bg-white/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-primary-dark uppercase tracking-wider">
                    Word
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-primary-dark uppercase tracking-wider">
                    Sentiment Score
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-bluish-gray">
                {results.preview.slice(0, 50).map((item, idx) => (
                  <tr key={idx} className="hover:bg-white/50 transition">
                    <td className="px-4 py-3 font-mono text-black">
                      {item.word}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`font-mono font-semibold ${
                          item.sentiment_score > 0
                            ? "text-primary"
                            : item.sentiment_score < 0
                            ? "text-red"
                            : "text-gray-mid"
                        }`}
                      >
                        {item.sentiment_score?.toFixed(3) ?? "0.000"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="bg-white/50 border border-bluish-gray rounded-xl p-8 text-center">
            <p className="text-gray">No preview data available</p>
          </div>
        )}
      </div>

      {/* Debug Info */}
      <div className="mt-6">
        <button
          onClick={() => setShowJsonModal(true)}
          className="px-4 py-2 text-sm border border-gray-light rounded-lg hover:bg-bluish-gray transition text-primary-dark font-medium"
        >
          Show raw JSON data
        </button>
      </div>
    </div>
  );
}
