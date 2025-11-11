export default function PreviewTable({ preview }) {
  if (!preview || preview.length === 0) {
    return (
      <div>
        <h3 className="text-base font-semibold text-black mb-3">
          Preview (First 50 words)
        </h3>
        <div className="bg-white/50 border border-bluish-gray rounded-xl p-8 text-center">
          <p className="text-gray">No preview data available</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h3 className="text-base font-semibold text-black mb-3">
        Preview (First 50 words)
      </h3>
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
            {preview.slice(0, 50).map((item, idx) => (
              <tr key={idx} className="hover:bg-white/50 transition">
                <td className="px-4 py-3 font-mono text-black">{item.word}</td>
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
    </div>
  );
}
