"use client";

import {
  MdChevronRight,
  MdUpload,
  MdPlayArrow,
  MdDownload,
  MdRefresh,
  MdCheckCircle,
  MdError,
} from "react-icons/md";
import { useState, useEffect } from "react";

export default function LexicalDictionary() {
  const [dictionaryFile, setDictionaryFile] = useState(null);
  const [keywordsFile, setKeywordsFile] = useState(null);
  const [dictionaryInfo, setDictionaryInfo] = useState(null);
  const [keywordsInfo, setKeywordsInfo] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [uploadingDict, setUploadingDict] = useState(false);
  const [uploadingKeywords, setUploadingKeywords] = useState(false);
  const [loadingKeywords, setLoadingKeywords] = useState(false);
  const [showJsonModal, setShowJsonModal] = useState(false);

  // Poll status when processing
  useEffect(() => {
    let interval;
    if (processing) {
      interval = setInterval(async () => {
        try {
          const response = await fetch("http://localhost:8000/lexical/status");
          const data = await response.json();
          setStatus(data);

          if (data.status === "completed") {
            setProcessing(false);
            fetchResults();
          } else if (data.status === "error") {
            setProcessing(false);
          }
        } catch (error) {
          console.error("Error fetching status:", error);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [processing]);

  const handleDictionaryUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setDictionaryFile(file);
    setUploadingDict(true);
    setDictionaryInfo(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        "http://localhost:8000/lexical/upload-dictionary",
        {
          method: "POST",
          body: formData,
        }
      );
      const data = await response.json();
      setDictionaryInfo(data);
    } catch (error) {
      console.error("Error uploading dictionary:", error);
      alert("Failed to upload dictionary file");
    } finally {
      setUploadingDict(false);
    }
  };

  const handleKeywordsLoad = async () => {
    setLoadingKeywords(true);
    setKeywordsInfo(null);

    try {
      const response = await fetch(
        "http://localhost:8000/lexical/load-climate-keywords"
      );
      const data = await response.json();
      setKeywordsInfo(data);
    } catch (error) {
      console.error("Error loading keywords:", error);
      alert(
        "Failed to load climate keywords. Make sure data/weatherkeywords.csv exists."
      );
    } finally {
      setLoadingKeywords(false);
    }
  };

  const handleKeywordsUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setKeywordsFile(file);
    setUploadingKeywords(true);
    setKeywordsInfo(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        "http://localhost:8000/lexical/upload-climate-keywords",
        {
          method: "POST",
          body: formData,
        }
      );
      const data = await response.json();
      setKeywordsInfo(data);
    } catch (error) {
      console.error("Error uploading keywords:", error);
      alert("Failed to upload keywords file");
    } finally {
      setUploadingKeywords(false);
    }
  };

  const startProcessing = async () => {
    if (!dictionaryInfo) {
      alert("Please upload dictionary file first");
      return;
    }

    const keywordsPath = keywordsInfo?.file_path || null;

    try {
      const response = await fetch("http://localhost:8000/lexical/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dictionary_path: dictionaryInfo.file_path,
          climate_keywords_path: keywordsPath,
        }),
      });

      if (response.ok) {
        setProcessing(true);
        setResults(null);
      } else {
        const error = await response.json();
        alert(error.detail || "Failed to start processing");
      }
    } catch (error) {
      console.error("Error starting processing:", error);
      alert("Failed to start processing");
    }
  };

  const fetchResults = async () => {
    try {
      const response = await fetch("http://localhost:8000/lexical/results");
      const data = await response.json();
      console.log("Results data:", data);
      setResults(data);
    } catch (error) {
      console.error("Error fetching results:", error);
    }
  };

  const downloadCSV = () => {
    window.open("http://localhost:8000/lexical/download", "_blank");
  };

  const resetAll = async () => {
    try {
      await fetch("http://localhost:8000/lexical/reset", { method: "POST" });
      setDictionaryFile(null);
      setKeywordsFile(null);
      setDictionaryInfo(null);
      setKeywordsInfo(null);
      setStatus(null);
      setResults(null);
      setProcessing(false);
    } catch (error) {
      console.error("Error resetting:", error);
    }
  };

  const getStat = (statName) => {
    const value = results?.stats?.[statName];
    if (value === null || value === undefined || isNaN(value)) {
      return 0;
    }
    return Number(value);
  };

  return (
    <div className="min-h-screen bg-[#F8FAFC] relative">
      {/* Header */}
      <div className="px-6 w-full h-20 flex items-center border-b border-gray-200 gap-4 bg-white">
        <div className="w-fit text-2xl font-bold bg-gradient-to-r from-[#222222] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent leading-snug">
          Sentiment Analysis Framework
        </div>
        <MdChevronRight className="text-3xl text-gray-400" />
        <p className="text-gray-700 text-2xl font-medium">Lexical Dictionary</p>
      </div>

      {/* Main Content */}
      <div className="p-6 max-w-7xl mx-auto space-y-6">
        {/* Upload Section - Hidden when results are shown */}
        {!results && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Dictionary Upload */}
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                1. Upload Dictionary (Excel)
              </h2>

              <label className="relative flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-xl p-8 cursor-pointer hover:border-blue-500 hover:bg-blue-50/50 transition group">
                <MdUpload className="text-5xl text-gray-400 group-hover:text-blue-500 mb-2 transition" />
                <span className="text-sm text-gray-600 text-center">
                  {uploadingDict ? (
                    <span className="text-blue-600 font-medium">
                      Uploading...
                    </span>
                  ) : dictionaryFile ? (
                    <span className="text-gray-700">{dictionaryFile.name}</span>
                  ) : (
                    "Click to upload Excel file"
                  )}
                </span>
                <input
                  type="file"
                  accept=".xlsx,.xls"
                  onChange={handleDictionaryUpload}
                  disabled={uploadingDict}
                  className="hidden"
                />

                {uploadingDict && (
                  <div className="absolute inset-0 bg-white/80 rounded-xl flex items-center justify-center">
                    <div className="flex flex-col items-center gap-2">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                      <span className="text-sm text-blue-600 font-medium">
                        Loading...
                      </span>
                    </div>
                  </div>
                )}
              </label>

              {dictionaryInfo && (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-xl text-sm">
                  <div className="flex items-start gap-2">
                    <MdCheckCircle className="text-green-600 text-xl flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <p className="font-semibold text-green-800">
                        Uploaded successfully
                      </p>
                      <p className="text-gray-700 mt-1">
                        Rows:{" "}
                        <span className="font-mono font-semibold">
                          {dictionaryInfo.rows.toLocaleString()}
                        </span>
                      </p>
                      <p className="text-gray-700">
                        Columns:{" "}
                        <span className="font-mono text-xs">
                          {dictionaryInfo.columns.join(", ")}
                        </span>
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Keywords Load/Upload */}
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                2. Climate Keywords (CSV)
              </h2>

              {/* Load from default location */}
              <button
                onClick={handleKeywordsLoad}
                disabled={loadingKeywords}
                className="relative w-full mb-4 flex items-center justify-center gap-2 border-2 border-blue-500 bg-blue-50 rounded-xl p-4 hover:bg-blue-100 transition disabled:opacity-50 disabled:cursor-not-allowed group"
              >
                <MdUpload className="text-2xl text-blue-600" />
                <span className="text-blue-700 font-medium">
                  {loadingKeywords
                    ? "Loading..."
                    : "Load from data/weatherkeywords.csv"}
                </span>

                {loadingKeywords && (
                  <div className="absolute inset-0 bg-blue-50/80 rounded-xl flex items-center justify-center">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                  </div>
                )}
              </button>

              {/* Or upload custom file */}
              <div className="relative my-4">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-200"></div>
                </div>
                <div className="relative flex justify-center text-xs">
                  <span className="px-2 bg-white text-gray-500">
                    Or upload custom file
                  </span>
                </div>
              </div>

              <label className="relative flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-xl p-6 cursor-pointer hover:border-blue-500 hover:bg-blue-50/50 transition group">
                <MdUpload className="text-4xl text-gray-400 group-hover:text-blue-500 mb-2 transition" />
                <span className="text-sm text-gray-600">
                  {uploadingKeywords ? (
                    <span className="text-blue-600 font-medium">
                      Uploading...
                    </span>
                  ) : keywordsFile ? (
                    keywordsFile.name
                  ) : (
                    "Click to upload custom CSV"
                  )}
                </span>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleKeywordsUpload}
                  disabled={uploadingKeywords}
                  className="hidden"
                />

                {uploadingKeywords && (
                  <div className="absolute inset-0 bg-white/80 rounded-xl flex items-center justify-center">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                  </div>
                )}
              </label>

              {keywordsInfo && (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-xl text-sm">
                  <div className="flex items-start gap-2">
                    <MdCheckCircle className="text-green-600 text-xl flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                      <p className="font-semibold text-green-800">
                        Keywords loaded successfully
                      </p>
                      <p className="text-gray-700 mt-1">
                        Keywords:{" "}
                        <span className="font-mono font-semibold">
                          {keywordsInfo.keyword_count}
                        </span>
                      </p>
                      <p className="text-xs text-gray-600 mt-1 font-mono truncate">
                        {keywordsInfo.file_path}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Processing Controls - Hidden when results are shown */}
        {!results && (
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              3. Process Dictionary
            </h2>

            {/* Info about FastText */}
            <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-xl">
              <p className="text-sm text-blue-800 flex items-start gap-2">
                <span className="text-blue-600 text-lg">⚡</span>
                <span>
                  <strong>FastText embeddings are automatically used</strong>{" "}
                  for semantic scoring. Models are pre-loaded at server startup
                  for fast processing.
                </span>
              </p>
            </div>

            <div className="flex gap-3">
              <button
                onClick={startProcessing}
                disabled={!dictionaryInfo || processing}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl hover:from-blue-700 hover:to-blue-800 disabled:from-gray-300 disabled:to-gray-400 disabled:cursor-not-allowed transition shadow-sm hover:shadow-md"
              >
                <MdPlayArrow className="text-xl" />
                {processing ? "Processing..." : "Start Processing"}
              </button>

              <button
                onClick={resetAll}
                className="flex items-center gap-2 px-6 py-3 bg-gray-100 text-gray-700 rounded-xl hover:bg-gray-200 transition"
              >
                <MdRefresh className="text-xl" />
                Reset
              </button>
            </div>

            {!keywordsInfo && !processing && dictionaryInfo && (
              <p className="text-sm text-gray-600 mt-3 flex items-center gap-2">
                <span>💡</span>
                <span>
                  No keywords loaded. Will use default from{" "}
                  <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs font-mono">
                    data/weatherkeywords.csv
                  </code>
                </span>
              </p>
            )}

            {/* Progress */}
            {status && (
              <div className="mt-4">
                <div className="flex justify-between text-sm text-gray-700 mb-2">
                  <span className="font-medium">{status.message}</span>
                  <span className="font-semibold text-blue-600">
                    {status.progress}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${status.progress}%` }}
                  />
                </div>

                {status.status === "error" && (
                  <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-xl flex items-start gap-2">
                    <MdError className="text-red-600 text-xl flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-semibold text-red-800">
                        Error occurred
                      </p>
                      <p className="text-sm text-red-700 mt-1">
                        {status.message}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Results Section */}
        {results && results.stats && (
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-semibold text-gray-900">
                4. Results
              </h2>
              <div className="flex gap-3">
                <button
                  onClick={downloadCSV}
                  className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-green-600 to-green-700 text-white rounded-xl hover:from-green-700 hover:to-green-800 transition shadow-sm hover:shadow-md"
                >
                  <MdDownload className="text-xl" />
                  Download CSV
                </button>
                <button
                  onClick={resetAll}
                  className="flex items-center gap-2 px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition shadow-sm hover:shadow-md"
                >
                  <MdRefresh className="text-xl" />
                  Start Again
                </button>
              </div>
            </div>

            {/* Statistics Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="p-5 bg-gradient-to-br from-blue-50 to-blue-100/50 border border-blue-200 rounded-xl">
                <p className="text-xs font-medium text-blue-700 mb-1">
                  Total Words
                </p>
                <p className="text-3xl font-bold text-blue-900">
                  {getStat("total_words").toLocaleString()}
                </p>
              </div>
              <div className="p-5 bg-gradient-to-br from-green-50 to-green-100/50 border border-green-200 rounded-xl">
                <p className="text-xs font-medium text-green-700 mb-1">
                  Climate Related
                </p>
                <p className="text-3xl font-bold text-green-900">
                  {getStat("climate_related").toLocaleString()}
                </p>
              </div>
              <div className="p-5 bg-gradient-to-br from-purple-50 to-purple-100/50 border border-purple-200 rounded-xl">
                <p className="text-xs font-medium text-purple-700 mb-1">
                  Positive
                </p>
                <p className="text-3xl font-bold text-purple-900">
                  {getStat("positive_words").toLocaleString()}
                </p>
              </div>
              <div className="p-5 bg-gradient-to-br from-red-50 to-red-100/50 border border-red-200 rounded-xl">
                <p className="text-xs font-medium text-red-700 mb-1">
                  Negative
                </p>
                <p className="text-3xl font-bold text-red-900">
                  {getStat("negative_words").toLocaleString()}
                </p>
              </div>
            </div>

            {/* Additional Stats */}
            {results.stats && (
              <div className="mb-6 p-5 bg-gray-50 border border-gray-200 rounded-xl">
                <h3 className="font-semibold text-gray-900 mb-3">
                  Average Sentiment Scores
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="flex flex-col">
                    <span className="text-gray-600 text-xs mb-1">
                      Climate Words
                    </span>
                    <span className="font-mono text-lg font-bold text-gray-900">
                      {(getStat("avg_score_climate") || 0).toFixed(3)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-gray-600 text-xs mb-1">
                      General Words
                    </span>
                    <span className="font-mono text-lg font-bold text-gray-900">
                      {(getStat("avg_score_general") || 0).toFixed(3)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-gray-600 text-xs mb-1">
                      Non-Climate
                    </span>
                    <span className="font-mono text-lg font-bold text-gray-900">
                      {getStat("non_climate").toLocaleString()}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-gray-600 text-xs mb-1">Neutral</span>
                    <span className="font-mono text-lg font-bold text-gray-900">
                      {getStat("neutral_words").toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Preview Table */}
            <div>
              <h3 className="text-base font-semibold text-gray-900 mb-3">
                Preview (First 50 words)
              </h3>
              {results.preview && results.preview.length > 0 ? (
                <div className="overflow-x-auto border border-gray-200 rounded-xl max-h-96 overflow-y-auto">
                  <table className="min-w-full text-sm divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                          Word
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider">
                          Sentiment Score
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {results.preview.slice(0, 50).map((item, idx) => (
                        <tr key={idx} className="hover:bg-gray-50 transition">
                          <td className="px-4 py-3 font-mono text-gray-900">
                            {item.word}
                          </td>
                          <td className="px-4 py-3">
                            <span
                              className={`font-mono font-semibold ${
                                item.sentiment_score > 0
                                  ? "text-green-600"
                                  : item.sentiment_score < 0
                                  ? "text-red-600"
                                  : "text-gray-600"
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
                <div className="bg-gray-50 border border-gray-200 rounded-xl p-8 text-center">
                  <p className="text-gray-500">No preview data available</p>
                </div>
              )}
            </div>

            {/* Debug Info */}
            <div className="mt-6">
              <button
                onClick={() => setShowJsonModal(true)}
                className="px-4 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 transition text-gray-700 font-medium"
              >
                Show raw JSON data
              </button>
            </div>
          </div>
        )}
      </div>

      {/* JSON Modal */}
      {showJsonModal && (
        <div
          className="absolute inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
          onClick={() => setShowJsonModal(false)}
        >
          <div
            className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[80vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">
                Raw JSON Data
              </h3>
              <button
                onClick={() => setShowJsonModal(false)}
                className="text-gray-400 hover:text-gray-600 transition"
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
            <div
              className="flex-1 overflow-y-auto p-4 bg-gray-900"
              style={{
                scrollbarWidth: "none",
                msOverflowStyle: "none",
              }}
            >
              <style>{`
                .flex-1.overflow-y-auto::-webkit-scrollbar {
                  display: none;
                }
              `}</style>
              <pre className="text-xs font-mono text-green-400 whitespace-pre">
                {JSON.stringify(results, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
