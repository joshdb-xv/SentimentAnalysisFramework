"use client";

import { useState, useEffect } from "react";

export default function TwitterScraperDebug() {
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(10);
  const [threshold, setThreshold] = useState(0.85);
  const [useExpansion, setUseExpansion] = useState(true);
  const [loading, setLoading] = useState(false);
  const [currentTask, setCurrentTask] = useState(null);
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  const API_BASE = "http://localhost:8000";

  // Poll for status updates
  useEffect(() => {
    if (!currentTask) return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch(
          `${API_BASE}/twitter/status/${currentTask}`
        );
        const data = await response.json();
        setStatus(data);

        if (
          data.status === "completed" ||
          data.status === "failed" ||
          data.status === "cancelled"
        ) {
          setLoading(false);
          clearInterval(interval);
          fetchHistory();
        }
      } catch (err) {
        console.error("Error fetching status:", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [currentTask]);

  // Fetch history on mount
  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await fetch(`${API_BASE}/twitter/history?limit=5`);
      const data = await response.json();
      setHistory(data.history || []);
    } catch (err) {
      console.error("Error fetching history:", err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setStatus(null);

    try {
      const response = await fetch(`${API_BASE}/twitter/scrape`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          limit: parseInt(limit),
          similarity_threshold: parseFloat(threshold),
          use_expansion: useExpansion,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to start scraping");
      }

      const data = await response.json();
      setCurrentTask(data.task_id);
      setStatus(data);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleCancel = async () => {
    if (!currentTask) return;

    try {
      await fetch(`${API_BASE}/twitter/cancel/${currentTask}`, {
        method: "DELETE",
      });
      setLoading(false);
      setCurrentTask(null);
    } catch (err) {
      console.error("Error cancelling task:", err);
    }
  };

  const downloadCSV = () => {
    if (!status || !status.tweets || status.tweets.length === 0) return;

    const tweets = status.tweets;
    const headers = Object.keys(tweets[0]);
    const csvContent = [
      headers.join(","),
      ...tweets.map((tweet) =>
        headers
          .map((header) => {
            const value = tweet[header];
            if (value === null || value === undefined) return "";
            const stringValue = String(value).replace(/"/g, '""');
            return `"${stringValue}"`;
          })
          .join(",")
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `tweets_${query.replace(/\s+/g, "_")}_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gray-50 mt-10 py-16 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            🐦 Twitter Scraper Debug
          </h1>
          <p className="text-gray-600">
            Search and scrape tweets from the Philippines with content filtering
          </p>
        </div>

        {/* Scraping Form */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Search Query
              </label>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter search keyword or phrase..."
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
                disabled={loading}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Tweet Limit
                </label>
                <input
                  type="number"
                  value={limit}
                  onChange={(e) => setLimit(e.target.value)}
                  min="1"
                  max="1000"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={loading}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Duplicate Threshold
                </label>
                <input
                  type="number"
                  value={threshold}
                  onChange={(e) => setThreshold(e.target.value)}
                  min="0.8"
                  max="0.95"
                  step="0.05"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={loading}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Word Expansion
                </label>
                <div className="flex items-center h-10">
                  <input
                    type="checkbox"
                    checked={useExpansion}
                    onChange={(e) => setUseExpansion(e.target.checked)}
                    className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                    disabled={loading}
                  />
                  <span className="ml-2 text-sm text-gray-600">
                    Enable Filipino variations
                  </span>
                </div>
              </div>
            </div>

            <div className="flex gap-4">
              <button
                type="submit"
                disabled={loading || !query}
                className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
              >
                {loading ? "🔄 Scraping..." : "🚀 Start Scraping"}
              </button>

              {loading && (
                <button
                  type="button"
                  onClick={handleCancel}
                  className="px-6 bg-red-600 text-white py-2 rounded-lg hover:bg-red-700 transition-colors font-medium"
                >
                  ✕ Cancel
                </button>
              )}
            </div>
          </form>

          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-800 text-sm">❌ {error}</p>
            </div>
          )}
        </div>

        {/* Status Display */}
        {status && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">
              📊 Scraping Status
            </h2>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Status:</span>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${
                    status.status === "completed"
                      ? "bg-green-100 text-green-800"
                      : status.status === "failed"
                      ? "bg-red-100 text-red-800"
                      : status.status === "cancelled"
                      ? "bg-gray-100 text-gray-800"
                      : "bg-blue-100 text-blue-800"
                  }`}
                >
                  {status.status}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-gray-600">Query:</span>
                <span className="font-medium text-gray-800">
                  {status.query}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-gray-600">Target:</span>
                <span className="font-medium text-gray-800">
                  {status.limit} tweets
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-gray-600">Progress:</span>
                <span className="font-medium text-gray-800">
                  {status.tweets_collected || 0} / {status.limit}
                </span>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${status.progress || 0}%` }}
                />
              </div>

              {/* Statistics */}
              {status.statistics && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <h3 className="font-medium text-gray-800 mb-3">
                    Statistics:
                  </h3>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <span className="text-gray-600">Total Tweets:</span>
                      <span className="ml-2 font-medium">
                        {status.statistics.total_tweets}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Total Likes:</span>
                      <span className="ml-2 font-medium">
                        {status.statistics.total_likes.toLocaleString()}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Total Retweets:</span>
                      <span className="ml-2 font-medium">
                        {status.statistics.total_retweets.toLocaleString()}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Total Replies:</span>
                      <span className="ml-2 font-medium">
                        {status.statistics.total_replies.toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Download Button */}
              {status.status === "completed" &&
                status.tweets &&
                status.tweets.length > 0 && (
                  <button
                    onClick={downloadCSV}
                    className="w-full mt-4 bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors font-medium"
                  >
                    ⬇️ Download CSV
                  </button>
                )}

              {/* Sample Tweets */}
              {status.tweets && status.tweets.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <h3 className="font-medium text-gray-800 mb-3">
                    Sample Tweets:
                  </h3>
                  <div className="space-y-2 max-h-60 overflow-y-auto">
                    {status.tweets.slice(0, 5).map((tweet, idx) => (
                      <div
                        key={idx}
                        className="p-3 bg-gray-50 rounded-lg text-sm"
                      >
                        <p className="text-gray-800">{tweet.text}</p>
                        <div className="mt-2 flex gap-4 text-xs text-gray-500">
                          <span>❤️ {tweet.like_count}</span>
                          <span>🔄 {tweet.retweet_count}</span>
                          <span>💬 {tweet.reply_count}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* History */}
        {history.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">
              📜 Recent Scraping History
            </h2>
            <div className="space-y-3">
              {history.map((task) => (
                <div
                  key={task.task_id}
                  className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <p className="font-medium text-gray-800">{task.query}</p>
                      <p className="text-sm text-gray-600 mt-1">
                        {task.tweets_collected} / {task.limit} tweets collected
                      </p>
                    </div>
                    <span
                      className={`px-2 py-1 rounded text-xs font-medium ${
                        task.status === "completed"
                          ? "bg-green-100 text-green-800"
                          : task.status === "failed"
                          ? "bg-red-100 text-red-800"
                          : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      {task.status}
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    {new Date(task.created_at).toLocaleString()}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
