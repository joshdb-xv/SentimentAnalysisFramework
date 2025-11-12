// pages/lexical-debug.js
// Simple debug UI for lexical dictionary system

"use client";

import { useState, useEffect } from "react";

export default function LexicalDebug() {
  const [status, setStatus] = useState(null);
  const [searchWord, setSearchWord] = useState("");
  const [searchResult, setSearchResult] = useState(null);
  const [editWord, setEditWord] = useState("");
  const [editLabel, setEditLabel] = useState("positive");
  const [updateResult, setUpdateResult] = useState(null);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [processResult, setProcessResult] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [loading, setLoading] = useState({});

  const API_BASE = "http://localhost:8000";

  // Check dictionary status on load
  useEffect(() => {
    checkStatus();
  }, []);

  // Helper function for API calls
  const apiCall = async (endpoint, options = {}) => {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "API Error");
    }
    return data;
  };

  // Check dictionary status
  const checkStatus = async () => {
    setLoading({ ...loading, status: true });
    try {
      const data = await apiCall("/lexical/dictionary/status");
      setStatus(data);
    } catch (error) {
      alert("Error checking status: " + error.message);
    } finally {
      setLoading({ ...loading, status: false });
    }
  };

  // Load cached dictionary
  const loadDictionary = async () => {
    setLoading({ ...loading, load: true });
    try {
      const data = await apiCall("/lexical/dictionary/load", {
        method: "POST",
      });
      alert("Dictionary loaded: " + data.total_words + " words");
      checkStatus();
    } catch (error) {
      alert("Error loading: " + error.message);
    } finally {
      setLoading({ ...loading, load: false });
    }
  };

  // Upload dictionary file
  const uploadDictionary = async () => {
    if (!uploadFile) {
      alert("Please select a file");
      return;
    }
    setLoading({ ...loading, upload: true });
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);

      const response = await fetch(`${API_BASE}/lexical/upload-dictionary`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      setUploadResult(data);
      alert("File uploaded: " + data.rows + " rows");
    } catch (error) {
      alert("Error uploading: " + error.message);
    } finally {
      setLoading({ ...loading, upload: false });
    }
  };

  // Process dictionary
  const processDictionary = async () => {
    if (!uploadResult || !uploadResult.file_path) {
      alert("Please upload a dictionary first");
      return;
    }
    setLoading({ ...loading, process: true });
    try {
      const data = await apiCall("/lexical/process", {
        method: "POST",
        body: JSON.stringify({
          dictionary_path: uploadResult.file_path,
          climate_keywords_path: null,
        }),
      });

      setProcessResult(data);
      alert("Processing started!");

      // Poll for status
      const interval = setInterval(async () => {
        const statusData = await apiCall("/lexical/status");
        setProcessingStatus(statusData);

        if (statusData.status === "completed") {
          clearInterval(interval);
          alert("Processing completed!");
        } else if (statusData.status === "error") {
          clearInterval(interval);
          alert("Processing failed: " + statusData.message);
        }
      }, 2000);
    } catch (error) {
      alert("Error processing: " + error.message);
    } finally {
      setLoading({ ...loading, process: false });
    }
  };

  // Save to cache
  const saveToCache = async () => {
    setLoading({ ...loading, save: true });
    try {
      const data = await apiCall("/lexical/dictionary/save", {
        method: "POST",
      });
      alert("Saved to cache: " + data.path);
      checkStatus();
    } catch (error) {
      alert("Error saving: " + error.message);
    } finally {
      setLoading({ ...loading, save: false });
    }
  };

  // Search word
  const searchForWord = async () => {
    if (!searchWord.trim()) {
      alert("Please enter a word");
      return;
    }
    setLoading({ ...loading, search: true });
    try {
      const data = await apiCall("/lexical/search", {
        method: "POST",
        body: JSON.stringify({ word: searchWord }),
      });
      setSearchResult(data);

      // Pre-fill edit form if word found
      if (data.found) {
        setEditWord(data.word);
        setEditLabel(data.sentiment_label || "neutral");
      }
    } catch (error) {
      alert("Error searching: " + error.message);
    } finally {
      setLoading({ ...loading, search: false });
    }
  };

  // Update word label
  const updateWordLabel = async () => {
    if (!editWord.trim()) {
      alert("Please enter a word");
      return;
    }
    setLoading({ ...loading, update: true });
    try {
      const data = await apiCall("/lexical/dictionary/update-word", {
        method: "PUT",
        body: JSON.stringify({
          word: editWord,
          new_label: editLabel,
        }),
      });
      setUpdateResult(data);
      alert(`Updated! ${data.old_score} → ${data.new_score}`);

      // Refresh search
      if (searchWord === editWord) {
        searchForWord();
      }
    } catch (error) {
      alert("Error updating: " + error.message);
    } finally {
      setLoading({ ...loading, update: false });
    }
  };

  // Reset cache
  const resetCache = async () => {
    if (!confirm("Are you sure? This will delete the cached dictionary.")) {
      return;
    }
    setLoading({ ...loading, reset: true });
    try {
      await apiCall("/lexical/dictionary/reset", { method: "DELETE" });
      alert("Cache deleted");
      checkStatus();
      setSearchResult(null);
      setUpdateResult(null);
    } catch (error) {
      alert("Error resetting: " + error.message);
    } finally {
      setLoading({ ...loading, reset: false });
    }
  };

  return (
    <div className="mt-20 p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Lexical Dictionary Debug UI</h1>

      {/* Status Section */}
      <div className="border border-gray-300 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-semibold mb-3">📊 Dictionary Status</h2>
        <button
          onClick={checkStatus}
          disabled={loading.status}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
        >
          {loading.status ? "Checking..." : "Refresh Status"}
        </button>

        {status && (
          <div className="mt-4 space-y-2">
            <p>
              <strong>Cached:</strong> {status.cached ? "✅ Yes" : "❌ No"}
            </p>
            <p>
              <strong>Loaded:</strong> {status.loaded ? "✅ Yes" : "❌ No"}
            </p>
            {status.metadata && (
              <>
                <p>
                  <strong>Total Words:</strong> {status.metadata.total_words}
                </p>
                <p>
                  <strong>Created:</strong> {status.metadata.created_at}
                </p>
                <p>
                  <strong>Last Updated:</strong> {status.metadata.last_updated}
                </p>
                {status.metadata.manual_updates && (
                  <p>
                    <strong>Manual Edits:</strong>{" "}
                    {status.metadata.manual_updates.length}
                  </p>
                )}
              </>
            )}

            {status.cached && !status.loaded && (
              <button
                onClick={loadDictionary}
                disabled={loading.load}
                className="mt-3 bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:bg-gray-400"
              >
                {loading.load ? "Loading..." : "Load Dictionary"}
              </button>
            )}
          </div>
        )}
      </div>

      {/* Upload & Process Section */}
      <div className="border border-gray-300 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-semibold mb-3">
          📤 Upload & Process Dictionary
        </h2>

        <div className="flex gap-2 mb-3">
          <input
            type="file"
            accept=".xlsx,.xls"
            onChange={(e) => setUploadFile(e.target.files[0])}
            className="border border-gray-300 rounded px-2 py-1"
          />
          <button
            onClick={uploadDictionary}
            disabled={loading.upload || !uploadFile}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
          >
            {loading.upload ? "Uploading..." : "Upload Dictionary"}
          </button>
        </div>

        {uploadResult && (
          <div className="bg-gray-100 p-3 rounded mb-3">
            <p>
              <strong>Uploaded:</strong> {uploadResult.file_path}
            </p>
            <p>
              <strong>Rows:</strong> {uploadResult.rows}
            </p>
            <p>
              <strong>Columns:</strong> {uploadResult.columns.join(", ")}
            </p>
          </div>
        )}

        <button
          onClick={processDictionary}
          disabled={loading.process || !uploadResult}
          className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600 disabled:bg-gray-400"
        >
          {loading.process ? "Processing..." : "Process Dictionary"}
        </button>

        {processingStatus && (
          <div className="mt-3 bg-blue-50 p-3 rounded">
            <p>
              <strong>Status:</strong> {processingStatus.status}
            </p>
            <p>
              <strong>Progress:</strong> {processingStatus.progress}%
            </p>
            <p>
              <strong>Message:</strong> {processingStatus.message}
            </p>
          </div>
        )}

        {processingStatus?.status === "completed" && (
          <button
            onClick={saveToCache}
            disabled={loading.save}
            className="mt-3 bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:bg-gray-400"
          >
            {loading.save ? "Saving..." : "💾 Save to Cache"}
          </button>
        )}
      </div>

      {/* Search Section */}
      <div className="border border-gray-300 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-semibold mb-3">🔍 Search Word</h2>

        <div className="flex gap-2">
          <input
            type="text"
            value={searchWord}
            onChange={(e) => setSearchWord(e.target.value)}
            placeholder="Enter word (e.g., bagyo)"
            className="border border-gray-300 rounded px-3 py-2 w-64"
            onKeyPress={(e) => e.key === "Enter" && searchForWord()}
          />
          <button
            onClick={searchForWord}
            disabled={loading.search}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
          >
            {loading.search ? "Searching..." : "Search"}
          </button>
        </div>

        {searchResult && (
          <div
            className={`mt-4 p-4 rounded ${
              searchResult.found ? "bg-green-50" : "bg-red-50"
            }`}
          >
            {searchResult.found ? (
              <>
                <h3 className="text-lg font-semibold mb-2">
                  ✅ Found: {searchResult.word}
                </h3>
                <div className="space-y-1">
                  <p>
                    <strong>Score:</strong> {searchResult.sentiment_score}
                  </p>
                  <p>
                    <strong>Label:</strong> {searchResult.sentiment_label}
                  </p>
                  <p>
                    <strong>Polarity:</strong> {searchResult.polarity}
                  </p>
                  <p>
                    <strong>Intensity:</strong> {searchResult.intensity}
                  </p>
                  <p>
                    <strong>Climate:</strong>{" "}
                    {searchResult.is_climate ? "Yes" : "No"}
                  </p>
                  <p>
                    <strong>Dialect:</strong> {searchResult.dialect}
                  </p>
                  <p>
                    <strong>Definition:</strong> {searchResult.definition}
                  </p>
                  <p className="italic text-gray-700">
                    {searchResult.interpretation}
                  </p>
                </div>

                {searchResult.detailed_breakdown && (
                  <details className="mt-3">
                    <summary className="cursor-pointer font-semibold">
                      View Detailed Breakdown
                    </summary>
                    <pre className="bg-gray-100 p-3 rounded mt-2 overflow-auto text-xs">
                      {JSON.stringify(searchResult.detailed_breakdown, null, 2)}
                    </pre>
                  </details>
                )}
              </>
            ) : (
              <>
                <h3 className="text-lg font-semibold mb-2">
                  ❌ Not Found: {searchResult.word}
                </h3>
                <p>{searchResult.message}</p>
                {searchResult.suggestions &&
                  searchResult.suggestions.length > 0 && (
                    <p className="mt-2">
                      <strong>Suggestions:</strong>{" "}
                      {searchResult.suggestions.join(", ")}
                    </p>
                  )}
              </>
            )}
          </div>
        )}
      </div>

      {/* Edit Section */}
      <div className="border border-gray-300 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-semibold mb-3">✏️ Update Word Label</h2>

        <div className="flex gap-2 mb-3">
          <input
            type="text"
            value={editWord}
            onChange={(e) => setEditWord(e.target.value)}
            placeholder="Word to update"
            className="border border-gray-300 rounded px-3 py-2 w-64"
          />

          <select
            value={editLabel}
            onChange={(e) => setEditLabel(e.target.value)}
            className="border border-gray-300 rounded px-3 py-2"
          >
            <option value="positive">Positive</option>
            <option value="negative">Negative</option>
            <option value="neutral">Neutral</option>
          </select>

          <button
            onClick={updateWordLabel}
            disabled={loading.update}
            className="bg-orange-500 text-white px-4 py-2 rounded hover:bg-orange-600 disabled:bg-gray-400"
          >
            {loading.update ? "Updating..." : "Update Label"}
          </button>
        </div>

        {updateResult && (
          <div className="bg-orange-50 p-4 rounded">
            <h3 className="text-lg font-semibold mb-2">
              ✅ Updated: {updateResult.word}
            </h3>
            <p>
              <strong>Label:</strong> {updateResult.old_label} →{" "}
              {updateResult.new_label}
            </p>
            <p>
              <strong>Score:</strong> {updateResult.old_score} →{" "}
              {updateResult.new_score}
            </p>
            <p className="italic text-gray-700 mt-2">{updateResult.message}</p>

            {updateResult.breakdown && (
              <details className="mt-3">
                <summary className="cursor-pointer font-semibold">
                  View Calculation Breakdown
                </summary>
                <pre className="bg-gray-100 p-3 rounded mt-2 overflow-auto text-xs">
                  {JSON.stringify(updateResult.breakdown, null, 2)}
                </pre>
              </details>
            )}
          </div>
        )}
      </div>

      {/* Danger Zone */}
      <div className="border-2 border-red-500 rounded-lg p-4 mb-6">
        <h2 className="text-xl font-semibold mb-3">⚠️ Danger Zone</h2>
        <button
          onClick={resetCache}
          disabled={loading.reset}
          className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 disabled:bg-gray-400"
        >
          {loading.reset ? "Resetting..." : "Reset Cache (Delete Dictionary)"}
        </button>
        <p className="text-gray-600 text-sm mt-2">
          This will delete the cached dictionary. You'll need to reprocess.
        </p>
      </div>

      {/* Quick Actions */}
      <div className="bg-gray-100 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-3">📝 Quick Test Workflow</h3>
        <ol className="list-decimal list-inside space-y-1">
          <li>Check status (should see if cache exists)</li>
          <li>If no cache: Upload dictionary → Process → Save to cache</li>
          <li>If cache exists: Load dictionary</li>
          <li>Search for a word (e.g., "bagyo", "init")</li>
          <li>Update word label and see score recalculate</li>
          <li>Search again to verify the change</li>
        </ol>
      </div>
    </div>
  );
}
