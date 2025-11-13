"use client";

import { useState, useEffect } from "react";

export default function LexicalDebug() {
  const [activeTab, setActiveTab] = useState("status");
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
  const [message, setMessage] = useState({ text: "", type: "" });

  const API_BASE = "http://localhost:8000";

  useEffect(() => {
    checkStatus();
  }, []);

  const showMessage = (text, type = "success") => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: "", type: "" }), 5000);
  };

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

  const checkStatus = async () => {
    setLoading({ ...loading, status: true });
    try {
      const data = await apiCall("/lexical/dictionary/status");
      setStatus(data);
    } catch (error) {
      showMessage("Error checking status: " + error.message, "error");
    } finally {
      setLoading({ ...loading, status: false });
    }
  };

  const loadDictionary = async () => {
    setLoading({ ...loading, load: true });
    try {
      const data = await apiCall("/lexical/dictionary/load", {
        method: "POST",
      });
      showMessage(`Dictionary loaded: ${data.total_words} words`);
      checkStatus();
    } catch (error) {
      showMessage("Error loading: " + error.message, "error");
    } finally {
      setLoading({ ...loading, load: false });
    }
  };

  const uploadDictionary = async () => {
    if (!uploadFile) {
      showMessage("Please select a file", "error");
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
      showMessage(`File uploaded: ${data.rows} rows`);
    } catch (error) {
      showMessage("Error uploading: " + error.message, "error");
    } finally {
      setLoading({ ...loading, upload: false });
    }
  };

  const processDictionary = async () => {
    if (!uploadResult || !uploadResult.file_path) {
      showMessage("Please upload a dictionary first", "error");
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
      showMessage("Processing started!");

      const interval = setInterval(async () => {
        const statusData = await apiCall("/lexical/status");
        setProcessingStatus(statusData);

        if (statusData.status === "completed") {
          clearInterval(interval);
          showMessage("Processing completed!");
        } else if (statusData.status === "error") {
          clearInterval(interval);
          showMessage("Processing failed: " + statusData.message, "error");
        }
      }, 2000);
    } catch (error) {
      showMessage("Error processing: " + error.message, "error");
    } finally {
      setLoading({ ...loading, process: false });
    }
  };

  const saveToCache = async () => {
    setLoading({ ...loading, save: true });
    try {
      const data = await apiCall("/lexical/dictionary/save", {
        method: "POST",
      });
      showMessage(`Saved to cache: ${data.path}`);
      checkStatus();
    } catch (error) {
      showMessage("Error saving: " + error.message, "error");
    } finally {
      setLoading({ ...loading, save: false });
    }
  };

  const searchForWord = async () => {
    if (!searchWord.trim()) {
      showMessage("Please enter a word", "error");
      return;
    }
    setLoading({ ...loading, search: true });
    try {
      const data = await apiCall("/lexical/search", {
        method: "POST",
        body: JSON.stringify({ word: searchWord }),
      });
      setSearchResult(data);

      if (data.found) {
        setEditWord(data.word);
        setEditLabel(data.sentiment_label || "neutral");
      }
    } catch (error) {
      showMessage("Error searching: " + error.message, "error");
    } finally {
      setLoading({ ...loading, search: false });
    }
  };

  const updateWordLabel = async () => {
    if (!editWord.trim()) {
      showMessage("Please enter a word", "error");
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
      showMessage(`Updated! ${data.old_score} → ${data.new_score}`);

      if (searchWord === editWord) {
        searchForWord();
      }
    } catch (error) {
      showMessage("Error updating: " + error.message, "error");
    } finally {
      setLoading({ ...loading, update: false });
    }
  };

  const resetCache = async () => {
    if (!confirm("Are you sure? This will delete the cached dictionary.")) {
      return;
    }
    setLoading({ ...loading, reset: true });
    try {
      await apiCall("/lexical/dictionary/reset", { method: "DELETE" });
      showMessage("Cache deleted");
      checkStatus();
      setSearchResult(null);
      setUpdateResult(null);
    } catch (error) {
      showMessage("Error resetting: " + error.message, "error");
    } finally {
      setLoading({ ...loading, reset: false });
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8 mt-20">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          {/* Header */}
          <div className="bg-gray-800 p-6">
            <h1 className="text-3xl font-bold text-white mb-2">
              Lexical Dictionary Management
            </h1>
            <p className="text-gray-300">
              Upload, Process, Search, and Update Dictionary Entries
            </p>

            {/* Status Bar */}
            {status && (
              <div className="mt-4 grid grid-cols-3 gap-3">
                <div className="bg-gray-700 rounded p-3">
                  <div className="text-gray-300 text-xs mb-1">Cached</div>
                  <div className="text-white font-semibold text-sm">
                    {status.cached ? "Yes" : "No"}
                  </div>
                </div>
                <div className="bg-gray-700 rounded p-3">
                  <div className="text-gray-300 text-xs mb-1">Loaded</div>
                  <div className="text-white font-semibold text-sm">
                    {status.loaded ? "Yes" : "No"}
                  </div>
                </div>
                <div className="bg-gray-700 rounded p-3">
                  <div className="text-gray-300 text-xs mb-1">Total Words</div>
                  <div className="text-white font-semibold text-lg">
                    {status.metadata?.total_words || 0}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Message Bar */}
          {message.text && (
            <div
              className={`p-4 ${
                message.type === "error"
                  ? "bg-red-50 text-red-800 border-l-4 border-red-500"
                  : "bg-green-50 text-green-800 border-l-4 border-green-500"
              }`}
            >
              {message.text}
            </div>
          )}

          {/* Loading Overlay */}
          {Object.values(loading).some((l) => l) && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white p-6 rounded-lg shadow-xl">
                <div className="animate-spin h-12 w-12 border-4 border-gray-300 border-t-gray-800 rounded-full mx-auto mb-4"></div>
                <p className="text-lg font-semibold text-gray-800">
                  Processing...
                </p>
              </div>
            </div>
          )}

          {/* Tabs */}
          <div className="border-b border-gray-200 bg-gray-50">
            <div className="flex overflow-x-auto">
              {["status", "upload", "search", "edit"].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-6 py-3 font-medium capitalize whitespace-nowrap transition-colors ${
                    activeTab === tab
                      ? "bg-white border-b-2 border-gray-800 text-gray-900"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>

          {/* Content */}
          <div className="p-6">
            {/* Status Tab */}
            {activeTab === "status" && (
              <div className="space-y-6">
                <div className="flex justify-between items-center">
                  <h2 className="text-2xl font-bold text-gray-900">
                    Dictionary Status
                  </h2>
                  <button
                    onClick={checkStatus}
                    disabled={loading.status}
                    className="px-4 py-2 bg-gray-800 text-white rounded font-medium hover:bg-gray-700 disabled:bg-gray-300 transition-colors"
                  >
                    {loading.status ? "Checking..." : "Refresh Status"}
                  </button>
                </div>

                {status && (
                  <div className="grid grid-cols-2 gap-4">
                    <div className="border rounded-lg p-4">
                      <h3 className="font-semibold text-lg mb-3 text-gray-900">
                        Cache Status
                      </h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Cached:</span>
                          <span className="font-medium">
                            {status.cached ? "Yes" : "No"}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Loaded:</span>
                          <span className="font-medium">
                            {status.loaded ? "Yes" : "No"}
                          </span>
                        </div>
                      </div>

                      {status.cached && !status.loaded && (
                        <button
                          onClick={loadDictionary}
                          disabled={loading.load}
                          className="mt-3 w-full px-4 py-2 bg-gray-800 text-white rounded font-medium hover:bg-gray-700 disabled:bg-gray-300 transition-colors"
                        >
                          {loading.load ? "Loading..." : "Load Dictionary"}
                        </button>
                      )}
                    </div>

                    <div className="border rounded-lg p-4">
                      <h3 className="font-semibold text-lg mb-3 text-gray-900">
                        Metadata
                      </h3>
                      {status.metadata ? (
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Total Words:</span>
                            <span className="font-medium">
                              {status.metadata.total_words}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Created:</span>
                            <span className="font-medium text-xs">
                              {new Date(
                                status.metadata.created_at
                              ).toLocaleString()}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Last Updated:</span>
                            <span className="font-medium text-xs">
                              {new Date(
                                status.metadata.last_updated
                              ).toLocaleString()}
                            </span>
                          </div>
                          {status.metadata.manual_updates && (
                            <div className="flex justify-between">
                              <span className="text-gray-600">
                                Manual Edits:
                              </span>
                              <span className="font-medium">
                                {status.metadata.manual_updates.length}
                              </span>
                            </div>
                          )}
                        </div>
                      ) : (
                        <p className="text-gray-600 text-sm">
                          No metadata available
                        </p>
                      )}
                    </div>
                  </div>
                )}

                <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg">
                  <h3 className="font-semibold text-lg mb-3 text-gray-900">
                    Quick Workflow
                  </h3>
                  <ol className="space-y-2 text-gray-700 text-sm">
                    <li className="flex gap-2">
                      <span className="font-semibold">1.</span> Check status to
                      see if cache exists
                    </li>
                    <li className="flex gap-2">
                      <span className="font-semibold">2.</span> If no cache:
                      Upload dictionary → Process → Save to cache
                    </li>
                    <li className="flex gap-2">
                      <span className="font-semibold">3.</span> If cache
                      exists: Load dictionary
                    </li>
                    <li className="flex gap-2">
                      <span className="font-semibold">4.</span> Search for
                      words and update labels as needed
                    </li>
                  </ol>
                </div>

                <div className="border-2 border-red-500 rounded-lg p-4">
                  <h3 className="font-semibold text-lg mb-2 text-red-900">
                    Danger Zone
                  </h3>
                  <p className="text-gray-600 text-sm mb-3">
                    This will delete the cached dictionary. You'll need to
                    reprocess.
                  </p>
                  <button
                    onClick={resetCache}
                    disabled={loading.reset}
                    className="px-4 py-2 bg-red-600 text-white rounded font-medium hover:bg-red-700 disabled:bg-gray-300 transition-colors"
                  >
                    {loading.reset ? "Resetting..." : "Reset Cache"}
                  </button>
                </div>
              </div>
            )}

            {/* Upload Tab */}
            {activeTab === "upload" && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900">
                  Upload & Process Dictionary
                </h2>

                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 text-gray-900">
                    Step 1: Upload Dictionary File
                  </h3>
                  <p className="text-sm text-gray-600 mb-3">
                    Upload an Excel file (.xlsx or .xls) containing dictionary
                    entries
                  </p>

                  <div className="flex gap-3 items-end">
                    <div className="flex-1">
                      <label className="block text-sm font-medium mb-2 text-gray-700">
                        Select Excel File
                      </label>
                      <input
                        type="file"
                        accept=".xlsx,.xls"
                        onChange={(e) => setUploadFile(e.target.files[0])}
                        className="w-full p-2 border border-gray-300 rounded focus:border-gray-500 focus:ring-1 focus:ring-gray-500"
                      />
                    </div>
                    <button
                      onClick={uploadDictionary}
                      disabled={loading.upload || !uploadFile}
                      className="px-6 py-2 bg-gray-800 text-white rounded font-medium hover:bg-gray-700 disabled:bg-gray-300 transition-colors"
                    >
                      {loading.upload ? "Uploading..." : "Upload"}
                    </button>
                  </div>

                  {uploadResult && (
                    <div className="mt-3 bg-gray-50 border border-gray-200 rounded p-3">
                      <div className="font-medium text-gray-900 text-sm mb-2">
                        Upload Successful
                      </div>
                      <div className="text-sm text-gray-700 space-y-1">
                        <div>File: {uploadResult.file_path}</div>
                        <div>Rows: {uploadResult.rows}</div>
                        <div>Columns: {uploadResult.columns.join(", ")}</div>
                      </div>
                    </div>
                  )}
                </div>

                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 text-gray-900">
                    Step 2: Process Dictionary
                  </h3>
                  <p className="text-gray-700 text-sm mb-3">
                    Process the uploaded dictionary with FastText models to
                    generate sentiment scores
                  </p>

                  <button
                    onClick={processDictionary}
                    disabled={loading.process || !uploadResult}
                    className="w-full px-6 py-3 bg-gray-800 text-white rounded font-semibold hover:bg-gray-700 disabled:bg-gray-300 transition-colors"
                  >
                    {loading.process ? "Processing..." : "Process Dictionary"}
                  </button>

                  {!uploadResult && (
                    <p className="text-red-700 mt-2 text-sm">
                      Upload a dictionary first
                    </p>
                  )}

                  {processingStatus && (
                    <div className="mt-4 bg-gray-50 border border-gray-200 rounded p-3">
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Status:</span>
                          <span className="font-medium">
                            {processingStatus.status}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Progress:</span>
                          <span className="font-medium">
                            {processingStatus.progress}%
                          </span>
                        </div>
                        <div className="text-gray-700">
                          {processingStatus.message}
                        </div>
                      </div>
                      <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-gray-800 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${processingStatus.progress}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>

                {processingStatus?.status === "completed" && (
                  <div className="border rounded-lg p-4 bg-green-50">
                    <h3 className="text-lg font-semibold mb-3 text-green-900">
                      Step 3: Save to Cache
                    </h3>
                    <p className="text-gray-700 text-sm mb-3">
                      Processing complete! Save the processed dictionary to
                      cache for future use.
                    </p>
                    <button
                      onClick={saveToCache}
                      disabled={loading.save}
                      className="w-full px-6 py-3 bg-green-700 text-white rounded font-semibold hover:bg-green-800 disabled:bg-gray-300 transition-colors"
                    >
                      {loading.save ? "Saving..." : "Save to Cache"}
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Search Tab */}
            {activeTab === "search" && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900">
                  Search Word
                </h2>

                <div className="border rounded-lg p-4">
                  <label className="block text-sm font-medium mb-2 text-gray-700">
                    Enter word to search
                  </label>
                  <div className="flex gap-3">
                    <input
                      type="text"
                      value={searchWord}
                      onChange={(e) => setSearchWord(e.target.value)}
                      placeholder="e.g., bagyo, init, ulan"
                      className="flex-1 p-3 border border-gray-300 rounded focus:border-gray-500 focus:ring-1 focus:ring-gray-500"
                      onKeyPress={(e) => e.key === "Enter" && searchForWord()}
                    />
                    <button
                      onClick={searchForWord}
                      disabled={loading.search}
                      className="px-6 py-3 bg-gray-800 text-white rounded font-semibold hover:bg-gray-700 disabled:bg-gray-300 transition-colors"
                    >
                      {loading.search ? "Searching..." : "Search"}
                    </button>
                  </div>
                </div>

                {searchResult && (
                  <div
                    className={`border rounded-lg p-4 ${
                      searchResult.found ? "bg-green-50" : "bg-red-50"
                    }`}
                  >
                    {searchResult.found ? (
                      <>
                        <h3 className="text-xl font-semibold mb-4 text-gray-900">
                          Found: {searchResult.word}
                        </h3>

                        <div className="grid grid-cols-2 gap-4 mb-4">
                          <div className="bg-white rounded p-3 border">
                            <div className="text-xs text-gray-600 mb-1">
                              Sentiment Score
                            </div>
                            <div className="text-2xl font-bold text-gray-900">
                              {searchResult.sentiment_score}
                            </div>
                          </div>
                          <div className="bg-white rounded p-3 border">
                            <div className="text-xs text-gray-600 mb-1">
                              Label
                            </div>
                            <div className="text-2xl font-bold text-gray-900">
                              {searchResult.sentiment_label}
                            </div>
                          </div>
                        </div>

                        <div className="bg-white rounded p-3 border space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Polarity:</span>
                            <span className="font-medium">
                              {searchResult.polarity}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Intensity:</span>
                            <span className="font-medium">
                              {searchResult.intensity}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Climate:</span>
                            <span className="font-medium">
                              {searchResult.is_climate ? "Yes" : "No"}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Dialect:</span>
                            <span className="font-medium">
                              {searchResult.dialect}
                            </span>
                          </div>
                        </div>

                        <div className="mt-3 bg-white rounded p-3 border">
                          <div className="text-xs text-gray-600 mb-1">
                            Definition
                          </div>
                          <div className="text-sm text-gray-800">
                            {searchResult.definition}
                          </div>
                        </div>

                        <div className="mt-3 bg-white rounded p-3 border">
                          <div className="text-xs text-gray-600 mb-1">
                            Interpretation
                          </div>
                          <div className="text-sm text-gray-800 italic">
                            {searchResult.interpretation}
                          </div>
                        </div>

                        {searchResult.detailed_breakdown && (
                          <details className="mt-3">
                            <summary className="cursor-pointer font-medium text-sm text-gray-700 hover:text-gray-900">
                              View Detailed Breakdown
                            </summary>
                            <pre className="bg-white p-3 rounded mt-2 overflow-auto text-xs border">
                              {JSON.stringify(
                                searchResult.detailed_breakdown,
                                null,
                                2
                              )}
                            </pre>
                          </details>
                        )}
                      </>
                    ) : (
                      <>
                        <h3 className="text-xl font-semibold mb-2 text-gray-900">
                          Not Found: {searchResult.word}
                        </h3>
                        <p className="text-gray-700">{searchResult.message}</p>
                        {searchResult.suggestions &&
                          searchResult.suggestions.length > 0 && (
                            <div className="mt-3">
                              <div className="text-sm font-medium text-gray-700 mb-1">
                                Suggestions:
                              </div>
                              <div className="text-sm text-gray-800">
                                {searchResult.suggestions.join(", ")}
                              </div>
                            </div>
                          )}
                      </>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Edit Tab */}
            {activeTab === "edit" && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900">
                  Update Word Label
                </h2>

                <div className="border rounded-lg p-4">
                  <p className="text-sm text-gray-600 mb-4">
                    Update the sentiment label of a word. The score will be
                    automatically recalculated.
                  </p>

                  <div className="space-y-3">
                    <div>
                      <label className="block text-sm font-medium mb-2 text-gray-700">
                        Word to Update
                      </label>
                      <input
                        type="text"
                        value={editWord}
                        onChange={(e) => setEditWord(e.target.value)}
                        placeholder="Enter word"
                        className="w-full p-3 border border-gray-300 rounded focus:border-gray-500 focus:ring-1 focus:ring-gray-500"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-2 text-gray-700">
                        New Sentiment Label
                      </label>
                      <select
                        value={editLabel}
                        onChange={(e) => setEditLabel(e.target.value)}
                        className="w-full p-3 border border-gray-300 rounded focus:border-gray-500 focus:ring-1 focus:ring-gray-500"
                      >
                        <option value="positive">Positive</option>
                        <option value="negative">Negative</option>
                        <option value="neutral">Neutral</option>
                      </select>
                    </div>

                    <button
                      onClick={updateWordLabel}
                      disabled={loading.update}
                      className="w-full px-6 py-3 bg-gray-800 text-white rounded font-semibold hover:bg-gray-700 disabled:bg-gray-300 transition-colors"
                    >
                      {loading.update ? "Updating..." : "Update Label"}
                    </button>
                  </div>
                </div>

                {updateResult && (
                  <div className="border rounded-lg p-4 bg-green-50">
                    <h3 className="text-xl font-semibold mb-4 text-green-900">
                      Updated: {updateResult.word}
                    </h3>

                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="bg-white rounded p-3 border">
                        <div className="text-xs text-gray-600 mb-1">
                          Score Change
                        </div>
                        <div className="text-sm font-medium text-gray-900">
                          {updateResult.old_score} → {updateResult.new_score}
                        </div>
                      </div>
                    </div>

                    <div className="bg-white rounded p-3 border">
                      <div className="text-sm text-gray-700">
                        {updateResult.message}
                      </div>
                    </div>

                    {updateResult.breakdown && (
                      <details className="mt-3">
                        <summary className="cursor-pointer font-medium text-sm text-gray-700 hover:text-gray-900">
                          View Calculation Breakdown
                        </summary>
                        <pre className="bg-white p-3 rounded mt-2 overflow-auto text-xs border">
                          {JSON.stringify(updateResult.breakdown, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// text-xs text-gray-600 mb-1">
//                           Label Change
//                         </div>
//                         <div className="text-sm font-medium text-gray-900">
//                           {updateResult.old_label} → {updateResult.new_label}
//                         </div>
//                       </div>
//                       <div className="bg-white rounded p-3 border">
//                         <div className="