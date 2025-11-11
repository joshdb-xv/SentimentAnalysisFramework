"use client";

import {
  MdChevronRight,
  MdUpload,
  MdPlayArrow,
  MdDownload,
  MdRefresh,
  MdCheckCircle,
  MdError,
  MdSearch,
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

  // Word search states
  const [searchWord, setSearchWord] = useState("");
  const [searchResult, setSearchResult] = useState(null);
  const [searching, setSearching] = useState(false);
  const [showSearchModal, setShowSearchModal] = useState(false);

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
      setSearchResult(null);
      setSearchWord("");
    } catch (error) {
      console.error("Error resetting:", error);
    }
  };

  const handleSearchWord = async (e) => {
    e.preventDefault();

    if (!searchWord.trim()) {
      alert("Please enter a word to search");
      return;
    }

    setSearching(true);
    setSearchResult(null);

    try {
      const response = await fetch("http://localhost:8000/lexical/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word: searchWord.trim() }),
      });

      const data = await response.json();
      setSearchResult(data);
      setShowSearchModal(true);
    } catch (error) {
      console.error("Error searching word:", error);
      alert(
        "Failed to search word. Make sure you have processed a dictionary first."
      );
    } finally {
      setSearching(false);
    }
  };

  const getStat = (statName) => {
    const value = results?.stats?.[statName];
    if (value === null || value === undefined || isNaN(value)) {
      return 0;
    }
    return Number(value);
  };

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
    <div className="min-h-screen bg-bluish-white relative">
      {/* Header */}
      <div className="bg-white px-6 w-full h-20 flex items-center justify-between border-b-2 border-primary-dark/5">
        <div className="flex items-center gap-4">
          <div className="w-fit text-2xl font-bold bg-gradient-to-r from-black via-primary-dark to-primary bg-clip-text text-transparent leading-snug">
            Sentiment Analysis Framework
          </div>
          <MdChevronRight className="text-3xl text-gray-light" />
          <p className="text-gray-mid text-2xl font-medium">
            Lexical Dictionary
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6 max-w-7xl mx-auto min-h-[calc(100vh-5rem)] flex items-center justify-center">
        <div className="w-full space-y-6">
          {/* Word Search Section - Always visible after processing */}
          {results && (
            <div className="bg-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl">
              <div className="flex flex-row items-center gap-4 mb-6">
                <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
                <p className="font-medium text-xl text-[#1E293B]">
                  SEARCH WORD
                </p>
              </div>

              <form onSubmit={handleSearchWord} className="flex gap-3">
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={searchWord}
                    onChange={(e) => setSearchWord(e.target.value)}
                    placeholder="Enter a word to search..."
                    className="w-full px-4 py-3 border-2 border-bluish-gray rounded-xl focus:border-primary focus:outline-none transition"
                  />
                  <MdSearch className="absolute right-4 top-1/2 -translate-y-1/2 text-2xl text-gray-light" />
                </div>
                <button
                  type="submit"
                  disabled={searching || !searchWord.trim()}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-primary to-primary-dark text-white rounded-xl hover:from-primary-dark hover:to-black disabled:from-gray disabled:to-gray-light disabled:cursor-not-allowed transition shadow-sm hover:shadow-md"
                >
                  <MdSearch className="text-xl" />
                  {searching ? "Searching..." : "Search"}
                </button>
              </form>

              <p className="text-sm text-gray-mid mt-3">
                💡 Search for any word in the processed lexical dictionary to
                see its sentiment score and breakdown.
              </p>
            </div>
          )}

          {/* Informative Section - Shows when no files uploaded */}
          {!dictionaryInfo && !results && (
            <div className="bg-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl">
              <div className="flex flex-row items-center gap-4 mb-6">
                <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
                <p className="font-medium text-xl text-[#1E293B]">
                  HOW IT WORKS
                </p>
              </div>

              <div className="space-y-6">
                <p className="text-gray-mid">
                  This tool processes Filipino/Cebuano dictionaries to generate
                  VADER-compatible sentiment lexicons using FastText embeddings.
                </p>

                <div className="bg-bluish-white p-6 rounded-xl border-l-4 border-primary">
                  <h3 className="font-semibold text-primary-dark mb-4">
                    Sample Processing: Word "init"
                  </h3>

                  <div className="space-y-4 text-sm">
                    {/* Step 1 */}
                    <div className="flex gap-4">
                      <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-bold">
                        1
                      </div>
                      <div className="flex-1">
                        <p className="font-semibold text-primary-dark mb-1">
                          Word Cleaning
                        </p>
                        <p className="text-gray-mid">
                          The word is converted to lowercase and hyphens are
                          removed for simpler form.
                        </p>
                        <div className="mt-2 p-3 bg-white rounded-lg border border-bluish-gray font-mono text-xs">
                          <span className="text-gray">"Init"</span> →{" "}
                          <span className="text-primary font-semibold">
                            "init"
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Step 2 */}
                    <div className="flex gap-4">
                      <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-bold">
                        2
                      </div>
                      <div className="flex-1">
                        <p className="font-semibold text-primary-dark mb-1">
                          Climate Keyword Detection
                        </p>
                        <p className="text-gray-mid">
                          The definition is checked against climate keywords
                          (weather, temperature, climate, etc.) to determine if
                          it's climate-related.
                        </p>
                        <div className="mt-2 p-3 bg-white rounded-lg border border-bluish-gray">
                          <p className="text-xs text-gray-mid mb-1">
                            Definition:
                          </p>
                          <p className="font-mono text-xs">
                            "Hot weather condition"
                          </p>
                          <p className="text-xs text-primary font-semibold mt-2">
                            ✓ Climate-related: Contains "weather"
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Step 3 */}
                    <div className="flex gap-4">
                      <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-bold">
                        3
                      </div>
                      <div className="flex-1">
                        <p className="font-semibold text-primary-dark mb-1">
                          Base Sentiment Score
                        </p>
                        <p className="text-gray-mid">
                          Assign base score from sentiment label.
                          Climate-related words get stronger scores.
                        </p>
                        <div className="mt-2 p-3 bg-white rounded-lg border border-bluish-gray">
                          <div className="grid grid-cols-2 gap-3 text-xs">
                            <div>
                              <p className="text-gray-mid mb-1">Non-climate:</p>
                              <p className="font-mono">
                                Positive:{" "}
                                <span className="text-blue font-semibold">
                                  +2.5
                                </span>
                              </p>
                              <p className="font-mono">
                                Negative:{" "}
                                <span className="text-red font-semibold">
                                  -2.5
                                </span>
                              </p>
                            </div>
                            <div>
                              <p className="text-gray-mid mb-1">
                                Climate-related:
                              </p>
                              <p className="font-mono">
                                Positive:{" "}
                                <span className="text-blue font-semibold">
                                  +3.25
                                </span>
                              </p>
                              <p className="font-mono">
                                Negative:{" "}
                                <span className="text-red font-semibold">
                                  -3.25
                                </span>
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Step 4 */}
                    <div className="flex gap-4">
                      <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-bold">
                        4
                      </div>
                      <div className="flex-1">
                        <p className="font-semibold text-primary-dark mb-1">
                          FastText Semantic Intensity
                        </p>
                        <p className="text-gray-mid">
                          Calculate semantic similarity to anchor words using
                          FastText embeddings to adjust intensity (0.6x to
                          1.4x).
                        </p>
                        <div className="mt-2 p-3 bg-white rounded-lg border border-bluish-gray">
                          <div className="text-xs space-y-2">
                            <p className="text-gray-mid">
                              Negative anchors: dili, dautan, grabe, lisud,
                              makuyaw
                            </p>
                            <p className="font-mono">
                              Avg similarity:{" "}
                              <span className="font-semibold">0.65</span>
                            </p>
                            <p className="font-mono">
                              Intensity multiplier:{" "}
                              <span className="font-semibold text-primary">
                                0.6 + (0.65 × 0.8) = 1.12
                              </span>
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Step 5 */}
                    <div className="flex gap-4">
                      <div className="flex-shrink-0 w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center font-bold">
                        5
                      </div>
                      <div className="flex-1">
                        <p className="font-semibold text-primary-dark mb-1">
                          Final Score Calculation
                        </p>
                        <p className="text-gray-mid">
                          Multiply base score by intensity multiplier for the
                          final sentiment score.
                        </p>
                        <div className="mt-2 p-3 bg-gradient-to-r from-primary/10 to-primary/5 rounded-lg border-2 border-primary">
                          <p className="font-mono text-sm">
                            <span className="text-gray-mid">
                              Base score × Intensity =
                            </span>
                            <span className="font-bold text-primary ml-2">
                              -3.25 × 1.12 = -3.64
                            </span>
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-blue/10 p-4 rounded-xl border border-blue">
                  <p className="text-sm text-blue flex items-start gap-2">
                    <span className="text-xl">💡</span>
                    <span>
                      <strong>Duplicate Handling:</strong> If multiple
                      definitions exist for the same word, the system
                      prioritizes climate-related definitions and keeps simpler
                      word forms (without hyphens).
                    </span>
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Upload Section - Hidden when results are shown */}
          {!results && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Dictionary Upload */}
              <div className="bg-bluish-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl flex flex-col min-h-[450px]">
                <div className="flex flex-row items-center gap-4">
                  <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
                  <p className="font-medium text-xl text-[#1E293B]">
                    UPLOAD DICTIONARY
                  </p>
                </div>

                <div className="flex-1 flex flex-col justify-center py-6">
                  <label className="relative flex flex-col items-center justify-center border-2 border-dashed border-gray-light rounded-xl p-8 cursor-pointer hover:border-primary hover:bg-primary/5 transition group">
                    <MdUpload className="text-5xl text-gray-light group-hover:text-primary mb-2 transition" />
                    <span className="text-sm text-gray-mid text-center">
                      {uploadingDict ? (
                        <span className="text-primary font-medium">
                          Uploading...
                        </span>
                      ) : dictionaryFile ? (
                        <span className="text-primary-dark">
                          {dictionaryFile.name}
                        </span>
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
                          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                          <span className="text-sm text-primary font-medium">
                            Loading...
                          </span>
                        </div>
                      </div>
                    )}
                  </label>

                  {dictionaryInfo && (
                    <div className="mt-4 p-4 bg-primary/10 border border-primary rounded-xl text-sm">
                      <div className="flex items-start gap-2">
                        <MdCheckCircle className="text-primary text-xl flex-shrink-0 mt-0.5" />
                        <div className="flex-1">
                          <p className="font-semibold text-primary">
                            Uploaded successfully
                          </p>
                          <p className="text-primary-dark mt-1">
                            Rows:{" "}
                            <span className="font-mono font-semibold">
                              {dictionaryInfo.rows.toLocaleString()}
                            </span>
                          </p>
                          <p className="text-primary-dark">
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
              </div>

              {/* Keywords Load/Upload */}
              <div className="bg-bluish-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl flex flex-col min-h-[450px]">
                <div className="flex flex-row items-center gap-4">
                  <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
                  <p className="font-medium text-xl text-[#1E293B]">
                    CLIMATE KEYWORDS
                  </p>
                </div>

                <div className="flex-1 flex flex-col justify-center py-6">
                  {/* Load from default location */}
                  <button
                    onClick={handleKeywordsLoad}
                    disabled={loadingKeywords}
                    className="relative w-full mb-4 flex items-center justify-center gap-2 border-2 border-primary bg-primary/10 rounded-xl p-4 hover:bg-primary/20 transition disabled:opacity-50 disabled:cursor-not-allowed group"
                  >
                    <MdUpload className="text-2xl text-primary" />
                    <span className="text-primary font-medium">
                      {loadingKeywords
                        ? "Loading..."
                        : "Load from data/weatherkeywords.csv"}
                    </span>

                    {loadingKeywords && (
                      <div className="absolute inset-0 bg-primary/10 rounded-xl flex items-center justify-center">
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
                      </div>
                    )}
                  </button>

                  {/* Or upload custom file */}
                  <div className="relative my-4">
                    <div className="absolute inset-0 flex items-center">
                      <div className="w-full border-t border-bluish-gray"></div>
                    </div>
                    <div className="relative flex justify-center text-xs">
                      <span className="px-2 bg-white text-gray">
                        Or upload custom file
                      </span>
                    </div>
                  </div>

                  <label className="relative flex flex-col items-center justify-center border-2 border-dashed border-gray-light rounded-xl p-6 cursor-pointer hover:border-primary hover:bg-primary/5 transition group">
                    <MdUpload className="text-4xl text-gray-light group-hover:text-primary mb-2 transition" />
                    <span className="text-sm text-gray-mid">
                      {uploadingKeywords ? (
                        <span className="text-primary font-medium">
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
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
                      </div>
                    )}
                  </label>

                  {keywordsInfo && (
                    <div className="mt-4 p-4 bg-primary/10 border border-primary rounded-xl text-sm">
                      <div className="flex items-start gap-2">
                        <MdCheckCircle className="text-primary text-xl flex-shrink-0 mt-0.5" />
                        <div className="flex-1">
                          <p className="font-semibold text-primary">
                            Keywords loaded successfully
                          </p>
                          <p className="text-primary-dark mt-1">
                            Keywords:{" "}
                            <span className="font-mono font-semibold">
                              {keywordsInfo.keyword_count}
                            </span>
                          </p>
                          <p className="text-xs text-gray-mid mt-1 font-mono truncate">
                            {keywordsInfo.file_path}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Processing Controls - Hidden when results are shown */}
          {!results && (
            <div className="bg-bluish-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl flex flex-col min-h-[280px]">
              <div className="flex flex-row items-center gap-4">
                <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
                <p className="font-medium text-xl text-[#1E293B]">
                  PROCESS DICTIONARY
                </p>
              </div>

              <div className="flex-1 flex flex-col justify-center py-6">
                {/* Info about FastText */}
                <div className="mb-4 p-4 bg-primary/10 border border-primary rounded-xl">
                  <p className="text-sm text-primary flex items-start gap-2">
                    <span className="text-primary text-lg">⚡</span>
                    <span>
                      <strong>
                        FastText embeddings are automatically used
                      </strong>{" "}
                      for semantic scoring. Models are pre-loaded at server
                      startup for fast processing.
                    </span>
                  </p>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={startProcessing}
                    disabled={!dictionaryInfo || processing}
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-primary to-primary-dark text-white rounded-xl hover:from-primary-dark hover:to-black disabled:from-gray disabled:to-gray-light disabled:cursor-not-allowed transition shadow-sm hover:shadow-md"
                  >
                    <MdPlayArrow className="text-xl" />
                    {processing ? "Processing..." : "Start Processing"}
                  </button>

                  <button
                    onClick={resetAll}
                    className="flex items-center gap-2 px-6 py-3 bg-bluish-gray text-gray-dark rounded-xl hover:bg-gray-light hover:text-white transition"
                  >
                    <MdRefresh className="text-xl" />
                    Reset
                  </button>
                </div>

                {!keywordsInfo && !processing && dictionaryInfo && (
                  <p className="text-sm text-gray-mid mt-3 flex items-center gap-2">
                    <span>💡</span>
                    <span>
                      No keywords loaded. Will use default from{" "}
                      <code className="bg-bluish-gray px-1.5 py-0.5 rounded text-xs font-mono">
                        data/weatherkeywords.csv
                      </code>
                    </span>
                  </p>
                )}

                {/* Progress */}
                {status && (
                  <div className="mt-4">
                    <div className="flex justify-between text-sm text-primary-dark mb-2">
                      <span className="font-medium">{status.message}</span>
                      <span className="font-semibold text-primary">
                        {status.progress}%
                      </span>
                    </div>
                    <div className="w-full bg-bluish-gray rounded-full h-3 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-primary to-primary-dark h-3 rounded-full transition-all duration-300 ease-out"
                        style={{ width: `${status.progress}%` }}
                      />
                    </div>

                    {status.status === "error" && (
                      <div className="mt-3 p-3 bg-red/10 border border-red rounded-xl flex items-start gap-2">
                        <MdError className="text-red text-xl flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="text-sm font-semibold text-red">
                            Error occurred
                          </p>
                          <p className="text-sm text-red mt-1">
                            {status.message}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Results Section */}
          {results && results.stats && (
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
                      <span className="text-gray-mid text-xs mb-1">
                        Climate Words
                      </span>
                      <span className="font-mono text-lg font-bold text-black">
                        {(getStat("avg_score_climate") || 0).toFixed(3)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-gray-mid text-xs mb-1">
                        General Words
                      </span>
                      <span className="font-mono text-lg font-bold text-black">
                        {(getStat("avg_score_general") || 0).toFixed(3)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-gray-mid text-xs mb-1">
                        Non-Climate
                      </span>
                      <span className="font-mono text-lg font-bold text-black">
                        {getStat("non_climate").toLocaleString()}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-gray-mid text-xs mb-1">
                        Neutral
                      </span>
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
                          <tr
                            key={idx}
                            className="hover:bg-white/50 transition"
                          >
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
          )}
        </div>
      </div>

      {/* Search Result Modal */}
      {showSearchModal && searchResult && (
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
                    <p className="text-gray-mid">
                      {searchResult.interpretation}
                    </p>
                  </div>

                  {/* NEW: Three-Stage Calculation Breakdown */}
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
                              searchResult.detailed_breakdown.stages
                                .stage_1_base.explanation
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

                          {/* Embedding Details */}
                          {searchResult.detailed_breakdown.stages
                            .stage_2_intensity.details && (
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
                      {searchResult.detailed_breakdown.stages
                        ?.stage_3_domain && (
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
                              searchResult.detailed_breakdown.stages
                                .stage_3_domain.explanation
                            }
                          </p>
                          <div className="bg-white p-3 rounded-lg border border-green-500/20">
                            <div className="flex items-center justify-between">
                              <span className="text-gray-mid">
                                Domain Weight:
                              </span>
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
                              <span className="text-gray-mid">
                                Final Score:
                              </span>
                              <span
                                className={`ml-3 font-mono font-bold text-2xl ${getScoreColor(
                                  searchResult.detailed_breakdown.calculation
                                    .result
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
                            <span className="text-gray-mid">
                              Climate-Related:
                            </span>
                            <span className="ml-2 font-semibold text-black">
                              {searchResult.detailed_breakdown
                                .is_climate_related
                                ? "Yes"
                                : "No"}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Basic Breakdown (fallback if no detailed breakdown) */}
                  {!searchResult.detailed_breakdown &&
                    searchResult.breakdown && (
                      <div className="bg-white p-5 rounded-xl border border-bluish-gray">
                        <p className="text-sm font-semibold text-primary-dark mb-3">
                          BASIC BREAKDOWN
                        </p>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div className="flex flex-col">
                            <span className="text-gray-mid text-xs mb-1">
                              Polarity
                            </span>
                            <span className="font-semibold text-black capitalize">
                              {searchResult.breakdown.polarity}
                            </span>
                          </div>
                          <div className="flex flex-col">
                            <span className="text-gray-mid text-xs mb-1">
                              Strength
                            </span>
                            <span className="font-mono font-semibold text-black">
                              {searchResult.breakdown.strength?.toFixed(3) ||
                                "N/A"}
                            </span>
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
      )}
    </div>
  );
}
