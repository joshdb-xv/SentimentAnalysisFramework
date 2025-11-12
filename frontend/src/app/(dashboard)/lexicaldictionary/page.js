"use client";

import { useState, useEffect } from "react";
import FileUploadSection from "@/components/lexicaldictionary/FileUploadSection";
import ProcessingControls from "@/components/lexicaldictionary/ProcessingControls";
import WordSearchSection from "@/components/lexicaldictionary/WordSearchSection";
import ResultsSection from "@/components/lexicaldictionary/ResultsSection";
import SearchResultModal from "@/components/lexicaldictionary/SearchResultModal";
import HowItWorksSection from "@/components/lexicaldictionary/HowItWorksSection";

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

  return (
    <div className="min-h-screen bg-[#F8FAFC] pt-20">
      <div className="p-6 mx-2 min-h-[calc(100vh-5rem)] flex items-center justify-center">
        <div className="w-full space-y-6">
          {/* Word Search Section - Always visible after processing */}
          {results && (
            <WordSearchSection
              searchWord={searchWord}
              setSearchWord={setSearchWord}
              handleSearchWord={handleSearchWord}
              searching={searching}
            />
          )}

          {/* Informative Section - Shows when no files uploaded */}
          {!dictionaryInfo && !results && <HowItWorksSection />}

          {/* Upload Section - Hidden when results are shown */}
          {!results && (
            <FileUploadSection
              dictionaryFile={dictionaryFile}
              keywordsFile={keywordsFile}
              dictionaryInfo={dictionaryInfo}
              keywordsInfo={keywordsInfo}
              uploadingDict={uploadingDict}
              uploadingKeywords={uploadingKeywords}
              loadingKeywords={loadingKeywords}
              handleDictionaryUpload={handleDictionaryUpload}
              handleKeywordsLoad={handleKeywordsLoad}
              handleKeywordsUpload={handleKeywordsUpload}
            />
          )}

          {/* Processing Controls - Hidden when results are shown */}
          {!results && (
            <ProcessingControls
              dictionaryInfo={dictionaryInfo}
              keywordsInfo={keywordsInfo}
              processing={processing}
              status={status}
              startProcessing={startProcessing}
              resetAll={resetAll}
            />
          )}

          {/* Results Section */}
          {results && results.stats && (
            <ResultsSection
              results={results}
              downloadCSV={downloadCSV}
              resetAll={resetAll}
              setShowJsonModal={setShowJsonModal}
            />
          )}
        </div>
      </div>

      {/* Search Result Modal */}
      <SearchResultModal
        searchResult={searchResult}
        showSearchModal={showSearchModal}
        setShowSearchModal={setShowSearchModal}
        setSearchWord={setSearchWord}
      />
    </div>
  );
}
