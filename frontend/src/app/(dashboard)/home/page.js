"use client";

import { IoSend, IoRefresh, IoCopyOutline } from "react-icons/io5";
import {
  MdMyLocation,
  MdUpload,
  MdDescription,
  MdChevronRight,
} from "react-icons/md";
import { useState, useRef } from "react";
import LocationSearch from "@/components/LocationSearch";

export default function HomePageClient() {
  const [singleTweet, setSingleTweet] = useState("");
  const [csvFile, setCsvFile] = useState(null);
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [location, setLocation] = useState("Indang");
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState(null);
  const [showJson, setShowJson] = useState(false);
  const [parsedResult, setParsedResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleCopyOutput = () => {
    navigator.clipboard
      .writeText(output)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch((err) => {
        console.error("Failed to copy: ", err);
      });
  };

  const handleSingleTweet = async () => {
    if (!singleTweet.trim()) {
      setError("Please enter a tweet first!");
      setOutput("");
      return;
    }
    if (!location) {
      setError("Please select a location first!");
      setOutput("");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        "http://localhost:8000/analyze-single-tweet",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            tweet: singleTweet,
            location: location,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.status === "error") {
        setError(result.error || "Analysis failed");
        setOutput("");
        setParsedResult(null);
      } else {
        setOutput(JSON.stringify(result, null, 2));
        setParsedResult(result);
        setError(null);
      }
    } catch (err) {
      console.error("Error analyzing tweet:", err);
      setError(
        `Connection error: ${err.message}. Make sure your backend is running on port 8000.`
      );
      setOutput("");
      setParsedResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type === "text/csv") {
      setCsvFile(file);
      setError(null);
      setOutput("");
    } else {
      setError("Please select a valid CSV file");
      setCsvFile(null);
      setOutput("");
    }
  };

  const handleCsvUpload = async () => {
    if (!csvFile) {
      setError("Please select a CSV file first!");
      setOutput("");
      return;
    }
    if (!location) {
      setError("Please select a location first!");
      setOutput("");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", csvFile);
      formData.append("location", location);

      const response = await fetch("http://localhost:8000/analyze-csv-tweets", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.status === "error") {
        setError(result.error || "CSV analysis failed");
        setOutput("");
        setParsedResult(null);
      } else {
        setOutput(JSON.stringify(result, null, 2));
        setParsedResult(result);
        setError(null);
      }
    } catch (err) {
      console.error("Error analyzing CSV:", err);
      setError(
        `Connection error: ${err.message}. Make sure your backend is running on port 8000.`
      );
      setOutput("");
      setParsedResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveFile = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setCsvFile(null);
    setOutput("");
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleClear = () => {
    setSingleTweet("");
    setCsvFile(null);
    setOutput("");
    setParsedResult(null);
    setLocation("");
    setError(null);
    setShowJson(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (csvFile) {
        handleCsvUpload();
      } else {
        handleSingleTweet();
      }
    }
  };

  const handleSubmit = () => {
    if (csvFile) {
      handleCsvUpload();
    } else {
      handleSingleTweet();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-bluish-white">
      {/* HEADER */}
      <div className="bg-white px-6 w-full h-20 flex items-center justify-between border-b-2 border-primary-dark/5">
        <div className="flex items-center gap-4">
          <div className="w-fit text-2xl font-bold bg-gradient-to-r from-black via-primary-dark to-primary bg-clip-text text-transparent leading-snug">
            Sentiment Analysis Framework
          </div>
          <MdChevronRight className="text-3xl text-gray-light" />
          <p className="text-gray-mid text-2xl font-medium">Home</p>
        </div>
        <button
          onClick={handleClear}
          className="px-6 py-2 bg-bluish-gray text-gray-dark text-sm font-semibold rounded-full transition-colors duration-200 ease-in-out hover:bg-primary hover:text-white active:bg-primary-dark active:text-white cursor-pointer"
        >
          Clear
        </button>
      </div>

      {/* OUTPUT SECTION */}
      <div className="flex-1 overflow-auto bg-white">
        {!output && !error ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md px-6">
              <div className="w-48 h-48 bg-gradient-to-r from-black via-primary-dark to-primary rounded-full flex items-center justify-center mx-auto mb-6 opacity-100"></div>
              <h3 className="text-2xl font-semibold mb-3 bg-gradient-to-r from-black via-primary-dark to-primary bg-clip-text text-transparent">
                Ready to analyze tweets
              </h3>
              <p className="text-gray-dark tracking-wide text-base">
                Select a location, then type your tweet or upload a CSV file to
                get started with sentiment analysis.
              </p>
            </div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full">
            <div className="max-w-2xl mx-auto px-6">
              <div className="bg-red/10 border-l-4 border-red p-6 rounded-lg">
                <div className="flex items-start">
                  <div className="flex-shrink-0">
                    <svg
                      className="h-6 w-6 text-red"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h3 className="text-lg font-medium text-red">Error</h3>
                    <p className="mt-2 text-sm text-red">{error}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-6xl mx-auto my-8 px-6">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-gradient-to-r from-black via-primary-dark to-primary rounded-full flex items-center justify-center flex-shrink-0"></div>
              <div className="flex-1">
                {/* Toggle Button */}
                <div className="flex justify-end mb-3">
                  <button
                    onClick={() => setShowJson(!showJson)}
                    className="px-4 py-2 bg-primary text-white text-sm font-semibold rounded-lg hover:bg-primary-dark transition-colors duration-200 flex items-center gap-2"
                  >
                    {showJson ? (
                      <>
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                          />
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                          />
                        </svg>
                        View Formatted
                      </>
                    ) : (
                      <>
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                          />
                        </svg>
                        View JSON
                      </>
                    )}
                  </button>
                </div>

                <div className="border border-gray-light rounded-2xl p-4 bg-white shadow-sm relative">
                  {/* Copy Button - only show in JSON view */}
                  {showJson && (
                    <>
                      <button
                        onClick={handleCopyOutput}
                        className="absolute top-4 right-4 z-10 p-2 bg-gray-mid hover:bg-gray-dark rounded-md transition-colors duration-200"
                        title="Copy JSON"
                      >
                        <IoCopyOutline className="text-white" size={16} />
                      </button>

                      {copied && (
                        <div className="absolute top-12 right-4 bg-black text-white text-xs py-1 px-2 rounded-md shadow-lg">
                          Copied!
                        </div>
                      )}
                    </>
                  )}

                  {showJson ? (
                    <div className="bg-black rounded-xl p-4 overflow-auto">
                      <pre className="text-sm text-blue font-mono leading-relaxed whitespace-pre-wrap">
                        {output}
                      </pre>
                    </div>
                  ) : parsedResult ? (
                    <div className="space-y-4">
                      {/* Tweet Info */}
                      <div className="bg-gradient-to-r from-blue/20 to-blue/10 p-4 rounded-lg border border-blue">
                        <h3 className="font-semibold text-lg text-black mb-2">
                          Tweet Analysis
                        </h3>
                        <p className="text-primary-dark text-base mb-2 italic">
                          &quot;{parsedResult.tweet}&quot;
                        </p>
                        <div className="flex gap-4 text-sm text-gray-mid">
                          <span>📍 {parsedResult.location}</span>
                          <span>📏 {parsedResult.length} characters</span>
                        </div>
                      </div>

                      {/* Climate Classification */}
                      {parsedResult.climate_classification && (
                        <div
                          className={`p-4 rounded-lg border ${
                            parsedResult.climate_classification
                              .is_climate_related
                              ? "bg-primary/10 border-primary"
                              : "bg-white/50 border-gray-light"
                          }`}
                        >
                          <h3 className="font-semibold text-lg text-black mb-2">
                            Climate Related:{" "}
                            {parsedResult.climate_classification
                              .is_climate_related
                              ? "Yes"
                              : "No"}
                          </h3>
                          <p className="text-sm text-gray-mid">
                            Confidence:{" "}
                            <span className="font-semibold">
                              {(
                                parsedResult.climate_classification.confidence *
                                100
                              ).toFixed(1)}
                              %
                            </span>
                          </p>
                        </div>
                      )}

                      {/* Category Classification */}
                      {parsedResult.category_classification &&
                        parsedResult.climate_classification
                          ?.is_climate_related && (
                          <div className="bg-primary/10 p-4 rounded-lg border border-primary">
                            <h3 className="font-semibold text-lg text-black mb-2">
                              Category
                            </h3>
                            <p className="text-base font-semibold text-primary mb-2">
                              {parsedResult.category_classification.prediction}
                            </p>
                            <p className="text-sm text-gray-mid mb-3">
                              Confidence:{" "}
                              <span className="font-semibold">
                                {(
                                  parsedResult.category_classification
                                    .confidence * 100
                                ).toFixed(1)}
                                %
                              </span>
                            </p>

                            {/* Top 3 probabilities */}
                            <div className="space-y-1">
                              <p className="text-xs font-semibold text-primary-dark mb-1">
                                Top Predictions:
                              </p>
                              {Object.entries(
                                parsedResult.category_classification
                                  .probabilities || {}
                              )
                                .slice(0, 3)
                                .map(([cat, prob], idx) => (
                                  <div
                                    key={idx}
                                    className="flex justify-between items-center text-xs"
                                  >
                                    <span className="text-primary-dark">
                                      {idx + 1}. {cat}
                                    </span>
                                    <span className="font-semibold text-primary">
                                      {(prob * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                ))}
                            </div>
                          </div>
                        )}

                      {/* Weather Validation */}
                      {parsedResult.weather_validation?.weather_data && (
                        <div className="bg-blue/10 p-4 rounded-lg border border-blue">
                          <div className="flex justify-between items-start mb-3">
                            <h3 className="font-semibold text-lg text-black">
                              Weather Data
                            </h3>
                            <span
                              className={`px-3 py-1 rounded-full text-xs font-semibold ${
                                parsedResult.weather_validation.validation
                                  ?.consistency === "consistent"
                                  ? "bg-primary/20 text-primary"
                                  : "bg-gray/20 text-gray-dark"
                              }`}
                            >
                              {parsedResult.weather_flag}
                            </span>
                          </div>

                          <div className="grid grid-cols-2 gap-3 text-sm">
                            <div>
                              <p className="text-gray-mid">Temperature</p>
                              <p className="font-semibold text-black">
                                {
                                  parsedResult.weather_validation.weather_data
                                    .temperature_c
                                }
                                °C /{" "}
                                {
                                  parsedResult.weather_validation.weather_data
                                    .temperature_f
                                }
                                °F
                              </p>
                            </div>
                            <div>
                              <p className="text-gray-mid">Feels Like</p>
                              <p className="font-semibold text-black">
                                {
                                  parsedResult.weather_validation.weather_data
                                    .feels_like_c
                                }
                                °C
                              </p>
                            </div>
                            <div>
                              <p className="text-gray-mid">Condition</p>
                              <p className="font-semibold text-black">
                                {
                                  parsedResult.weather_validation.weather_data
                                    .condition
                                }
                              </p>
                            </div>
                            <div>
                              <p className="text-gray-mid">Humidity</p>
                              <p className="font-semibold text-black">
                                {
                                  parsedResult.weather_validation.weather_data
                                    .humidity
                                }
                                %
                              </p>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Sentiment Analysis */}
                      {parsedResult.sentiment_analysis?.sentiment && (
                        <div className="bg-gray/10 p-4 rounded-lg border border-gray">
                          <h3 className="font-semibold text-lg text-black mb-3">
                            Sentiment Analysis
                          </h3>

                          <div className="mb-3">
                            <span
                              className={`inline-block px-4 py-2 rounded-full text-sm font-bold ${
                                parsedResult.sentiment_analysis.sentiment
                                  .classification === "positive"
                                  ? "bg-primary/20 text-primary"
                                  : parsedResult.sentiment_analysis.sentiment
                                      .classification === "negative"
                                  ? "bg-red/20 text-red"
                                  : "bg-gray/20 text-gray-dark"
                              }`}
                            >
                              {parsedResult.sentiment_analysis.sentiment.classification.toUpperCase()}
                            </span>
                          </div>

                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-mid">
                                Compound Score:
                              </span>
                              <span className="font-semibold">
                                {parsedResult.sentiment_analysis.sentiment.compound.toFixed(
                                  3
                                )}
                              </span>
                            </div>
                            <div className="space-y-1">
                              <div className="flex justify-between items-center">
                                <span className="text-primary">Positive:</span>
                                <div className="flex items-center gap-2">
                                  <div className="w-24 bg-bluish-gray rounded-full h-2">
                                    <div
                                      className="bg-primary h-2 rounded-full"
                                      style={{
                                        width: `${
                                          parsedResult.sentiment_analysis
                                            .sentiment.positive * 100
                                        }%`,
                                      }}
                                    ></div>
                                  </div>
                                  <span className="font-semibold w-12 text-right">
                                    {(
                                      parsedResult.sentiment_analysis.sentiment
                                        .positive * 100
                                    ).toFixed(1)}
                                    %
                                  </span>
                                </div>
                              </div>
                              <div className="flex justify-between items-center">
                                <span className="text-gray-mid">Neutral:</span>
                                <div className="flex items-center gap-2">
                                  <div className="w-24 bg-bluish-gray rounded-full h-2">
                                    <div
                                      className="bg-gray h-2 rounded-full"
                                      style={{
                                        width: `${
                                          parsedResult.sentiment_analysis
                                            .sentiment.neutral * 100
                                        }%`,
                                      }}
                                    ></div>
                                  </div>
                                  <span className="font-semibold w-12 text-right">
                                    {(
                                      parsedResult.sentiment_analysis.sentiment
                                        .neutral * 100
                                    ).toFixed(1)}
                                    %
                                  </span>
                                </div>
                              </div>
                              <div className="flex justify-between items-center">
                                <span className="text-red">Negative:</span>
                                <div className="flex items-center gap-2">
                                  <div className="w-24 bg-bluish-gray rounded-full h-2">
                                    <div
                                      className="bg-red h-2 rounded-full"
                                      style={{
                                        width: `${
                                          parsedResult.sentiment_analysis
                                            .sentiment.negative * 100
                                        }%`,
                                      }}
                                    ></div>
                                  </div>
                                  <span className="font-semibold w-12 text-right">
                                    {(
                                      parsedResult.sentiment_analysis.sentiment
                                        .negative * 100
                                    ).toFixed(1)}
                                    %
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Database Info */}
                      {parsedResult.saved_to_db && (
                        <div className="bg-white/50 p-3 rounded-lg border border-gray-light text-sm text-gray-mid">
                          ✅ Saved to database (ID: {parsedResult.database_id})
                        </div>
                      )}
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* INPUT SECTION */}
      <div className="p-6 bg-white border-t border-bluish-gray">
        <div className="max-w-6xl mx-auto">
          {/* LOCATION SEARCH BAR */}
          <div className="mb-3">
            <LocationSearch
              value={location}
              onChange={setLocation}
              placeholder="Search location..."
              icon={MdMyLocation}
            />
          </div>

          {/* INPUT FIELD */}
          <div className="flex items-center bg-bluish-white shadow-[0px_2px_8px_0px_rgba(30,41,59,0.15)] rounded-2xl border border-bluish-gray">
            {/* CSV Upload Button - Left */}
            <div className="flex items-center border-r border-gray-light">
              <label className="cursor-pointer flex flex-col items-center py-4 text-gray-dark font-semibold relative group w-24">
                {csvFile ? (
                  <div className="relative">
                    <MdDescription size={40} className="text-primary" />
                    <button
                      onClick={handleRemoveFile}
                      className="absolute -top-1 -right-1 w-4 h-4 bg-red hover:bg-red/80 rounded-full flex items-center justify-center transition-colors"
                      title="Remove file"
                    >
                      <svg
                        className="w-2.5 h-2.5 text-white"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={3}
                          d="M6 18L18 6M6 6l12 12"
                        />
                      </svg>
                    </button>
                  </div>
                ) : (
                  <MdUpload size={40} className="text-gray" />
                )}

                <span className="text-xs mt-1">
                  {csvFile ? "FILE" : "UPLOAD"}
                </span>

                {csvFile && (
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-black text-white text-xs rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10">
                    {csvFile.name}
                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-black"></div>
                  </div>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </label>
            </div>

            {/* Text Area - Middle */}
            <textarea
              value={csvFile ? csvFile.name : singleTweet}
              onChange={(e) => setSingleTweet(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                csvFile
                  ? `${csvFile.name} - Press Enter to analyze CSV file...`
                  : "Enter tweet here..."
              }
              rows="1"
              className="flex-1 my-2 mx-6 resize-none text-primary-dark disabled:text-gray outline-none text-lg bg-transparent"
              disabled={csvFile}
              readOnly={csvFile}
            />

            {/* Send Button - Right */}
            <button
              onClick={handleSubmit}
              disabled={
                loading || !location || (!singleTweet.trim() && !csvFile)
              }
              className={`${
                loading || !location || (!singleTweet.trim() && !csvFile)
                  ? "cursor-not-allowed"
                  : "cursor-pointer"
              } bg-transparent border-none flex items-center justify-center w-12 h-12 flex-shrink-0 rounded my-3 mx-5`}
            >
              {loading ? (
                <IoRefresh
                  className={`w-8 h-8 animate-spin transition-colors ${
                    loading || !location || (!singleTweet.trim() && !csvFile)
                      ? "text-gray"
                      : "text-gray-dark hover:text-primary-dark"
                  }`}
                />
              ) : (
                <IoSend
                  className={`w-8 h-8 transition-colors ${
                    loading || !location || (!singleTweet.trim() && !csvFile)
                      ? "text-gray"
                      : "text-gray-dark hover:text-primary-dark"
                  }`}
                />
              )}
            </button>
          </div>

          {/* BOTTOM TEXT */}
          <p className="mt-3 text-xs text-gray-light text-center font-medium tracking-wide">
            POWERED BY VADER AND NAIVE BAYES
          </p>
        </div>
      </div>
    </div>
  );
}
