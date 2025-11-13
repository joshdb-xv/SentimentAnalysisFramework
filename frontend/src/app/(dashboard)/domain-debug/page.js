"use client";

import { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000";

export default function DomainClassifierPage() {
  const [activeTab, setActiveTab] = useState("status");
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ text: "", type: "" });

  // Upload states
  const [trainingFile, setTrainingFile] = useState(null);
  const [unlabeledFile, setUnlabeledFile] = useState(null);

  // Training states
  const [replaceExisting, setReplaceExisting] = useState(false);
  const [batchName, setBatchName] = useState("");

  // Pseudo-labeling states
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.9);
  const [saveLowConfidence, setSaveLowConfidence] = useState(true);

  // Prediction state
  const [predictionText, setPredictionText] = useState("");
  const [predictionResult, setPredictionResult] = useState(null);

  useEffect(() => {
    loadStatus();
    loadHistory();
  }, []);

  const loadStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/domain-classifier/status`);
      const data = await res.json();
      setStatus(data);
    } catch (error) {
      console.error("Error loading status:", error);
    }
  };

  const loadHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/domain-classifier/history`);
      if (!res.ok) {
        console.error("Error loading history:", res.statusText);
        setHistory({
          history: [],
          improvement_stats: { improvements: [], total_batches: 0 },
        });
        return;
      }
      const data = await res.json();
      setHistory(data);
    } catch (error) {
      console.error("Error loading history:", error);
      setHistory({
        history: [],
        improvement_stats: { improvements: [], total_batches: 0 },
      });
    }
  };

  const showMessage = (text, type = "success") => {
    setMessage({ text, type });
    setTimeout(() => setMessage({ text: "", type: "" }), 5000);
  };

  const handleUploadTraining = async () => {
    if (!trainingFile) {
      showMessage("Please select a CSV file", "error");
      return;
    }

    const formData = new FormData();
    formData.append("file", trainingFile);

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/domain-classifier/upload/training`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await res.json();
      showMessage(
        `Uploaded ${data.rows} training samples. Total: ${data.total_staged_samples}`
      );
      setTrainingFile(null);
      loadStatus();
    } catch (error) {
      showMessage("Upload failed: " + error.message, "error");
    } finally {
      setLoading(false);
    }
  };

  const handleUploadUnlabeled = async () => {
    if (!unlabeledFile) {
      showMessage("Please select a CSV file", "error");
      return;
    }

    const formData = new FormData();
    formData.append("file", unlabeledFile);

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/domain-classifier/upload/unlabeled`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await res.json();
      showMessage(`Uploaded ${data.rows} unlabeled samples`);
      setUnlabeledFile(null);
      loadStatus();
    } catch (error) {
      showMessage("Upload failed: " + error.message, "error");
    } finally {
      setLoading(false);
    }
  };

  const handleTrainInitial = async () => {
    if (!status?.staged_training_samples) {
      showMessage(
        "No training data available. Upload training CSV first.",
        "error"
      );
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/domain-classifier/train/initial`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ replace_existing: replaceExisting }),
      });
      const data = await res.json();

      if (!data.success && res.status === 409) {
        showMessage(data.message, "error");
      } else {
        showMessage(
          `Model trained! Accuracy: ${(
            data.benchmarks.accuracy.mean * 100
          ).toFixed(2)}%`
        );
        loadStatus();
        loadHistory();
        setActiveTab("history");
      }
    } catch (error) {
      showMessage("Training failed: " + error.message, "error");
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async () => {
    if (!batchName.trim()) {
      showMessage("Please enter a batch name", "error");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/domain-classifier/retrain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ batch_name: batchName }),
      });
      const data = await res.json();

      const improvementMsg = data.improvement
        ? ` (${data.improvement > 0 ? "+" : ""}${data.improvement.toFixed(
            2
          )}% improvement)`
        : "";
      showMessage(`Model retrained!${improvementMsg}`);
      setBatchName("");
      loadStatus();
      loadHistory();
      setActiveTab("history");
    } catch (error) {
      showMessage("Retraining failed: " + error.message, "error");
    } finally {
      setLoading(false);
    }
  };

  const handlePseudoLabel = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/domain-classifier/pseudo-label`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confidence_threshold: confidenceThreshold,
          save_low_confidence: saveLowConfidence,
        }),
      });
      const data = await res.json();

      showMessage(
        `Pseudo-labeled ${data.total_processed} tweets. High confidence: ${data.high_confidence_count}, ` +
          `Low confidence: ${data.low_confidence_count}. Now retrain to improve model!`
      );
      loadStatus();
    } catch (error) {
      showMessage("Pseudo-labeling failed: " + error.message, "error");
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!predictionText.trim()) {
      showMessage("Please enter text to classify", "error");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/domain-classifier/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: predictionText }),
      });
      const data = await res.json();
      setPredictionResult(data);
    } catch (error) {
      showMessage("Prediction failed: " + error.message, "error");
    } finally {
      setLoading(false);
    }
  };

  const getAccuracy = (benchmarks) => {
    if (benchmarks?.overall_metrics?.accuracy !== undefined) {
      return benchmarks.overall_metrics.accuracy;
    }
    if (benchmarks?.accuracy !== undefined) {
      return benchmarks.accuracy;
    }
    return 0;
  };

  const getPrecision = (benchmarks) => {
    if (benchmarks?.overall_metrics?.precision_weighted !== undefined) {
      return benchmarks.overall_metrics.precision_weighted;
    }
    if (benchmarks?.precision_weighted !== undefined) {
      return benchmarks.precision_weighted;
    }
    return 0;
  };

  const getF1 = (benchmarks) => {
    if (benchmarks?.overall_metrics?.f1_weighted !== undefined) {
      return benchmarks.overall_metrics.f1_weighted;
    }
    if (benchmarks?.f1_weighted !== undefined) {
      return benchmarks.f1_weighted;
    }
    return 0;
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8 mt-20">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          {/* Header */}
          <div className="bg-gray-800 p-6">
            <h1 className="text-3xl font-bold text-white mb-2">
              Domain Classifier Training System
            </h1>
            <p className="text-gray-300">
              Binary Classification: Climate-Related vs Not Climate-Related
            </p>

            {/* Status Bar */}
            {status && (
              <div className="mt-4 grid grid-cols-4 gap-3">
                <div className="bg-gray-700 rounded p-3">
                  <div className="text-gray-300 text-xs mb-1">Model Status</div>
                  <div className="text-white font-semibold text-sm">
                    {status.has_model ? "Active" : "None"}
                  </div>
                </div>
                <div className="bg-gray-700 rounded p-3">
                  <div className="text-gray-300 text-xs mb-1">
                    Training Samples
                  </div>
                  <div className="text-white font-semibold text-lg">
                    {status.staged_training_samples}
                  </div>
                </div>
                <div className="bg-gray-700 rounded p-3">
                  <div className="text-gray-300 text-xs mb-1">
                    Unlabeled Files
                  </div>
                  <div className="text-white font-semibold text-lg">
                    {status.staged_unlabeled_files}
                  </div>
                </div>
                <div className="bg-gray-700 rounded p-3">
                  <div className="text-gray-300 text-xs mb-1">
                    Training Batches
                  </div>
                  <div className="text-white font-semibold text-lg">
                    {status.training_batches}
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
          {loading && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white p-6 rounded-lg shadow-xl">
                <div className="animate-spin h-12 w-12 border-4 border-gray-300 border-t-gray-800 rounded-full mx-auto mb-4"></div>
                <p className="text-lg font-semibold text-gray-800">Processing...</p>
                <p className="text-sm text-gray-600 mt-1">
                  This may take a few minutes
                </p>
              </div>
            </div>
          )}

          {/* Tabs */}
          <div className="border-b border-gray-200 bg-gray-50">
            <div className="flex overflow-x-auto">
              {["status", "train", "pseudo-label", "history", "predict"].map(
                (tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-6 py-3 font-medium capitalize whitespace-nowrap transition-colors ${
                      activeTab === tab
                        ? "bg-white border-b-2 border-gray-800 text-gray-900"
                        : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"
                    }`}
                  >
                    {tab.replace("-", " ")}
                  </button>
                )
              )}
            </div>
          </div>

          {/* Content */}
          <div className="p-6">
            {/* Status Tab */}
            {activeTab === "status" && status && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  System Status
                </h2>

                <div className="grid grid-cols-2 gap-4">
                  <div className="border rounded-lg p-4">
                    <h3 className="font-semibold text-lg mb-3 text-gray-900">
                      Current Model
                    </h3>
                    {status.has_model ? (
                      <div>
                        <div className="text-sm text-gray-600 mb-1">
                          Model Name
                        </div>
                        <div className="font-mono text-xs bg-gray-50 p-2 rounded border">
                          {status.current_model}
                        </div>
                        <div className="mt-2 text-green-700 font-medium text-sm">
                          Model Ready
                        </div>
                      </div>
                    ) : (
                      <div className="text-red-700 font-medium">
                        No model trained yet
                      </div>
                    )}
                  </div>

                  <div className="border rounded-lg p-4">
                    <h3 className="font-semibold text-lg mb-3 text-gray-900">
                      Training Data
                    </h3>
                    <div className="space-y-2">
                      <div>
                        <div className="text-sm text-gray-600">
                          Staged Training Samples
                        </div>
                        <div className="text-2xl font-bold text-gray-900">
                          {status.staged_training_samples}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">
                          Total Training Batches
                        </div>
                        <div className="text-xl font-bold text-gray-900">
                          {status.training_batches}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h3 className="font-semibold text-lg mb-3 text-gray-900">
                      Unlabeled Data
                    </h3>
                    <div>
                      <div className="text-sm text-gray-600">
                        Files Ready for Pseudo-Labeling
                      </div>
                      <div className="text-2xl font-bold text-gray-900">
                        {status.staged_unlabeled_files}
                      </div>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h3 className="font-semibold text-lg mb-3 text-gray-900">
                      Low Confidence
                    </h3>
                    <div>
                      <div className="text-sm text-gray-600">
                        Files for Manual Labeling
                      </div>
                      <div className="text-2xl font-bold text-gray-900">
                        {status.low_confidence_files}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg">
                  <h3 className="font-semibold text-lg mb-3 text-gray-900">
                    Workflow Guide
                  </h3>
                  <ol className="space-y-2 text-gray-700 text-sm">
                    <li className="flex gap-2">
                      <span className="font-semibold">1.</span> Upload labeled training CSV (text, category/label columns where value is 0 or 1)
                    </li>
                    <li className="flex gap-2">
                      <span className="font-semibold">2.</span> Train initial model or retrain with new data
                    </li>
                    <li className="flex gap-2">
                      <span className="font-semibold">3.</span> Upload unlabeled CSV files (text column only)
                    </li>
                    <li className="flex gap-2">
                      <span className="font-semibold">4.</span> Run pseudo-labeling to auto-label high-confidence tweets
                    </li>
                    <li className="flex gap-2">
                      <span className="font-semibold">5.</span> Return to Train tab and retrain model with expanded dataset
                    </li>
                    <li className="flex gap-2">
                      <span className="font-semibold">6.</span> Check history tab for improvement metrics
                    </li>
                  </ol>
                </div>
              </div>
            )}

            {/* Train Tab */}
            {activeTab === "train" && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Train / Retrain Model
                </h2>

                {/* Training Data Upload */}
                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 text-gray-900">
                    Step 1: Upload Training Data
                  </h3>
                  <p className="text-sm text-gray-600 mb-3">
                    CSV must contain: <code className="bg-gray-100 px-2 py-1 rounded">text</code>, <code className="bg-gray-100 px-2 py-1 rounded">category</code> or <code className="bg-gray-100 px-2 py-1 rounded">label</code> column (0=not climate, 1=climate)
                  </p>

                  <div className="flex gap-3 items-end">
                    <div className="flex-1">
                      <label className="block text-sm font-medium mb-2 text-gray-700">
                        Select CSV File
                      </label>
                      <input
                        type="file"
                        accept=".csv"
                        onChange={(e) => setTrainingFile(e.target.files[0])}
                        className="w-full p-2 border border-gray-300 rounded focus:border-gray-500 focus:ring-1 focus:ring-gray-500"
                      />
                    </div>
                    <button
                      onClick={handleUploadTraining}
                      disabled={!trainingFile || loading}
                      className="px-6 py-2 bg-gray-800 text-white rounded font-medium hover:bg-gray-700 disabled:bg-gray-300 transition-colors"
                    >
                      Upload Training Data
                    </button>
                  </div>
                  {trainingFile && (
                    <div className="mt-2 text-sm text-gray-600">
                      Selected: <span className="font-medium">{trainingFile.name}</span>
                    </div>
                  )}

                  {status && (
                    <div className="mt-3 bg-gray-50 border border-gray-200 rounded p-3">
                      <div className="font-medium text-gray-900 text-sm">
                        Current Status:
                      </div>
                      <div className="text-gray-700 text-sm mt-1">
                        Staged training samples: <span className="font-semibold">{status.staged_training_samples}</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Initial Training */}
                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 text-gray-900">
                    Step 2: Initial Training
                  </h3>
                  <p className="text-gray-700 text-sm mb-3">
                    Train a new model with all staged training data. Always performs 5 runs with different seeds.
                  </p>

                  {status?.has_model && (
                    <label className="flex items-center gap-2 mb-3 p-2 bg-yellow-50 rounded border border-yellow-200">
                      <input
                        type="checkbox"
                        checked={replaceExisting}
                        onChange={(e) => setReplaceExisting(e.target.checked)}
                        className="w-4 h-4"
                      />
                      <span className="font-medium text-gray-800 text-sm">
                        Replace existing model (current model will be archived)
                      </span>
                    </label>
                  )}

                  <button
                    onClick={handleTrainInitial}
                    disabled={!status?.staged_training_samples || loading}
                    className="w-full px-6 py-3 bg-gray-800 text-white rounded font-semibold hover:bg-gray-700 disabled:bg-gray-300 transition-colors"
                  >
                    Train Initial Model (5 runs)
                  </button>

                  {!status?.staged_training_samples && (
                    <p className="text-red-700 mt-2 text-sm">
                      Upload training data first
                    </p>
                  )}
                </div>

                {/* Retrain */}
                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 text-gray-900">
                    Step 3: Retrain with New Data
                  </h3>
                  <p className="text-gray-700 text-sm mb-3">
                    Retrain model after adding new labeled data or pseudo-labeled tweets. Old model is automatically archived.
                  </p>

                  <div className="mb-3">
                    <label className="block text-sm font-medium mb-2 text-gray-700">
                      Batch Name
                    </label>
                    <input
                      type="text"
                      value={batchName}
                      onChange={(e) => setBatchName(e.target.value)}
                      placeholder="e.g., batch_2_with_pseudolabels"
                      className="w-full p-2 border border-gray-300 rounded focus:border-gray-500 focus:ring-1 focus:ring-gray-500"
                    />
                  </div>

                  <button
                    onClick={handleRetrain}
                    disabled={!status?.has_model || !batchName.trim() || loading}
                    className="w-full px-6 py-3 bg-gray-800 text-white rounded font-semibold hover:bg-gray-700 disabled:bg-gray-300 transition-colors"
                  >
                    Retrain Model (5 runs)
                  </button>

                  {!status?.has_model && (
                    <p className="text-red-700 mt-2 text-sm">
                      Train an initial model first
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Pseudo-Label Tab */}
            {activeTab === "pseudo-label" && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Pseudo-Labeling
                </h2>

                {/* Unlabeled Data Upload */}
                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 text-gray-900">
                    Step 1: Upload Unlabeled Data
                  </h3>
                  <p className="text-sm text-gray-600 mb-3">
                    CSV must contain: <code className="bg-gray-100 px-2 py-1 rounded">text</code> column
                  </p>

                  <div className="flex gap-3 items-end">
                    <div className="flex-1">
                      <label className="block text-sm font-medium mb-2 text-gray-700">
                        Select CSV File
                      </label>
                      <input
                        type="file"
                        accept=".csv"
                        onChange={(e) => setUnlabeledFile(e.target.files[0])}
                        className="w-full p-2 border border-gray-300 rounded focus:border-gray-500 focus:ring-1 focus:ring-gray-500"
                      />
                    </div>
                    <button
                      onClick={handleUploadUnlabeled}
                      disabled={!unlabeledFile || loading}
                      className="px-6 py-2 bg-gray-800 text-white rounded font-medium hover:bg-gray-700 disabled:bg-gray-300 transition-colors"
                    >
                      Upload Unlabeled Data
                    </button>
                  </div>
                  {unlabeledFile && (
                    <div className="mt-2 text-sm text-gray-600">
                      Selected: <span className="font-medium">{unlabeledFile.name}</span>
                    </div>
                  )}

                  {status && (
                    <div className="mt-3 bg-gray-50 border border-gray-200 rounded p-3">
                      <div className="font-medium text-gray-900 text-sm">
                        Current Status:
                      </div>
                      <div className="text-gray-700 text-sm mt-1">
                        Unlabeled files ready: <span className="font-semibold">{status.staged_unlabeled_files}</span>
                      </div>
                    </div>
                  )}
                </div>

                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 text-gray-900">
                    Step 2: Configure & Run Pseudo-Labeling
                  </h3>
                  <p className="text-gray-700 text-sm mb-4">
                    Automatically label unlabeled tweets using the current model. High-confidence predictions are added to training data, low-confidence ones are saved for manual review.
                  </p>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-2 text-gray-700">
                        Confidence Threshold: {confidenceThreshold.toFixed(2)}
                      </label>
                      <input
                        type="range"
                        min="0.5"
                        max="1.0"
                        step="0.05"
                        value={confidenceThreshold}
                        onChange={(e) =>
                          setConfidenceThreshold(parseFloat(e.target.value))
                        }
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-gray-600 mt-1">
                        <span>0.5 (More labels)</span>
                        <span>1.0 (Fewer, more accurate)</span>
                      </div>
                    </div>

                    <label className="flex items-center gap-2 p-2 bg-gray-50 rounded border border-gray-200">
                      <input
                        type="checkbox"
                        checked={saveLowConfidence}
                        onChange={(e) => setSaveLowConfidence(e.target.checked)}
                        className="w-4 h-4"
                      />
                      <span className="font-medium text-gray-800 text-sm">
                        Save low-confidence predictions for manual labeling
                      </span>
                    </label>
                  </div>

                  <button
                    onClick={handlePseudoLabel}
                    disabled={
                      !status?.has_model ||
                      !status?.staged_unlabeled_files ||
                      loading
                    }
                    className="w-full mt-4 px-6 py-3 bg-gray-800 text-white rounded font-semibold hover:bg-gray-700 disabled:bg-gray-300 transition-colors"
                  >
                    Run Pseudo-Labeling
                  </button>

                  {!status?.has_model && (
                    <p className="text-red-700 mt-2 text-sm">
                      Train a model first
                    </p>
                  )}
                  {status?.has_model && !status?.staged_unlabeled_files && (
                    <p className="text-red-700 mt-2 text-sm">
                      Upload unlabeled data first
                    </p>
                  )}
                </div>

                <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg">
                  <h3 className="font-semibold mb-2 text-gray-900">
                    After Pseudo-Labeling
                  </h3>
                  <p className="text-gray-700 text-sm">
                    High-confidence labeled tweets are automatically added to your training data. Go back to the <strong>Train</strong> tab and use the <strong>Retrain</strong> function to improve your model with the newly labeled data.
                  </p>
                </div>
              </div>
            )}

            {/* History Tab */}
            {activeTab === "history" && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Training History
                </h2>

                {!history ? (
                  <div className="text-center py-12">
                    <div className="animate-spin h-10 w-10 border-4 border-gray-300 border-t-gray-800 rounded-full mx-auto mb-4"></div>
                    <div className="text-gray-600">Loading history...</div>
                  </div>
                ) : history.history.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    <div className="text-5xl mb-4">📊</div>
                    <div className="text-xl">No training history yet</div>
                    <div className="text-sm mt-2">
                      Train a model to see history
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Improvement Stats */}
                    {history.improvement_stats &&
                      history.improvement_stats.improvements &&
                      history.improvement_stats.improvements.length > 0 && (
                        <div className="border rounded-lg p-4 bg-gray-50">
                          <h3 className="font-semibold text-lg mb-3 text-gray-900">
                            Overall Improvement
                          </h3>
                          <div className="text-4xl font-bold text-gray-900 mb-1">
                            +{history.improvement_stats.overall_improvement.toFixed(2)}%
                          </div>
                          <div className="text-gray-700 text-sm">
                            From first batch to current ({history.improvement_stats.total_batches} batches)
                          </div>
                        </div>
                      )}

                    {/* Batch by Batch */}
                    <div className="space-y-4">
                      <h3 className="font-semibold text-lg text-gray-900">
                        Training Batches
                      </h3>
                      {history.history.map((batch, idx) => (
                        <div
                          key={idx}
                          className="border rounded-lg p-4 hover:shadow-md transition-shadow"
                        >
                          <div className="flex justify-between items-start mb-3">
                            <div>
                              <h4 className="font-semibold text-base text-gray-900">
                                Batch #{batch.batch_number}: {batch.batch_name}
                              </h4>
                              <div className="text-xs text-gray-600">
                                {new Date(batch.timestamp).toLocaleString()}
                              </div>
                            </div>
                            {batch.improvement_from_previous !== undefined && (
                              <div
                                className={`px-3 py-1 rounded font-semibold text-sm ${
                                  batch.improvement_from_previous > 0
                                    ? "bg-green-100 text-green-800"
                                    : batch.improvement_from_previous < 0
                                    ? "bg-red-100 text-red-800"
                                    : "bg-gray-100 text-gray-800"
                                }`}
                              >
                                {batch.improvement_from_previous > 0 ? "+" : ""}
                                {batch.improvement_from_previous.toFixed(2)}%
                              </div>
                            )}
                          </div>

                          <div className="grid grid-cols-4 gap-3">
                            <div className="bg-gray-50 rounded p-2">
                              <div className="text-xs text-gray-600 mb-1">
                                Training Samples
                              </div>
                              <div className="text-xl font-bold text-gray-900">
                                {batch.training_samples}
                              </div>
                            </div>
                            <div className="bg-gray-50 rounded p-2">
                              <div className="text-xs text-gray-600 mb-1">
                                Accuracy
                              </div>
                              <div className="text-xl font-bold text-gray-900">
                                {(getAccuracy(batch.benchmarks) * 100).toFixed(2)}%
                              </div>
                            </div>
                            <div className="bg-gray-50 rounded p-2">
                              <div className="text-xs text-gray-600 mb-1">
                                Precision
                              </div>
                              <div className="text-xl font-bold text-gray-900">
                                {(getPrecision(batch.benchmarks) * 100).toFixed(2)}%
                              </div>
                            </div>
                            <div className="bg-gray-50 rounded p-2">
                              <div className="text-xs text-gray-600 mb-1">
                                F1-Score
                              </div>
                              <div className="text-xl font-bold text-gray-900">
                                {(getF1(batch.benchmarks) * 100).toFixed(2)}%
                              </div>
                            </div>
                          </div>

                          {batch.multiple_runs_stats && (
                            <div className="mt-3 p-3 bg-gray-50 rounded border border-gray-200">
                              <div className="text-xs font-medium text-gray-700 mb-2">
                                5 Runs Statistics (for thesis)
                              </div>
                              <div className="grid grid-cols-2 gap-2 text-xs">
                                <div>
                                  <span className="text-gray-600">Accuracy:</span>{" "}
                                  <span className="font-semibold">
                                    {(batch.multiple_runs_stats.accuracy.mean * 100).toFixed(2)}% ±{" "}
                                    {(batch.multiple_runs_stats.accuracy.std * 100).toFixed(2)}%
                                  </span>
                                </div>
                                <div>
                                  <span className="text-gray-600">F1-Score:</span>{" "}
                                  <span className="font-semibold">
                                    {(batch.multiple_runs_stats.f1.mean * 100).toFixed(2)}% ±{" "}
                                    {(batch.multiple_runs_stats.f1.std * 100).toFixed(2)}%
                                  </span>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>

                    {/* Batch Improvements */}
                    {history.improvement_stats.improvements &&
                      history.improvement_stats.improvements.length > 0 && (
                        <div className="border rounded-lg p-4">
                          <h3 className="font-semibold text-lg mb-3 text-gray-900">
                            Batch-to-Batch Improvements
                          </h3>
                          <div className="space-y-2">
                            {history.improvement_stats.improvements.map((imp, idx) => (
                              <div
                                key={idx}
                                className="flex items-center justify-between p-3 bg-gray-50 rounded"
                              >
                                <div className="flex items-center gap-3">
                                  <div className="text-2xl">→</div>
                                  <div>
                                    <div className="font-medium text-gray-800 text-sm">
                                      {imp.from_batch} → {imp.to_batch}
                                    </div>
                                    <div className="text-xs text-gray-600">
                                      Added {imp.added_samples} samples ({imp.prev_samples} → {imp.curr_samples})
                                    </div>
                                  </div>
                                </div>
                                <div className="text-right">
                                  <div
                                    className={`text-xl font-bold ${
                                      imp.improvement_percent > 0
                                        ? "text-green-700"
                                        : "text-red-700"
                                    }`}
                                  >
                                    {imp.improvement_percent > 0 ? "+" : ""}
                                    {imp.improvement_percent.toFixed(2)}%
                                  </div>
                                  <div className="text-xs text-gray-600">
                                    {imp.prev_accuracy.toFixed(2)}% → {imp.curr_accuracy.toFixed(2)}%
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                  </>
                )}
              </div>
            )}

            {/* Predict Tab */}
            {activeTab === "predict" && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Test Predictions
                </h2>

                <div className="border rounded-lg p-4">
                  <label className="block text-sm font-medium mb-2 text-gray-700">
                    Enter Tweet or Text to Classify
                  </label>
                  <textarea
                    value={predictionText}
                    onChange={(e) => setPredictionText(e.target.value)}
                    placeholder="Enter a tweet or text to classify..."
                    className="w-full p-3 border border-gray-300 rounded h-32 focus:border-gray-500 focus:ring-1 focus:ring-gray-500"
                  />

                  <button
                    onClick={handlePredict}
                    disabled={!predictionText.trim() || !status?.has_model || loading}
                    className="w-full mt-3 px-6 py-3 bg-gray-800 text-white rounded font-semibold hover:bg-gray-700 disabled:bg-gray-300 transition-colors"
                  >
                    Predict
                  </button>

                  {!status?.has_model && (
                    <p className="text-red-700 mt-2 text-sm">
                      Train a model first
                    </p>
                  )}
                </div>

                {predictionResult && (
                  <div className="border rounded-lg p-4">
                    <h3 className="font-semibold text-xl mb-4 text-gray-900">
                      Prediction Results
                    </h3>

                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="border rounded-lg p-4 bg-gray-50">
                        <div className="text-sm text-gray-600 mb-2">
                          Classification
                        </div>
                        <div className="text-3xl font-bold text-gray-900">
                          {predictionResult.is_climate_related ? "Climate-Related" : "Not Climate-Related"}
                        </div>
                        <div className="text-sm text-gray-600 mt-1">
                          Label: {predictionResult.prediction}
                        </div>
                      </div>
                      <div className="border rounded-lg p-4 bg-gray-50">
                        <div className="text-sm text-gray-600 mb-2">
                          Confidence
                        </div>
                        <div className="text-3xl font-bold text-gray-900">
                          {(predictionResult.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    <div className="bg-gray-50 rounded-lg p-4 mb-4">
                      <div className="font-medium mb-3 text-gray-900 text-sm">
                        Class Probabilities
                      </div>
                      <div className="space-y-2">
                        {Object.entries(predictionResult.probabilities).map(([cls, prob]) => (
                          <div key={cls}>
                            <div className="flex justify-between text-xs mb-1">
                              <span className="font-medium text-gray-700">
                                {cls === "0" ? "Not Climate-Related" : "Climate-Related"}
                              </span>
                              <span className="text-gray-600">
                                {(prob * 100).toFixed(2)}%
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                              <div
                                className="bg-gray-800 h-2 rounded-full transition-all duration-500"
                                style={{ width: `${prob * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                      <div className="text-xs text-gray-600 mb-1">
                        Processed Text
                      </div>
                      <div className="text-gray-800 text-sm">
                        {predictionResult.processed_text}
                      </div>
                    </div>
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