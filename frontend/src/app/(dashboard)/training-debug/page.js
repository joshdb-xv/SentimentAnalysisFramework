"use client";

import { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000";

export default function ClassifierTestPage() {
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
      const res = await fetch(`${API_BASE}/classifier/status`);
      const data = await res.json();
      setStatus(data);
    } catch (error) {
      console.error("Error loading status:", error);
    }
  };

  const loadHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/classifier/history`);
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
      const res = await fetch(`${API_BASE}/classifier/upload/training`, {
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
      const res = await fetch(`${API_BASE}/classifier/upload/unlabeled`, {
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
      const res = await fetch(`${API_BASE}/classifier/train/initial`, {
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
      const res = await fetch(`${API_BASE}/classifier/retrain`, {
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
      const res = await fetch(`${API_BASE}/classifier/pseudo-label`, {
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
      const res = await fetch(`${API_BASE}/classifier/predict`, {
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

  // Helper function to safely extract accuracy from benchmark
  const getAccuracy = (benchmarks) => {
    if (benchmarks?.overall_metrics?.accuracy !== undefined) {
      return benchmarks.overall_metrics.accuracy;
    }
    if (benchmarks?.accuracy !== undefined) {
      return benchmarks.accuracy;
    }
    return 0;
  };

  // Helper function to safely extract precision
  const getPrecision = (benchmarks) => {
    if (benchmarks?.overall_metrics?.precision_weighted !== undefined) {
      return benchmarks.overall_metrics.precision_weighted;
    }
    if (benchmarks?.precision_weighted !== undefined) {
      return benchmarks.precision_weighted;
    }
    return 0;
  };

  // Helper function to safely extract F1
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-8 mt-20">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white/95 backdrop-blur rounded-2xl shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 p-8">
            <h1 className="text-4xl font-bold text-white mb-3">
              🌍 Climate Classifier Training System
            </h1>
            <p className="text-blue-100 text-lg">
              Iterative Model Improvement with Pseudo-Labeling
            </p>

            {/* Status Bar */}
            {status && (
              <div className="mt-6 grid grid-cols-4 gap-4">
                <div className="bg-white/20 backdrop-blur rounded-lg p-3">
                  <div className="text-white/80 text-xs mb-1">Model Status</div>
                  <div className="text-white font-bold">
                    {status.has_model ? "✅ Active" : "❌ None"}
                  </div>
                </div>
                <div className="bg-white/20 backdrop-blur rounded-lg p-3">
                  <div className="text-white/80 text-xs mb-1">
                    Training Samples
                  </div>
                  <div className="text-white font-bold text-xl">
                    {status.staged_training_samples}
                  </div>
                </div>
                <div className="bg-white/20 backdrop-blur rounded-lg p-3">
                  <div className="text-white/80 text-xs mb-1">
                    Unlabeled Files
                  </div>
                  <div className="text-white font-bold text-xl">
                    {status.staged_unlabeled_files}
                  </div>
                </div>
                <div className="bg-white/20 backdrop-blur rounded-lg p-3">
                  <div className="text-white/80 text-xs mb-1">
                    Training Batches
                  </div>
                  <div className="text-white font-bold text-xl">
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
                  ? "bg-red-100 text-red-800 border-l-4 border-red-500"
                  : "bg-green-100 text-green-800 border-l-4 border-green-500"
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="text-xl">
                  {message.type === "error" ? "❌" : "✅"}
                </span>
                <span>{message.text}</span>
              </div>
            </div>
          )}

          {/* Loading Overlay */}
          {loading && (
            <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50">
              <div className="bg-white p-8 rounded-2xl shadow-2xl">
                <div className="animate-spin h-16 w-16 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
                <p className="text-xl font-bold text-gray-800">Processing...</p>
                <p className="text-sm text-gray-600 mt-2">
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
                    className={`px-8 py-4 font-semibold capitalize whitespace-nowrap transition-all ${
                      activeTab === tab
                        ? "bg-white border-b-4 border-blue-600 text-blue-600 shadow-sm"
                        : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"
                    }`}
                  >
                    {tab === "status" && "📊"}
                    {tab === "train" && "🎓"}
                    {tab === "pseudo-label" && "🏷️"}
                    {tab === "history" && "📈"}
                    {tab === "predict" && "🔮"} {tab.replace("-", " ")}
                  </button>
                )
              )}
            </div>
          </div>

          {/* Content */}
          <div className="p-8">
            {/* Status Tab */}
            {activeTab === "status" && status && (
              <div className="space-y-6">
                <h2 className="text-3xl font-bold text-gray-900 mb-6">
                  System Status
                </h2>

                <div className="grid grid-cols-2 gap-6">
                  <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border-2 border-blue-200">
                    <h3 className="font-bold text-lg mb-4 text-blue-900">
                      Current Model
                    </h3>
                    {status.has_model ? (
                      <div>
                        <div className="text-sm text-gray-600 mb-1">
                          Model Name
                        </div>
                        <div className="font-mono text-xs bg-white p-2 rounded border">
                          {status.current_model}
                        </div>
                        <div className="mt-3 text-green-600 font-semibold">
                          ✅ Model Ready
                        </div>
                      </div>
                    ) : (
                      <div className="text-red-600 font-semibold">
                        ❌ No model trained yet
                      </div>
                    )}
                  </div>

                  <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border-2 border-green-200">
                    <h3 className="font-bold text-lg mb-4 text-green-900">
                      Training Data
                    </h3>
                    <div className="space-y-3">
                      <div>
                        <div className="text-sm text-gray-600">
                          Staged Training Samples
                        </div>
                        <div className="text-3xl font-bold text-green-600">
                          {status.staged_training_samples}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">
                          Total Training Batches
                        </div>
                        <div className="text-2xl font-bold text-green-700">
                          {status.training_batches}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-6 border-2 border-purple-200">
                    <h3 className="font-bold text-lg mb-4 text-purple-900">
                      Unlabeled Data
                    </h3>
                    <div className="space-y-2">
                      <div>
                        <div className="text-sm text-gray-600">
                          Files Ready for Pseudo-Labeling
                        </div>
                        <div className="text-3xl font-bold text-purple-600">
                          {status.staged_unlabeled_files}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-orange-50 to-amber-50 rounded-xl p-6 border-2 border-orange-200">
                    <h3 className="font-bold text-lg mb-4 text-orange-900">
                      Low Confidence
                    </h3>
                    <div>
                      <div className="text-sm text-gray-600">
                        Files for Manual Labeling
                      </div>
                      <div className="text-3xl font-bold text-orange-600">
                        {status.low_confidence_files}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-lg">
                  <h3 className="font-bold text-lg mb-3 text-blue-900">
                    💡 Workflow Guide
                  </h3>
                  <ol className="space-y-2 text-gray-700">
                    <li className="flex gap-2">
                      <span className="font-bold text-blue-600">1.</span> Go to
                      Train tab and upload labeled training CSV (text, label
                      columns)
                    </li>
                    <li className="flex gap-2">
                      <span className="font-bold text-blue-600">2.</span> Train
                      initial model or retrain with new data
                    </li>
                    <li className="flex gap-2">
                      <span className="font-bold text-blue-600">3.</span> Go to
                      Pseudo-Label tab and upload unlabeled CSV files (text
                      column)
                    </li>
                    <li className="flex gap-2">
                      <span className="font-bold text-blue-600">4.</span> Run
                      pseudo-labeling to auto-label high-confidence tweets
                    </li>
                    <li className="flex gap-2">
                      <span className="font-bold text-blue-600">5.</span> Return
                      to Train tab and retrain model with expanded dataset
                    </li>
                    <li className="flex gap-2">
                      <span className="font-bold text-blue-600">6.</span> Check
                      history tab for improvement metrics
                    </li>
                  </ol>
                </div>
              </div>
            )}

            {/* Train Tab */}
            {activeTab === "train" && (
              <div className="space-y-8">
                <h2 className="text-3xl font-bold text-gray-900 mb-6">
                  Train / Retrain Model
                </h2>

                {/* Training Data Upload Section */}
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border-2 border-green-300">
                  <h3 className="text-xl font-bold mb-4 text-green-900">
                    📊 Step 1: Upload Training Data
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">
                    CSV must contain:{" "}
                    <code className="bg-white px-2 py-1 rounded">text</code>,{" "}
                    <code className="bg-white px-2 py-1 rounded">label</code>{" "}
                    columns
                  </p>

                  <div className="flex gap-4 items-end">
                    <div className="flex-1">
                      <label className="block text-sm font-semibold mb-2 text-gray-700">
                        Select CSV File
                      </label>
                      <input
                        type="file"
                        accept=".csv"
                        onChange={(e) => setTrainingFile(e.target.files[0])}
                        className="w-full p-3 border-2 border-gray-300 rounded-lg focus:border-green-500 focus:ring-2 focus:ring-green-200"
                      />
                    </div>
                    <button
                      onClick={handleUploadTraining}
                      disabled={!trainingFile || loading}
                      className="px-8 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 disabled:bg-gray-300 transition-all shadow-lg hover:shadow-xl"
                    >
                      Upload Training Data
                    </button>
                  </div>
                  {trainingFile && (
                    <div className="mt-3 text-sm text-gray-600">
                      Selected:{" "}
                      <span className="font-semibold">{trainingFile.name}</span>
                    </div>
                  )}

                  {status && (
                    <div className="mt-4 bg-white border border-green-200 rounded-lg p-4">
                      <div className="font-semibold text-green-900">
                        Current Status:
                      </div>
                      <div className="text-gray-700 mt-2">
                        • Staged training samples:{" "}
                        <span className="font-bold">
                          {status.staged_training_samples}
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Initial Training */}
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border-2 border-blue-300">
                  <h3 className="text-xl font-bold mb-4 text-blue-900">
                    🎓 Step 2: Initial Training
                  </h3>
                  <p className="text-gray-700 mb-4">
                    Train a new model with all staged training data. Always
                    performs 5 runs with different seeds.
                  </p>

                  {status?.has_model && (
                    <label className="flex items-center gap-3 mb-4 p-3 bg-yellow-50 rounded-lg border border-yellow-300">
                      <input
                        type="checkbox"
                        checked={replaceExisting}
                        onChange={(e) => setReplaceExisting(e.target.checked)}
                        className="w-5 h-5"
                      />
                      <span className="font-medium text-gray-800">
                        Replace existing model (current model will be archived)
                      </span>
                    </label>
                  )}

                  <button
                    onClick={handleTrainInitial}
                    disabled={!status?.staged_training_samples || loading}
                    className="w-full px-8 py-4 bg-blue-600 text-white rounded-lg font-bold text-lg hover:bg-blue-700 disabled:bg-gray-300 transition-all shadow-lg hover:shadow-xl"
                  >
                    Train Initial Model (5 runs)
                  </button>

                  {!status?.staged_training_samples && (
                    <p className="text-red-600 mt-3 text-sm">
                      ⚠️ Upload training data first
                    </p>
                  )}
                </div>

                {/* Retrain */}
                <div className="bg-gradient-to-br from-orange-50 to-amber-50 rounded-xl p-6 border-2 border-orange-300">
                  <h3 className="text-xl font-bold mb-4 text-orange-900">
                    🔄 Step 3: Retrain with New Data
                  </h3>
                  <p className="text-gray-700 mb-4">
                    Retrain model after adding new labeled data or
                    pseudo-labeled tweets. Old model is automatically archived.
                  </p>

                  <div className="mb-4">
                    <label className="block text-sm font-semibold mb-2 text-gray-700">
                      Batch Name
                    </label>
                    <input
                      type="text"
                      value={batchName}
                      onChange={(e) => setBatchName(e.target.value)}
                      placeholder="e.g., batch_2_with_pseudolabels"
                      className="w-full p-3 border-2 border-gray-300 rounded-lg focus:border-orange-500 focus:ring-2 focus:ring-orange-200"
                    />
                  </div>

                  <button
                    onClick={handleRetrain}
                    disabled={
                      !status?.has_model || !batchName.trim() || loading
                    }
                    className="w-full px-8 py-4 bg-orange-600 text-white rounded-lg font-bold text-lg hover:bg-orange-700 disabled:bg-gray-300 transition-all shadow-lg hover:shadow-xl"
                  >
                    Retrain Model (5 runs)
                  </button>

                  {!status?.has_model && (
                    <p className="text-red-600 mt-3 text-sm">
                      ⚠️ Train an initial model first
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Pseudo-Label Tab */}
            {activeTab === "pseudo-label" && (
              <div className="space-y-6">
                <h2 className="text-3xl font-bold text-gray-900 mb-6">
                  🏷️ Pseudo-Labeling
                </h2>

                {/* Unlabeled Data Upload Section */}
                <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-6 border-2 border-purple-300">
                  <h3 className="text-xl font-bold mb-4 text-purple-900">
                    📤 Step 1: Upload Unlabeled Data
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">
                    CSV must contain:{" "}
                    <code className="bg-white px-2 py-1 rounded">text</code>{" "}
                    column
                  </p>

                  <div className="flex gap-4 items-end">
                    <div className="flex-1">
                      <label className="block text-sm font-semibold mb-2 text-gray-700">
                        Select CSV File
                      </label>
                      <input
                        type="file"
                        accept=".csv"
                        onChange={(e) => setUnlabeledFile(e.target.files[0])}
                        className="w-full p-3 border-2 border-gray-300 rounded-lg focus:border-purple-500 focus:ring-2 focus:ring-purple-200"
                      />
                    </div>
                    <button
                      onClick={handleUploadUnlabeled}
                      disabled={!unlabeledFile || loading}
                      className="px-8 py-3 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 disabled:bg-gray-300 transition-all shadow-lg hover:shadow-xl"
                    >
                      Upload Unlabeled Data
                    </button>
                  </div>
                  {unlabeledFile && (
                    <div className="mt-3 text-sm text-gray-600">
                      Selected:{" "}
                      <span className="font-semibold">
                        {unlabeledFile.name}
                      </span>
                    </div>
                  )}

                  {status && (
                    <div className="mt-4 bg-white border border-purple-200 rounded-lg p-4">
                      <div className="font-semibold text-purple-900">
                        Current Status:
                      </div>
                      <div className="text-gray-700 mt-2">
                        • Unlabeled files ready:{" "}
                        <span className="font-bold">
                          {status.staged_unlabeled_files}
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                <div className="bg-gradient-to-br from-indigo-50 to-blue-50 rounded-xl p-6 border-2 border-indigo-300">
                  <h3 className="text-xl font-bold mb-4 text-indigo-900">
                    ⚙️ Step 2: Configure & Run Pseudo-Labeling
                  </h3>
                  <p className="text-gray-700 mb-6">
                    Automatically label unlabeled tweets using the current
                    model. High-confidence predictions are added to training
                    data, low-confidence ones are saved for manual review.
                  </p>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-semibold mb-2 text-gray-700">
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
                        className="w-full h-3 bg-gradient-to-r from-red-300 via-yellow-300 to-green-300 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-gray-600 mt-1">
                        <span>0.5 (More labels, less accurate)</span>
                        <span>1.0 (Fewer labels, very accurate)</span>
                      </div>
                    </div>

                    <label className="flex items-center gap-3 p-3 bg-white rounded-lg border-2 border-gray-200">
                      <input
                        type="checkbox"
                        checked={saveLowConfidence}
                        onChange={(e) => setSaveLowConfidence(e.target.checked)}
                        className="w-5 h-5"
                      />
                      <span className="font-medium text-gray-800">
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
                    className="w-full mt-6 px-8 py-4 bg-indigo-600 text-white rounded-lg font-bold text-lg hover:bg-indigo-700 disabled:bg-gray-300 transition-all shadow-lg hover:shadow-xl"
                  >
                    Run Pseudo-Labeling
                  </button>

                  {!status?.has_model && (
                    <p className="text-red-600 mt-3 text-sm">
                      ⚠️ Train a model first
                    </p>
                  )}
                  {status?.has_model && !status?.staged_unlabeled_files && (
                    <p className="text-red-600 mt-3 text-sm">
                      ⚠️ Upload unlabeled data first
                    </p>
                  )}
                </div>

                <div className="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-lg">
                  <h3 className="font-bold text-lg mb-3 text-blue-900">
                    💡 After Pseudo-Labeling
                  </h3>
                  <p className="text-gray-700">
                    High-confidence labeled tweets are automatically added to
                    your training data. Go back to the <strong>Train</strong>{" "}
                    tab and use the <strong>Retrain</strong> function to improve
                    your model with the newly labeled data!
                  </p>
                </div>
              </div>
            )}

            {/* History Tab */}
            {activeTab === "history" && (
              <div className="space-y-6">
                <h2 className="text-3xl font-bold text-gray-900 mb-6">
                  📈 Training History
                </h2>

                {!history ? (
                  <div className="text-center py-12">
                    <div className="animate-spin h-12 w-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
                    <div className="text-gray-600">Loading history...</div>
                  </div>
                ) : history.history.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    <div className="text-6xl mb-4">📊</div>
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
                        <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border-2 border-green-300">
                          <h3 className="font-bold text-xl mb-4 text-green-900">
                            📊 Overall Improvement
                          </h3>
                          <div className="text-5xl font-bold text-green-600 mb-2">
                            +
                            {history.improvement_stats.overall_improvement.toFixed(
                              2
                            )}
                            %
                          </div>
                          <div className="text-gray-700">
                            From first batch to current (
                            {history.improvement_stats.total_batches} batches)
                          </div>
                        </div>
                      )}

                    {/* Batch by Batch */}
                    <div className="space-y-4">
                      <h3 className="font-bold text-xl text-gray-900">
                        Training Batches
                      </h3>
                      {history.history.map((batch, idx) => (
                        <div
                          key={idx}
                          className="bg-white rounded-xl p-6 border-2 border-gray-200 shadow-sm hover:shadow-md transition-shadow"
                        >
                          <div className="flex justify-between items-start mb-4">
                            <div>
                              <h4 className="font-bold text-lg text-gray-900">
                                Batch #{batch.batch_number}: {batch.batch_name}
                              </h4>
                              <div className="text-sm text-gray-600">
                                {new Date(batch.timestamp).toLocaleString()}
                              </div>
                            </div>
                            {batch.improvement_from_previous !== undefined && (
                              <div
                                className={`px-4 py-2 rounded-lg font-bold ${
                                  batch.improvement_from_previous > 0
                                    ? "bg-green-100 text-green-700"
                                    : batch.improvement_from_previous < 0
                                    ? "bg-red-100 text-red-700"
                                    : "bg-gray-100 text-gray-700"
                                }`}
                              >
                                {batch.improvement_from_previous > 0
                                  ? "↗️"
                                  : batch.improvement_from_previous < 0
                                  ? "↘️"
                                  : "➡️"}
                                {batch.improvement_from_previous > 0 ? "+" : ""}
                                {batch.improvement_from_previous.toFixed(2)}%
                              </div>
                            )}
                          </div>

                          <div className="grid grid-cols-4 gap-4">
                            <div className="bg-blue-50 rounded-lg p-3">
                              <div className="text-xs text-gray-600 mb-1">
                                Training Samples
                              </div>
                              <div className="text-2xl font-bold text-blue-600">
                                {batch.training_samples}
                              </div>
                            </div>
                            <div className="bg-green-50 rounded-lg p-3">
                              <div className="text-xs text-gray-600 mb-1">
                                Accuracy
                              </div>
                              <div className="text-2xl font-bold text-green-600">
                                {(getAccuracy(batch.benchmarks) * 100).toFixed(
                                  2
                                )}
                                %
                              </div>
                            </div>
                            <div className="bg-purple-50 rounded-lg p-3">
                              <div className="text-xs text-gray-600 mb-1">
                                Precision
                              </div>
                              <div className="text-2xl font-bold text-purple-600">
                                {(getPrecision(batch.benchmarks) * 100).toFixed(
                                  2
                                )}
                                %
                              </div>
                            </div>
                            <div className="bg-orange-50 rounded-lg p-3">
                              <div className="text-xs text-gray-600 mb-1">
                                F1-Score
                              </div>
                              <div className="text-2xl font-bold text-orange-600">
                                {(getF1(batch.benchmarks) * 100).toFixed(2)}%
                              </div>
                            </div>
                          </div>

                          {batch.multiple_runs_stats && (
                            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                              <div className="text-sm font-semibold text-gray-700 mb-2">
                                📊 5 Runs Statistics (for thesis)
                              </div>
                              <div className="grid grid-cols-2 gap-3 text-sm">
                                <div>
                                  <span className="text-gray-600">
                                    Accuracy:
                                  </span>{" "}
                                  <span className="font-semibold">
                                    {(
                                      batch.multiple_runs_stats.accuracy.mean *
                                      100
                                    ).toFixed(2)}
                                    % ±{" "}
                                    {(
                                      batch.multiple_runs_stats.accuracy.std *
                                      100
                                    ).toFixed(2)}
                                    %
                                  </span>
                                </div>
                                <div>
                                  <span className="text-gray-600">
                                    F1-Score:
                                  </span>{" "}
                                  <span className="font-semibold">
                                    {(
                                      batch.multiple_runs_stats.f1.mean * 100
                                    ).toFixed(2)}
                                    % ±{" "}
                                    {(
                                      batch.multiple_runs_stats.f1.std * 100
                                    ).toFixed(2)}
                                    %
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
                        <div className="bg-white rounded-xl p-6 border-2 border-gray-200">
                          <h3 className="font-bold text-xl mb-4 text-gray-900">
                            🔄 Batch-to-Batch Improvements
                          </h3>
                          <div className="space-y-3">
                            {history.improvement_stats.improvements.map(
                              (imp, idx) => (
                                <div
                                  key={idx}
                                  className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                                >
                                  <div className="flex items-center gap-4">
                                    <div className="text-4xl">→</div>
                                    <div>
                                      <div className="font-semibold text-gray-800">
                                        {imp.from_batch} → {imp.to_batch}
                                      </div>
                                      <div className="text-sm text-gray-600">
                                        Added {imp.added_samples} samples (
                                        {imp.prev_samples} → {imp.curr_samples})
                                      </div>
                                    </div>
                                  </div>
                                  <div className="text-right">
                                    <div
                                      className={`text-2xl font-bold ${
                                        imp.improvement_percent > 0
                                          ? "text-green-600"
                                          : "text-red-600"
                                      }`}
                                    >
                                      {imp.improvement_percent > 0 ? "+" : ""}
                                      {imp.improvement_percent.toFixed(2)}%
                                    </div>
                                    <div className="text-xs text-gray-600">
                                      {imp.prev_accuracy.toFixed(2)}% →{" "}
                                      {imp.curr_accuracy.toFixed(2)}%
                                    </div>
                                  </div>
                                </div>
                              )
                            )}
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
                <h2 className="text-3xl font-bold text-gray-900 mb-6">
                  🔮 Test Predictions
                </h2>

                <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl p-6 border-2 border-indigo-300">
                  <label className="block text-sm font-semibold mb-2 text-gray-700">
                    Enter Tweet or Text to Classify
                  </label>
                  <textarea
                    value={predictionText}
                    onChange={(e) => setPredictionText(e.target.value)}
                    placeholder="Enter a tweet or text to classify..."
                    className="w-full p-4 border-2 border-gray-300 rounded-lg h-32 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
                  />

                  <button
                    onClick={handlePredict}
                    disabled={
                      !predictionText.trim() || !status?.has_model || loading
                    }
                    className="w-full mt-4 px-8 py-4 bg-indigo-600 text-white rounded-lg font-bold text-lg hover:bg-indigo-700 disabled:bg-gray-300 transition-all shadow-lg hover:shadow-xl"
                  >
                    Predict
                  </button>

                  {!status?.has_model && (
                    <p className="text-red-600 mt-3 text-sm">
                      ⚠️ Train a model first
                    </p>
                  )}
                </div>

                {predictionResult && (
                  <div className="bg-white rounded-xl p-6 border-2 border-gray-200 shadow-lg">
                    <h3 className="font-bold text-2xl mb-6 text-gray-900">
                      Prediction Results
                    </h3>

                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-xl p-6 border-2 border-indigo-300">
                        <div className="text-sm text-gray-600 mb-2">
                          Predicted Class
                        </div>
                        <div className="text-5xl font-bold text-indigo-600">
                          {predictionResult.prediction}
                        </div>
                      </div>
                      <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6 border-2 border-green-300">
                        <div className="text-sm text-gray-600 mb-2">
                          Confidence
                        </div>
                        <div className="text-5xl font-bold text-green-600">
                          {(predictionResult.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    <div className="bg-gray-50 rounded-lg p-6 mb-6">
                      <div className="font-semibold mb-4 text-gray-900">
                        Class Probabilities
                      </div>
                      <div className="space-y-3">
                        {Object.entries(predictionResult.probabilities).map(
                          ([cls, prob]) => (
                            <div key={cls}>
                              <div className="flex justify-between text-sm mb-1">
                                <span className="font-semibold text-gray-700">
                                  Class {cls}
                                </span>
                                <span className="text-gray-600">
                                  {(prob * 100).toFixed(2)}%
                                </span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                                <div
                                  className="bg-gradient-to-r from-indigo-500 to-purple-500 h-3 rounded-full transition-all duration-500"
                                  style={{ width: `${prob * 100}%` }}
                                />
                              </div>
                            </div>
                          )
                        )}
                      </div>
                    </div>

                    <div className="bg-blue-50 rounded-lg p-4 border-l-4 border-blue-500">
                      <div className="text-sm text-gray-600 mb-1">
                        Processed Text
                      </div>
                      <div className="text-gray-800 italic">
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
