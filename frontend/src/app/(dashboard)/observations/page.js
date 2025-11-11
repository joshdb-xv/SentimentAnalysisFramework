"use client";

import { useState, useEffect } from "react";
import { MdChevronRight } from "react-icons/md";
import ClimateImpactChart from "@/components/observations/BarChart";
import SentimentDistributionChart from "@/components/observations/PieChart";
import ClimateTrendsChart from "@/components/observations/AreaChart";
import MetricsModal from "@/components/observations/MetricsModal";
import FrameworkAccuracyCard from "@/components/observations/FrameworkAccuracyCard";

export default function Observations() {
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [selectedDays, setSelectedDays] = useState(365);
  const [benchmarks, setBenchmarks] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isMetricsModalOpen, setIsMetricsModalOpen] = useState(false);

  // Fetch benchmarks on component mount
  useEffect(() => {
    fetchBenchmarks();
  }, []);

  const fetchBenchmarks = async () => {
    try {
      setLoading(true);
      const response = await fetch("http://localhost:8000/benchmarks/all");

      if (!response.ok) {
        throw new Error("Failed to fetch benchmarks");
      }

      const data = await response.json();
      setBenchmarks(data);
      setError(null);
    } catch (err) {
      console.error("Error fetching benchmarks:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getCurrentTimestamp = () => {
    const now = new Date();
    return now.toLocaleString("en-US", {
      month: "2-digit",
      day: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    });
  };

  return (
    <div className="flex flex-col h-screen bg-[#F8FAFC] relative">
      {/* Metrics Modal - Positioned relative to parent container */}
      {isMetricsModalOpen && (
        <div
          className="absolute inset-0 bg-black/50 backdrop-blur-xs z-50 flex items-center justify-center"
          onClick={() => setIsMetricsModalOpen(false)}
        >
          <MetricsModal
            isOpen={isMetricsModalOpen}
            onClose={() => setIsMetricsModalOpen(false)}
            benchmarks={benchmarks}
          />
        </div>
      )}

      {/* HEADER */}
      <div className="px-6 w-full h-20 flex items-center justify-between border-b-2 border-primary-dark/10">
        <div className="flex items-center gap-4">
          <div className="w-fit text-2xl font-bold bg-gradient-to-r from-[#222222] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent leading-snug">
            Sentiment Analysis Framework
          </div>
          <MdChevronRight className="text-3xl text-gray-400" />
          <p className="text-gray-600 text-2xl font-medium">Observations</p>
        </div>
      </div>

      <div className="flex flex-col">
        {/* TOP BENTO */}
        <div className="flex flex-row items-start justify-center mt-8 mx-8 gap-8 h-auto">
          {/* LEFT - Bar Chart */}
          <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-4/6 h-full p-4 rounded-2xl">
            <ClimateImpactChart
              location={selectedLocation}
              days={selectedDays}
              onLocationChange={setSelectedLocation}
              onDaysChange={setSelectedDays}
            />
          </div>
          {/* RIGHT - Pie Chart */}
          <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-2/6 h-full p-4 rounded-2xl">
            <SentimentDistributionChart
              location={selectedLocation}
              days={selectedDays}
            />
          </div>
        </div>
      </div>

      {/* BOTTOM BENTO */}
      <div className="flex flex-row items-start justify-center mt-8 mx-8 gap-8 h-auto">
        {/* LEFT - Stats Cards */}
        <div className="flex flex-col w-1/5 h-full gap-8">
          {/* FRAMEWORK ACCURACY */}
          <FrameworkAccuracyCard
            benchmarks={benchmarks}
            loading={loading}
            error={error}
            onRetry={fetchBenchmarks}
            onViewDetails={() => setIsMetricsModalOpen(true)}
          />
        </div>
        {/* RIGHT - Area Chart */}
        <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-4/5 h-full p-4 rounded-2xl">
          <ClimateTrendsChart location={selectedLocation} days={selectedDays} />
        </div>
      </div>

      {/* Location Info */}
      <div className="flex items-center justify-center mt-6">
        <p className="text-sm text-[#6B7280] font-medium tracking-wide">
          AS OF {getCurrentTimestamp().toUpperCase()}
        </p>
      </div>
    </div>
  );
}
