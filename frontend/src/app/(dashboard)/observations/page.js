"use client";

import { useState, useEffect } from "react";
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
    <>
      {/* Metrics Modal - Fixed positioning for content area only */}
      {isMetricsModalOpen && (
        <div
          className="fixed top-0 right-0 bottom-0 left-[20%] bg-black/50 backdrop-blur-xs z-50 flex items-center justify-center"
          onClick={() => setIsMetricsModalOpen(false)}
        >
          <MetricsModal
            isOpen={isMetricsModalOpen}
            onClose={() => setIsMetricsModalOpen(false)}
            benchmarks={benchmarks}
          />
        </div>
      )}

      <div className="h-screen bg-[#F8FAFC] pt-20 overflow-y-hidden">
        <div className="flex flex-col">
          {/* TOP BENTO */}
          <div className="flex flex-row items-stretch justify-center mt-8 mx-8 gap-8">
            {/* LEFT - Bar Chart */}
            <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-4/6 p-4 rounded-2xl">
              <ClimateImpactChart
                location={selectedLocation}
                days={selectedDays}
                onLocationChange={setSelectedLocation}
                onDaysChange={setSelectedDays}
              />
            </div>
            {/* RIGHT - Pie Chart */}
            <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-2/6 p-4 rounded-2xl">
              <SentimentDistributionChart
                location={selectedLocation}
                days={selectedDays}
              />
            </div>
          </div>

          {/* BOTTOM BENTO */}
          <div className="flex flex-row items-stretch justify-center mt-8 mx-8 gap-8 mb-8">
            {/* LEFT - Stats Cards */}
            <div className="flex flex-col w-1/5 gap-8">
              {/* FRAMEWORK ACCURACY */}
              <div className="flex-1">
                <FrameworkAccuracyCard
                  benchmarks={benchmarks}
                  loading={loading}
                  error={error}
                  onRetry={fetchBenchmarks}
                  onViewDetails={() => setIsMetricsModalOpen(true)}
                />
              </div>
            </div>
            {/* RIGHT - Area Chart */}
            <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-4/5 p-4 rounded-2xl">
              <ClimateTrendsChart
                location={selectedLocation}
                days={selectedDays}
              />
            </div>
          </div>

          {/* Location Info */}
          <div className="flex items-center justify-center pb-8">
            <p className="text-sm text-[#6B7280] font-medium tracking-wide">
              AS OF {getCurrentTimestamp().toUpperCase()}
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
