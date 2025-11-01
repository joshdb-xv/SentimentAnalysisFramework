"use client";

import { useState, useEffect } from "react";
import { MdChevronRight } from "react-icons/md";
import ClimateImpactChart from "@/components/BarChart";
import SentimentDistributionChart from "@/components/PieChart";
import ClimateTrendsChart from "@/components/AreaChart";
import GlobalFilterPanel from "@/components/GlobalFilterPanel";

export default function Observations() {
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [selectedDays, setSelectedDays] = useState(365);
  const [isFilterPanelOpen, setIsFilterPanelOpen] = useState(false);
  const [benchmarks, setBenchmarks] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

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
    <div className="flex flex-col h-screen bg-[#F8FAFC]">
      {/* HEADER */}
      <div className="px-6 w-full h-20 flex items-center justify-between border-b-2 border-primary-dark/10">
        <div className="flex items-center gap-4">
          <div className="w-fit text-2xl font-bold bg-gradient-to-r from-[#222222] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent leading-snug">
            Sentiment Analysis Framework
          </div>
          <MdChevronRight className="text-3xl text-gray-400" />
          <p className="text-gray-600 text-2xl font-medium">Observations</p>
        </div>

        {/* Global Filter Button - Top Right */}
        <GlobalFilterPanel
          location={selectedLocation}
          onLocationChange={setSelectedLocation}
          days={selectedDays}
          onDaysChange={setSelectedDays}
          isOpen={isFilterPanelOpen}
          onToggle={() => setIsFilterPanelOpen(!isFilterPanelOpen)}
          onClose={() => setIsFilterPanelOpen(false)}
        />
      </div>

      <div className="flex flex-col">
        {/* TOP BENTO */}
        <div className="flex flex-row items-start justify-center mt-8 mx-8 gap-8 h-auto">
          {/* LEFT - Bar Chart */}
          <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-4/6 h-full p-4 rounded-2xl">
            <ClimateImpactChart
              location={selectedLocation}
              days={selectedDays}
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
          <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-full h-full p-4 rounded-2xl">
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
              <p className="font-medium text-[#1E293B] tracking-wide text-lg">
                FRAMEWORK ACCURACY
              </p>
            </div>

            {/* ACCURACY SCORES */}
            <div className="flex flex-col items-center justify-center gap-5">
              {/* VADER - Sentiment Identifier - STATIC */}
              <div className="flex flex-col justify-center items-center mt-4 gap-2">
                <p className="text-4xl font-extrabold bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent py-1">
                  81.58%
                </p>
                <p className="text-base text-[#1E293B] text-center font-semibold">
                  VADER - Tweet Sentiment Identifier
                </p>
              </div>

              {/* Naive Bayes - Climate Related Checker - DYNAMIC */}
              <div className="flex flex-col justify-center items-center mt-4 gap-2">
                {loading ? (
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#0A3D91]"></div>
                ) : error ? (
                  <div className="flex flex-col items-center gap-1">
                    <p className="text-xl text-red-500">Error</p>
                    <button
                      onClick={fetchBenchmarks}
                      className="text-xs text-[#0A3D91] hover:underline"
                    >
                      Retry
                    </button>
                  </div>
                ) : benchmarks ? (
                  <p className="text-4xl font-extrabold bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent py-1">
                    {benchmarks.naive_bayes_climate_checker.toFixed(2)}%
                  </p>
                ) : null}
                <p className="text-base text-[#1E293B] text-center font-semibold">
                  Naive Bayes - Climate Related Checker
                </p>
              </div>

              {/* Naive Bayes - Climate Domain Identifier */}
              <div className="flex flex-col justify-center items-center mt-4 gap-2">
                {loading ? (
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#0A3D91]"></div>
                ) : error ? (
                  <div className="flex flex-col items-center gap-1">
                    <p className="text-xl text-red-500">Error</p>
                    <button
                      onClick={fetchBenchmarks}
                      className="text-xs text-[#0A3D91] hover:underline"
                    >
                      Retry
                    </button>
                  </div>
                ) : benchmarks &&
                  benchmarks.naive_bayes_domain_identifier !== null ? (
                  <p className="text-4xl font-extrabold bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent py-1">
                    {benchmarks.naive_bayes_domain_identifier.toFixed(2)}%
                  </p>
                ) : null}
                <p className="text-base text-[#1E293B] text-center font-semibold">
                  Naive Bayes - Climate Domain Identifier
                </p>
              </div>
            </div>
          </div>
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
