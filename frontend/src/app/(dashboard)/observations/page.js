"use client";

import { useState } from "react";
import { MdChevronRight } from "react-icons/md";
import ClimateImpactChart from "@/components/BarChart";
import SentimentDistributionChart from "@/components/PieChart";
import ClimateTrendsChart from "@/components/AreaChart";
import GlobalFilterPanel from "@/components/GlobalFilterPanel"; // Import the new component

export default function Observations() {
  // Shared state for all filters - lifted up to parent
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [selectedDays, setSelectedDays] = useState(365); // Global time period
  const [isFilterPanelOpen, setIsFilterPanelOpen] = useState(false);

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
              days={selectedDays} // Pass global days
            />
          </div>
          {/* RIGHT - Pie Chart */}
          <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-2/6 h-full p-4 rounded-2xl">
            <SentimentDistributionChart
              location={selectedLocation}
              days={selectedDays} // Pass global days
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
            <div className="flex flex-col justify-center items-center mt-4 gap-2">
              <p className="text-4xl font-extrabold bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent py-1">
                84.6%
              </p>
              <p className="text-sm text-[#6B7280] font-medium">
                End-to-end pipeline validation
              </p>
            </div>
          </div>
        </div>
        {/* RIGHT - Area Chart */}
        <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-4/5 h-full p-4 rounded-2xl">
          <ClimateTrendsChart
            location={selectedLocation}
            days={selectedDays} // Pass global days
          />
        </div>
      </div>
    </div>
  );
}
