"use client";

import { useState, useEffect, useRef } from "react";
import { MdMyLocation } from "react-icons/md";
import { IoChevronBack, IoChevronForward } from "react-icons/io5";

export default function ClimateImpactChart({ location, days }) {
  const [hoveredData, setHoveredData] = useState(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 5;
  const filterRef = useRef(null);

  const API_BASE_URL = "http://localhost:8000";

  // Fetch category data from backend
  useEffect(() => {
    fetchCategoryData();
  }, [location, days]);

  // Initial fetch on mount
  useEffect(() => {
    fetchCategoryData();
  }, []);

  // Reset to first page when data changes
  useEffect(() => {
    setCurrentPage(1);
  }, [chartData]);

  const fetchCategoryData = async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (location) params.append("location", location);
      params.append("days", days);

      const response = await fetch(
        `${API_BASE_URL}/observations/categories?${params.toString()}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.status === "ok" && data.categories) {
        const formattedData = data.categories.map((cat) => ({
          category: cat.name,
          positive: cat.positive || 0,
          neutral: cat.neutral || 0,
          negative: cat.negative || 0,
          total: cat.total,
        }));

        setChartData(formattedData);
      } else {
        throw new Error(data.error || "Failed to fetch category data");
      }
    } catch (err) {
      console.error("Error fetching category data:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleMouseEnter = (item, type, event) => {
    const value = item[type];
    const percentage = ((value / item.total) * 100).toFixed(1);

    setHoveredData({
      category: item.category,
      type: type,
      percentage: percentage,
      value: value,
      total: item.total,
    });
    setMousePosition({ x: event.clientX, y: event.clientY });
  };

  const handleMouseLeave = () => {
    setHoveredData(null);
  };

  const handleMouseMove = (event) => {
    if (hoveredData) {
      setMousePosition({ x: event.clientX, y: event.clientY });
    }
  };

  const calculatePercentage = (value, total) => {
    return (value / total) * 100;
  };

  // Pagination calculations
  const totalPages = Math.ceil(chartData.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentData = chartData.slice(startIndex, endIndex);

  const goToNextPage = () => {
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
    }
  };

  const goToPreviousPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  const goToPage = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  return (
    <div className="">
      {/* Tooltip */}
      {hoveredData && (
        <div
          className="fixed bg-gray-800 text-white text-xs rounded-lg px-3 py-2 pointer-events-none z-50 shadow-lg"
          style={{
            left: mousePosition.x + 10,
            top: mousePosition.y - 10,
            transform: "translate(0, -100%)",
          }}
        >
          <div className="font-semibold">{hoveredData.category}</div>
          <div className="capitalize text-gray-200">
            {hoveredData.type}: {hoveredData.value} ({hoveredData.percentage}%)
          </div>
          <div className="text-gray-300 text-xs">
            Total: {hoveredData.total}
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
        <p className="font-medium text-xl text-[#1E293B] tracking-wide">
          CLIMATE CHANGE IMPACT CATEGORIES
        </p>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-end gap-14 mt-2 mb-4 mr-13">
        <div className="relative" ref={filterRef}></div>
        <div className="flex gap-12">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-[#7DD3FC] rounded-full"></div>
            <span className="text-[#6B7280] font-medium text-sm">POSITIVE</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-[#94A3B8] rounded-full"></div>
            <span className="text-[#6B7280] font-medium text-sm">NEUTRAL</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-[#F87171] rounded-full"></div>
            <span className="text-[#6B7280] font-medium text-sm">NEGATIVE</span>
          </div>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#0A3D91] mx-auto mb-4"></div>
            <p className="text-[#6B7280]">Loading chart data...</p>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <p className="text-red-500 mb-4">Error: {error}</p>
            <button
              onClick={fetchCategoryData}
              className="px-4 py-2 bg-[#0A3D91] text-white rounded-lg hover:bg-[#1E293B]"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Chart */}
      {!loading && !error && (
        <>
          {chartData.length > 0 ? (
            <>
              <div className="space-y-5 min-h-56" onMouseMove={handleMouseMove}>
                {currentData.map((item, index) => (
                  <div
                    key={startIndex + index}
                    className="flex items-center gap-4"
                  >
                    {/* Category Label */}
                    <div className="flex-1 text-lg font-semibold text-[#1E293B] text-left tracking-wide flex-shrink-0">
                      {item.category}
                    </div>

                    {/* Total Count */}
                    <div className="text-lg font-semibold text-[#6B7280] w-12 text-right flex-shrink-0">
                      {item.total}
                    </div>

                    {/* Progress Bar */}
                    <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden flex">
                      <div
                        className="bg-[#7DD3FC] h-full transition-all duration-200 hover:brightness-115 cursor-pointer"
                        style={{
                          width: `${calculatePercentage(
                            item.positive,
                            item.total
                          )}%`,
                        }}
                        onMouseEnter={(e) =>
                          handleMouseEnter(item, "positive", e)
                        }
                        onMouseLeave={handleMouseLeave}
                      ></div>
                      <div
                        className="bg-[#94A3B8] h-full transition-all duration-200 hover:brightness-115 cursor-pointer"
                        style={{
                          width: `${calculatePercentage(
                            item.neutral,
                            item.total
                          )}%`,
                        }}
                        onMouseEnter={(e) =>
                          handleMouseEnter(item, "neutral", e)
                        }
                        onMouseLeave={handleMouseLeave}
                      ></div>
                      <div
                        className="bg-[#F87171] h-full transition-all duration-200 hover:brightness-115 cursor-pointer"
                        style={{
                          width: `${calculatePercentage(
                            item.negative,
                            item.total
                          )}%`,
                        }}
                        onMouseEnter={(e) =>
                          handleMouseEnter(item, "negative", e)
                        }
                        onMouseLeave={handleMouseLeave}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Pagination Controls */}
              <div className="flex items-center justify-center gap-4 mt-6">
                <button
                  onClick={goToPreviousPage}
                  disabled={currentPage === 1}
                  className={`p-2 rounded-lg transition-all ${
                    currentPage === 1
                      ? "text-gray-300 cursor-not-allowed"
                      : "text-[#1E293B] hover:bg-[#E2E8F0]"
                  }`}
                >
                  <IoChevronBack size={20} />
                </button>

                <div className="flex gap-2">
                  {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                    (pageNum) => (
                      <button
                        key={pageNum}
                        onClick={() => goToPage(pageNum)}
                        className={`w-10 h-10 rounded-lg font-semibold transition-all ${
                          currentPage === pageNum
                            ? "bg-[#0A3D91] text-white"
                            : "text-[#1E293B] hover:bg-[#E2E8F0]"
                        }`}
                      >
                        {pageNum}
                      </button>
                    )
                  )}
                </div>

                <button
                  onClick={goToNextPage}
                  disabled={currentPage === totalPages}
                  className={`p-2 rounded-lg transition-all ${
                    currentPage === totalPages
                      ? "text-gray-300 cursor-not-allowed"
                      : "text-[#1E293B] hover:bg-[#E2E8F0]"
                  }`}
                >
                  <IoChevronForward size={20} />
                </button>
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-64">
              <p className="text-[#6B7280]">
                No climate data available for the selected period.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
