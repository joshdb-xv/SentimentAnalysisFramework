"use client";

import { useState, useEffect, useRef } from "react";
import { MdMyLocation, MdSearch } from "react-icons/md";
import {
  IoChevronDown,
  IoFilter,
  IoChevronBack,
  IoChevronForward,
} from "react-icons/io5";

// Location Search Component - Modified for Database Locations
const LocationSearch = ({ value, onChange, placeholder }) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [locations, setLocations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const dropdownRef = useRef(null);
  const debounceTimer = useRef(null);

  const API_BASE_URL = "http://localhost:8000";

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Fetch locations from database
  const searchLocations = async (query) => {
    if (!query || query.trim().length < 2) {
      setLocations([]);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/observations/search-locations?search=${encodeURIComponent(
          query.trim()
        )}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.status === "ok" && data.locations) {
        setLocations(data.locations);
      } else {
        setError(data.error || "Failed to fetch locations");
        setLocations([]);
      }
    } catch (err) {
      console.error("Error searching locations:", err);
      setError("Could not connect to server");
      setLocations([]);
    } finally {
      setLoading(false);
    }
  };

  // Debounced search
  useEffect(() => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    if (searchTerm.trim().length >= 2) {
      debounceTimer.current = setTimeout(() => {
        searchLocations(searchTerm);
      }, 500);
    } else {
      setLocations([]);
    }

    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, [searchTerm]);

  const handleSelect = (location) => {
    const locationName =
      typeof location === "string" ? location : location.name || location;
    onChange(locationName);
    setSearchTerm("");
    setLocations([]);
    setIsOpen(false);
  };

  const handleInputChange = (e) => {
    setSearchTerm(e.target.value);
    setIsOpen(true);
  };

  const handleInputFocus = () => {
    setIsOpen(true);
    if (searchTerm.trim().length >= 2) {
      searchLocations(searchTerm);
    }
  };

  const handleClear = () => {
    onChange(null);
    setSearchTerm("");
    setLocations([]);
    setIsOpen(false);
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={handleInputFocus}
        className="flex flex-row gap-2 bg-[#E2E8F0] px-4 py-1 rounded-full items-center justify-center cursor-pointer shadow-[0px_2px_8px_0px_rgba(30,41,59,0.25)] hover:bg-[#CBD5E1] transition-colors"
      >
        <MdMyLocation color="#0A3D91" />
        <p className="text-[#1E293B] text-lg font-medium">
          {value || placeholder}
        </p>
        <IoChevronDown className="ml-2" color="#1E293B" />
      </button>

      {/* Search Dropdown */}
      {isOpen && (
        <div className="absolute top-full mt-2 left-0 min-w-[320px] bg-white border border-[#E2E8F0] rounded-xl shadow-lg z-50">
          {/* Search Input */}
          <div className="p-3 border-b border-[#E2E8F0]">
            <div className="relative flex items-center">
              <MdSearch
                size={20}
                color="#64748B"
                className="absolute left-3 pointer-events-none"
              />
              <input
                type="text"
                value={searchTerm}
                onChange={handleInputChange}
                placeholder="Search locations..."
                autoFocus
                className="w-full py-2 pl-10 pr-3 bg-[#F8FAFC] text-[#222222] text-sm font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-[#0A3D91] focus:ring-opacity-50"
              />
            </div>
          </div>

          {/* Results */}
          <div className="max-h-60 overflow-auto">
            {value && (
              <button
                onClick={handleClear}
                className="w-full text-left px-4 py-3 text-sm font-medium border-b border-[#E2E8F0] hover:bg-[#FEF2F2] text-red-600 transition-colors"
              >
                Clear selection
              </button>
            )}

            {loading ? (
              <div className="px-4 py-8 text-center text-sm text-[#64748B]">
                <div className="inline-block w-6 h-6 border-2 border-[#0A3D91] border-t-transparent rounded-full animate-spin mb-2"></div>
                <p>Searching locations...</p>
              </div>
            ) : error ? (
              <div className="px-4 py-8 text-center text-sm text-red-500">
                <p className="font-medium">{error}</p>
                <p className="text-xs mt-1 text-[#64748B]">
                  Make sure your backend is running
                </p>
              </div>
            ) : searchTerm.trim().length >= 2 && locations.length > 0 ? (
              <>
                <div className="px-3 py-2 text-xs text-[#64748B] font-medium bg-[#F8FAFC]">
                  {locations.length} location(s) found
                </div>
                {locations.map((location, index) => {
                  const locationName =
                    typeof location === "string"
                      ? location
                      : location.name || location;

                  return (
                    <button
                      key={index}
                      onClick={() => handleSelect(location)}
                      className={`w-full text-left px-4 py-3 text-sm font-medium transition-colors hover:bg-[#F1F5F9] focus:outline-none ${
                        value === locationName
                          ? "bg-[#EEF2FF] text-[#0A3D91] font-semibold"
                          : "text-[#334155]"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <MdMyLocation size={16} className="text-[#64748B]" />
                        <span>{locationName}</span>
                      </div>
                    </button>
                  );
                })}
              </>
            ) : searchTerm.trim().length >= 2 ? (
              <div className="px-4 py-8 text-center text-sm text-[#64748B]">
                <p className="font-medium">No locations found</p>
                <p className="text-xs mt-1">Try a different search term</p>
              </div>
            ) : (
              <div className="px-4 py-8 text-center text-sm text-[#64748B]">
                <p>Type at least 2 characters to search</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default function ClimateImpactChart({ location, onLocationChange }) {
  const [hoveredData, setHoveredData] = useState(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // const [location, setLocation] = useState(null);
  const [days, setDays] = useState(365); // Increased to 90 days to capture more data
  const [currentPage, setCurrentPage] = useState(1);
  const [showFilterDropdown, setShowFilterDropdown] = useState(false);
  const itemsPerPage = 5;
  const filterRef = useRef(null);

  const API_BASE_URL = "http://localhost:8000";

  // Close filter dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (filterRef.current && !filterRef.current.contains(event.target)) {
        setShowFilterDropdown(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

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

  const handleDaysChange = (newDays) => {
    setDays(newDays);
    setShowFilterDropdown(false);
  };

  const daysOptions = [
    { label: "Last 7 days", value: 7 },
    { label: "Last 30 days", value: 30 },
    { label: "Last 6 months", value: 180 },
    { label: "Last year", value: 365 },
  ];

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

      {/* Location and Date */}
      <div className="flex items-center gap-4 text-sm text-[#6B7280] ml-5 mt-4">
        <LocationSearch
          value={location}
          onChange={onLocationChange} // Use the prop function instead
          placeholder="All Locations"
        />
        <p className="font-medium tracking-wide text-[#6B7280]">
          AS OF {getCurrentTimestamp().toUpperCase()}
        </p>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-end gap-14 mt-2 mb-4 mr-13">
        <div className="relative" ref={filterRef}>
          <IoFilter
            size={20}
            color={"#6B7280"}
            className="cursor-pointer hover:text-[#0A3D91] transition-colors"
            onClick={() => setShowFilterDropdown(!showFilterDropdown)}
            title="Filter by time period"
          />

          {/* Filter Dropdown */}
          {showFilterDropdown && (
            <div className="absolute top-full right-0 mt-2 bg-white border border-[#E2E8F0] rounded-xl shadow-lg z-50 min-w-[180px]">
              <div className="px-3 py-2 text-xs text-[#64748B] font-medium border-b border-[#E2E8F0] bg-[#F8FAFC]">
                Time Period
              </div>
              {daysOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => handleDaysChange(option.value)}
                  className={`w-full text-left px-4 py-3 text-sm font-medium transition-colors hover:bg-[#F1F5F9] focus:outline-none ${
                    days === option.value
                      ? "bg-[#EEF2FF] text-[#0A3D91] font-semibold"
                      : "text-[#334155]"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          )}
        </div>
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
              <div className="space-y-4 min-h-56" onMouseMove={handleMouseMove}>
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
                    <div className="text-base font-semibold text-[#6B7280] w-12 text-right flex-shrink-0">
                      {item.total}
                    </div>

                    {/* Progress Bar */}
                    <div className="flex-1 h-5 bg-gray-100 rounded-full overflow-hidden flex">
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
              <div className="flex items-center justify-center gap-4">
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

              {/* Page Info
              {totalPages > 1 && (
                <div className="mt-4 text-center text-sm text-[#6B7280]">
                  Showing {startIndex + 1}-
                  {Math.min(endIndex, chartData.length)} of {chartData.length}{" "}
                  categories
                </div>
              )} */}
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
