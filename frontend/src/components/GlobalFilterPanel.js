import { useState, useEffect, useRef } from "react";
import { MdMyLocation, MdSearch } from "react-icons/md";
import { IoChevronDown, IoClose } from "react-icons/io5";
import { HiAdjustmentsHorizontal } from "react-icons/hi2";

const GlobalFilterPanel = ({
  location,
  onLocationChange,
  days,
  onDaysChange,
  isOpen,
  onToggle,
  onClose,
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [locations, setLocations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const panelRef = useRef(null);
  const debounceTimer = useRef(null);

  const API_BASE_URL = "http://localhost:8000";

  // Close panel when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (panelRef.current && !panelRef.current.contains(event.target)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen, onClose]);

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

  const handleLocationSelect = (locationItem) => {
    const locationName =
      typeof locationItem === "string"
        ? locationItem
        : locationItem.name || locationItem;
    onLocationChange(locationName);
    setSearchTerm("");
    setLocations([]);
  };

  const handleLocationClear = () => {
    onLocationChange(null);
    setSearchTerm("");
    setLocations([]);
  };

  const daysOptions = [
    { label: "Last 7 days", value: 7 },
    { label: "Last 30 days", value: 30 },
    { label: "Last 6 months", value: 180 },
    { label: "Last year", value: 365 },
  ];

  const activeFiltersCount = (location ? 1 : 0) + (days !== 365 ? 1 : 0);

  // Get current time period label
  const getCurrentPeriodLabel = () => {
    const option = daysOptions.find((opt) => opt.value === days);
    return option ? option.label : "Custom period";
  };

  return (
    <div className="relative" ref={panelRef}>
      {/* Modern Filter Toggle Button */}
      <button
        onClick={onToggle}
        className="group relative flex items-center gap-3 px-5 py-2.5 bg-white hover:bg-gradient-to-r hover:from-[#111111] hover:via-[#1E293B] hover:to-[#0A3D91] text-[#1E293B] hover:text-white rounded-xl shadow-[0px_2px_8px_0px_rgba(30,41,59,0.15)] hover:shadow-[0px_4px_16px_0px_rgba(30,41,59,0.25)] transition-all duration-300 border border-[#E2E8F0] hover:border-transparent"
      >
        <HiAdjustmentsHorizontal
          size={20}
          className="transition-transform duration-300 group-hover:rotate-90"
        />
        <div className="flex flex-col items-start min-w-0">
          {location ? (
            <>
              <span className="text-xs font-medium opacity-70 tracking-wide">
                LOCATION
              </span>
              <span className="text-sm font-semibold tracking-wide truncate max-w-[200px]">
                {location}
              </span>
            </>
          ) : (
            <>
              <span className="text-sm font-semibold tracking-wide">
                FILTERS
              </span>
              {activeFiltersCount > 0 && (
                <span className="text-[10px] font-medium opacity-70">
                  {activeFiltersCount} active
                </span>
              )}
            </>
          )}
        </div>
        <IoChevronDown
          size={16}
          className={`transition-transform duration-300 flex-shrink-0 ${
            isOpen ? "rotate-180" : ""
          }`}
        />

        {/* Active filter badge */}
        {activeFiltersCount > 0 && (
          <div className="absolute -top-1.5 -right-1.5 bg-[#F87171] text-white text-[10px] font-bold rounded-full w-5 h-5 flex items-center justify-center shadow-lg">
            {activeFiltersCount}
          </div>
        )}
      </button>

      {/* Filter Panel with Glassmorphism */}
      {isOpen && (
        <div className="absolute top-full right-0 mt-3 w-full sm:w-[420px] max-w-[95vw] rounded-2xl overflow-hidden z-50 animate-slideDown shadow-[0px_20px_60px_0px_rgba(30,41,59,0.35)]">
          {/* Glassmorphism backdrop */}
          <div className="absolute inset-0 bg-white/70 backdrop-blur-xl border border-white/20"></div>

          {/* Content */}
          <div className="relative">
            {/* Header with gradient */}
            <div className="relative px-5 py-4 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] overflow-hidden">
              {/* Decorative circles */}
              <div className="absolute top-0 right-0 w-32 h-32 bg-white/5 rounded-full -mr-16 -mt-16"></div>
              <div className="absolute bottom-0 left-0 w-24 h-24 bg-white/5 rounded-full -ml-12 -mb-12"></div>

              <div className="relative flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-white/10 rounded-lg backdrop-blur-sm">
                    <HiAdjustmentsHorizontal size={20} className="text-white" />
                  </div>
                  <div>
                    <h3 className="font-bold text-white text-lg tracking-wide">
                      Filter Options
                    </h3>
                    <p className="text-xs text-white/70 mt-0.5">
                      Customize your data view
                    </p>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="text-white/80 hover:text-white hover:bg-white/10 p-2 rounded-lg transition-all duration-200"
                >
                  <IoClose size={22} />
                </button>
              </div>
            </div>

            {/* Dynamic content area */}
            <div className="p-5 space-y-6 max-h-[70vh] sm:max-h-[500px] overflow-y-auto">
              {/* Location Filter */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <label className="flex items-center gap-2 text-sm font-bold text-[#1E293B] tracking-wide">
                    <div className="w-1 h-4 bg-gradient-to-b from-[#0A3D91] to-[#1E293B] rounded-full"></div>
                    LOCATION
                  </label>
                  {location && (
                    <span className="text-xs text-[#0A3D91] font-medium">
                      1 selected
                    </span>
                  )}
                </div>

                {/* Current Location Display or Search */}
                {location ? (
                  <div className="flex items-center justify-between bg-gradient-to-r from-[#EEF2FF] to-[#E0E7FF] border border-[#0A3D91]/20 rounded-xl px-4 py-3 group hover:shadow-md transition-all duration-200">
                    <div className="flex items-center gap-2 min-w-0">
                      <div className="p-1.5 bg-white rounded-lg flex-shrink-0">
                        <MdMyLocation size={16} className="text-[#0A3D91]" />
                      </div>
                      <span className="text-sm font-semibold text-[#0A3D91] truncate">
                        {location}
                      </span>
                    </div>
                    <button
                      onClick={handleLocationClear}
                      className="text-[#F87171] hover:text-red-700 hover:bg-white/50 p-1.5 rounded-lg transition-all duration-200 flex-shrink-0"
                    >
                      <IoClose size={18} />
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="relative">
                      <MdSearch
                        size={18}
                        color="#64748B"
                        className="absolute left-4 top-1/2 -translate-y-1/2 pointer-events-none z-10"
                      />
                      <input
                        type="text"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        placeholder="Search locations..."
                        className="w-full py-3 pl-11 pr-4 bg-white/60 backdrop-blur-sm text-[#222222] text-sm font-medium rounded-xl border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#0A3D91] focus:ring-opacity-30 focus:border-[#0A3D91] transition-all duration-200 placeholder:text-[#94A3B8]"
                      />
                    </div>

                    {/* Search Results */}
                    {searchTerm.trim().length >= 2 && (
                      <div className="mt-2 max-h-56 overflow-auto border border-[#E2E8F0] rounded-xl bg-white/80 backdrop-blur-sm shadow-inner">
                        {loading ? (
                          <div className="px-4 py-8 text-center text-sm text-[#64748B]">
                            <div className="inline-block w-5 h-5 border-2 border-[#0A3D91] border-t-transparent rounded-full animate-spin mb-2"></div>
                            <p>Searching...</p>
                          </div>
                        ) : error ? (
                          <div className="px-4 py-6 text-center text-sm text-red-500">
                            <p className="font-medium">{error}</p>
                          </div>
                        ) : locations.length > 0 ? (
                          <>
                            <div className="px-4 py-2.5 text-xs text-[#64748B] font-semibold bg-white/60 backdrop-blur-sm border-b border-[#E2E8F0] sticky top-0">
                              {locations.length} location(s) found
                            </div>
                            {locations.map((loc, index) => {
                              const locationName =
                                typeof loc === "string" ? loc : loc.name || loc;
                              return (
                                <button
                                  key={index}
                                  onClick={() => handleLocationSelect(loc)}
                                  className={`w-full text-left px-4 py-3 text-sm font-medium transition-all duration-200 border-b border-[#E2E8F0] last:border-b-0 ${
                                    location === locationName
                                      ? "bg-gradient-to-r from-[#EEF2FF] to-[#E0E7FF] text-[#0A3D91] font-semibold"
                                      : "text-[#334155] hover:bg-white/40"
                                  }`}
                                >
                                  <div className="flex items-center gap-2">
                                    <MdMyLocation
                                      size={14}
                                      className={
                                        location === locationName
                                          ? "text-[#0A3D91]"
                                          : "text-[#64748B]"
                                      }
                                    />
                                    <span className="truncate">
                                      {locationName}
                                    </span>
                                  </div>
                                </button>
                              );
                            })}
                          </>
                        ) : (
                          <div className="px-4 py-8 text-center text-sm text-[#64748B]">
                            <p className="font-medium">No locations found</p>
                            <p className="text-xs mt-1">
                              Try a different search term
                            </p>
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Time Period Filter */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <label className="flex items-center gap-2 text-sm font-bold text-[#1E293B] tracking-wide">
                    <div className="w-1 h-4 bg-gradient-to-b from-[#0A3D91] to-[#1E293B] rounded-full"></div>
                    TIME PERIOD
                  </label>
                  <span className="text-xs text-[#64748B] font-medium">
                    {getCurrentPeriodLabel()}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2.5">
                  {daysOptions.map((option) => (
                    <button
                      key={option.value}
                      onClick={() => onDaysChange(option.value)}
                      className={`relative px-4 py-3.5 text-sm font-semibold rounded-xl transition-all duration-300 overflow-hidden group ${
                        days === option.value
                          ? "bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] text-white shadow-lg scale-105"
                          : "bg-white/60 backdrop-blur-sm text-[#334155] hover:bg-white/80 border border-[#E2E8F0] hover:border-[#CBD5E1] hover:shadow-md"
                      }`}
                    >
                      {/* Shine effect for selected */}
                      {days === option.value && (
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent transform -skew-x-12 animate-shine"></div>
                      )}
                      <span className="relative z-10">{option.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Clear All Filters */}
              {activeFiltersCount > 0 && (
                <div className="pt-4 border-t border-white/30">
                  <button
                    onClick={() => {
                      handleLocationClear();
                      onDaysChange(365);
                    }}
                    className="w-full py-3 text-sm font-bold text-red-600 hover:text-white bg-red-50/80 backdrop-blur-sm hover:bg-gradient-to-r hover:from-[#F87171] hover:to-[#EF4444] rounded-xl transition-all duration-300 border border-red-200 hover:border-transparent hover:shadow-lg"
                  >
                    Clear all filters
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes shine {
          0% {
            left: -100%;
          }
          100% {
            left: 200%;
          }
        }

        .animate-slideDown {
          animation: slideDown 0.3s ease-out;
        }

        .animate-shine {
          animation: shine 2s ease-in-out infinite;
        }

        /* Custom scrollbar for glassmorphism */
        .overflow-y-auto::-webkit-scrollbar,
        .overflow-auto::-webkit-scrollbar {
          width: 6px;
        }

        .overflow-y-auto::-webkit-scrollbar-track,
        .overflow-auto::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 3px;
        }

        .overflow-y-auto::-webkit-scrollbar-thumb,
        .overflow-auto::-webkit-scrollbar-thumb {
          background: rgba(30, 41, 59, 0.3);
          border-radius: 3px;
        }

        .overflow-y-auto::-webkit-scrollbar-thumb:hover,
        .overflow-auto::-webkit-scrollbar-thumb:hover {
          background: rgba(30, 41, 59, 0.5);
        }
      `}</style>
    </div>
  );
};

export default GlobalFilterPanel;
