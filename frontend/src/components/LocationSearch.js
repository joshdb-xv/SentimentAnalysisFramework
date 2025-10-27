import { useState, useRef, useEffect } from "react";
import { MdMyLocation, MdSearch } from "react-icons/md";

// Location Search Component with API Integration
const LocationSearch = ({ value, onChange, placeholder, icon: Icon }) => {
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

  // Fetch locations from API with debouncing
  const searchLocations = async (query) => {
    if (!query || query.trim().length < 2) {
      setLocations([]);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/weather/search-location`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query.trim() }),
      });

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
      setError("Could not connect to weather service");
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
      }, 500); // Wait 500ms after user stops typing
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
    // WeatherAPI returns location objects, extract the name
    const locationName =
      typeof location === "string" ? location : location.name;
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
    // If there's already a search term, trigger search again
    if (searchTerm.trim().length >= 2) {
      searchLocations(searchTerm);
    }
  };

  return (
    <div className="relative max-w-xs" ref={dropdownRef}>
      <div className="relative flex items-center h-10">
        <Icon
          size={20}
          color="#0A3D91"
          className="absolute left-4 pointer-events-none"
        />
        <input
          type="text"
          value={searchTerm}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          placeholder={value || placeholder}
          className={`w-full h-full py-2 pl-12 pr-12 bg-[#E2E8F0] text-[#222222] text-sm font-semibold rounded-full focus:outline-none focus:ring-2 focus:ring-[#0A3D91] focus:ring-opacity-50 hover:bg-[#CBD5E1] transition-colors duration-200 ${
            value ? "placeholder-[#222222]" : "placeholder-[#64748B]"
          }`}
        />
        {loading ? (
          <div className="absolute right-4 w-4 h-4 border-2 border-[#64748B] border-t-transparent rounded-full animate-spin"></div>
        ) : (
          <MdSearch
            size={20}
            color="#64748B"
            className="absolute right-4 pointer-events-none"
          />
        )}
      </div>

      {/* Dropdown Results */}
      {isOpen && searchTerm.trim().length >= 2 && (
        <div className="absolute top-full mt-1 left-0 right-0 bg-white border border-[#E2E8F0] rounded-xl shadow-lg max-h-60 overflow-auto z-50">
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
          ) : locations.length > 0 ? (
            <>
              <div className="px-3 py-2 text-xs text-[#64748B] font-medium border-b border-gray-100">
                {locations.length} location(s) found
              </div>
              {locations.map((location, index) => {
                // Handle both string and object responses
                const locationName =
                  typeof location === "string" ? location : location.name;
                const locationDetails =
                  typeof location === "object"
                    ? `${location.region ? location.region + ", " : ""}${
                        location.country || ""
                      }`
                    : "";

                return (
                  <button
                    key={index}
                    onClick={() => handleSelect(location)}
                    className={`w-full text-left px-4 py-3 text-sm font-medium transition-colors duration-150 hover:bg-[#F1F5F9] focus:outline-none focus:bg-[#F1F5F9] ${
                      value === locationName
                        ? "bg-[#EEF2FF] text-[#0A3D91] font-semibold"
                        : "text-[#334155]"
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <MdMyLocation
                        size={16}
                        className="text-[#64748B] mt-0.5"
                      />
                      <div>
                        <div className="font-semibold">{locationName}</div>
                        {locationDetails && (
                          <div className="text-xs text-[#64748B]">
                            {locationDetails}
                          </div>
                        )}
                      </div>
                    </div>
                  </button>
                );
              })}
            </>
          ) : (
            <div className="px-4 py-8 text-center text-sm text-[#64748B]">
              <p className="font-medium">No locations found</p>
              <p className="text-xs mt-1">Try a different search term</p>
            </div>
          )}
        </div>
      )}

      {/* Search hint when empty */}
      {isOpen && searchTerm.trim().length < 2 && searchTerm.length > 0 && (
        <div className="absolute top-full mt-1 left-0 right-0 bg-white border border-[#E2E8F0] rounded-lg shadow-sm px-3 py-2 text-xs text-[#64748B]">
          Type at least 2 characters to search...
        </div>
      )}
    </div>
  );
};

export default LocationSearch;
