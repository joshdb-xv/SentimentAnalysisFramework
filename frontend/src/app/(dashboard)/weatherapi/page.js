"use client";

import { useState, useEffect } from "react";
import { MdChevronRight, MdMyLocation, MdRefresh } from "react-icons/md";
import {
  WiDaySunny,
  WiNightClear,
  WiDayCloudy,
  WiCloudy,
  WiRain,
  WiThunderstorm,
  WiSnow,
  WiFog,
  WiDayRain,
  WiNightRain,
  WiDayThunderstorm,
  WiNightThunderstorm,
  WiDaySnow,
  WiNightSnow,
  WiDayFog,
  WiNightFog,
} from "react-icons/wi";
import LocationSearch from "@/components/LocationSearch";

export default function WeatherAPIClient() {
  const [currentWeather, setCurrentWeather] = useState(null);
  const [hourlyForecast, setHourlyForecast] = useState(null);
  const [weeklyForecast, setWeeklyForecast] = useState(null);
  const [location, setLocation] = useState("Indang");
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const API_BASE_URL = "http://localhost:8000";

  // Function to map weather conditions to React Icons
  const getWeatherIcon = (condition, isDay = true, size = 144) => {
    if (!condition)
      return (
        <WiDaySunny size={size} className="text-yellow-500 drop-shadow-lg" />
      );

    const conditionLower = condition.toLowerCase();
    const baseClass = "drop-shadow-lg";

    // Sunny/Clear
    if (conditionLower.includes("sunny") || conditionLower.includes("clear")) {
      return isDay ? (
        <WiDaySunny size={size} className={`text-yellow-500 ${baseClass}`} />
      ) : (
        <WiNightClear size={size} className={`text-blue-300 ${baseClass}`} />
      );
    }

    // Partly Cloudy
    if (
      conditionLower.includes("partly cloudy") ||
      conditionLower.includes("partly")
    ) {
      return isDay ? (
        <WiDayCloudy size={size} className={`text-blue-400 ${baseClass}`} />
      ) : (
        <WiNightClear size={size} className={`text-blue-300 ${baseClass}`} />
      );
    }

    // Cloudy/Overcast
    if (
      conditionLower.includes("cloudy") ||
      conditionLower.includes("overcast")
    ) {
      return <WiCloudy size={size} className={`text-gray-500 ${baseClass}`} />;
    }

    // Rain
    if (
      conditionLower.includes("rain") ||
      conditionLower.includes("shower") ||
      conditionLower.includes("drizzle")
    ) {
      if (
        conditionLower.includes("heavy") ||
        conditionLower.includes("moderate")
      ) {
        return <WiRain size={size} className={`text-blue-600 ${baseClass}`} />;
      }
      return isDay ? (
        <WiDayRain size={size} className={`text-blue-500 ${baseClass}`} />
      ) : (
        <WiNightRain size={size} className={`text-blue-400 ${baseClass}`} />
      );
    }

    // Thunderstorm
    if (
      conditionLower.includes("thunder") ||
      conditionLower.includes("storm")
    ) {
      return isDay ? (
        <WiDayThunderstorm
          size={size}
          className={`text-purple-600 ${baseClass}`}
        />
      ) : (
        <WiNightThunderstorm
          size={size}
          className={`text-purple-500 ${baseClass}`}
        />
      );
    }

    // Snow
    if (
      conditionLower.includes("snow") ||
      conditionLower.includes("blizzard")
    ) {
      return isDay ? (
        <WiDaySnow size={size} className={`text-blue-200 ${baseClass}`} />
      ) : (
        <WiNightSnow size={size} className={`text-blue-100 ${baseClass}`} />
      );
    }

    // Fog/Mist
    if (
      conditionLower.includes("fog") ||
      conditionLower.includes("mist") ||
      conditionLower.includes("haze")
    ) {
      return isDay ? (
        <WiDayFog size={size} className={`text-gray-400 ${baseClass}`} />
      ) : (
        <WiNightFog size={size} className={`text-gray-300 ${baseClass}`} />
      );
    }

    // Default fallback
    return isDay ? (
      <WiDaySunny size={size} className={`text-yellow-500 ${baseClass}`} />
    ) : (
      <WiNightClear size={size} className={`text-blue-300 ${baseClass}`} />
    );
  };

  // Fetch current weather data
  const fetchCurrentWeather = async (loc) => {
    try {
      const response = await fetch(`${API_BASE_URL}/weather/current`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ location: loc }),
      });

      const data = await response.json();
      if (data.status === "ok" && !data.data.error) {
        setCurrentWeather(data.data);
        setLastUpdated(new Date().toLocaleString());
      } else {
        setError(data.data.error || "Failed to fetch weather data");
      }
    } catch (err) {
      setError("Network error while fetching weather data");
      console.error("Error fetching current weather:", err);
    }
  };

  // Fetch hourly forecast
  const fetchHourlyForecast = async (loc) => {
    try {
      const response = await fetch(`${API_BASE_URL}/weather/hourly`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ location: loc }),
      });

      const data = await response.json();
      if (data.status === "ok" && !data.forecast.error) {
        setHourlyForecast(data.forecast);
      } else {
        console.error("Hourly forecast error:", data.forecast.error);
      }
    } catch (err) {
      console.error("Error fetching hourly forecast:", err);
    }
  };

  // Fetch weekly forecast
  const fetchWeeklyForecast = async (loc) => {
    try {
      const response = await fetch(`${API_BASE_URL}/weather/weekly`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ location: loc }),
      });

      const data = await response.json();
      if (data.status === "ok" && !data.forecast.error) {
        setWeeklyForecast(data.forecast);
      } else {
        console.error("Weekly forecast error:", data.forecast.error);
      }
    } catch (err) {
      console.error("Error fetching weekly forecast:", err);
    }
  };

  // Fetch all weather data
  const fetchAllWeatherData = async (loc = location) => {
    setError(null);

    await Promise.all([
      fetchCurrentWeather(loc),
      fetchHourlyForecast(loc),
      fetchWeeklyForecast(loc),
    ]);
  };

  // Initial data fetch
  useEffect(() => {
    fetchAllWeatherData();
  }, []);

  // Handle location change
  const handleLocationChange = (newLocation) => {
    if (newLocation && newLocation.trim()) {
      setLocation(newLocation);
      fetchAllWeatherData(newLocation);
    }
  };

  // Manual refresh
  const handleRefresh = () => {
    fetchAllWeatherData();
  };

  // Filter hourly forecast for specific times
  const getFilteredHourlyForecast = () => {
    if (!hourlyForecast || !hourlyForecast.intervals) return [];

    const targetHours = [
      "12AM",
      "3AM",
      "6AM",
      "9AM",
      "12PM",
      "3PM",
      "6PM",
      "9PM",
    ];

    const filtered = hourlyForecast.intervals.filter((interval) => {
      if (interval.hour_12) {
        return targetHours.includes(interval.hour_12);
      }
      if (interval.time) {
        const hour = new Date(interval.time).getHours();
        const targetHours24 = [0, 3, 6, 9, 12, 15, 18, 21];
        return targetHours24.includes(hour);
      }
      return false;
    });

    if (filtered.length === 0) {
      return hourlyForecast.intervals.slice(0, 8);
    }

    return filtered.slice(0, 8);
  };

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#F8FAFC]">
        <div className="text-center">
          <p className="text-red-500 mb-4">Error: {error}</p>
          <button
            onClick={handleRefresh}
            className="px-4 py-2 bg-[#0A3D91] text-white rounded-lg hover:bg-[#1E293B]"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-[#F8FAFC]">
      {/* HEADER */}
      <div className="px-6 w-full h-20 flex items-center justify-between border-b-2 border-primary-dark/10">
        <div className="flex items-center gap-4">
          <div className="w-fit text-2xl font-bold bg-gradient-to-r from-[#222222] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent leading-snug">
            Sentiment Analysis Framework
          </div>
          <MdChevronRight className="text-3xl text-gray-400" />
          <p className="text-gray-600 text-2xl font-medium">WeatherAPI</p>
        </div>
        <button
          onClick={handleRefresh}
          className="p-2 text-[#6B7280] hover:text-[#0A3D91] transition-colors"
          title="Refresh weather data"
        >
          <MdRefresh size={20} />
        </button>
      </div>

      {/* Content Area - Scrollable */}
      <div className="flex-1 overflow-auto">
        {/* TOP BENTO */}
        <div className="flex flex-row items-center justify-center mt-8 mx-8 gap-8">
          {/* LEFT */}
          <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-4/6 h-full p-4 rounded-2xl">
            <div className="flex flex-row items-center gap-4">
              <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
              <p className="font-medium text-xl text-[#1E293B]">LOCATION</p>
            </div>
            <div className="flex flex-row gap-4 items-center mt-2 mx-8">
              <MdMyLocation color="#1E293B" size={24} />

              {/* Location Search Component */}
              <LocationSearch
                value={location}
                onChange={handleLocationChange}
                placeholder="Search location..."
                icon={MdMyLocation}
              />

              <p className="text-[#6B7280] text-xl">|</p>
              <p className="text-[#6B7280] text-xl tracking-wider">
                {currentWeather
                  ? `${currentWeather.location.lat}°N ${currentWeather.location.lon}°E`
                  : "Loading..."}
              </p>
            </div>
            <div className="flex flex-row items-center justify-center gap-8">
              <h1 className="text-9xl font-bold bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent my-16 tracking-widest">
                {currentWeather ? `${currentWeather.current.temp_c}°` : "--°"}
              </h1>
              {currentWeather ? (
                getWeatherIcon(
                  currentWeather.current.condition,
                  currentWeather.current.is_day !== 0,
                  200
                )
              ) : (
                <div className="w-48 h-48 bg-[#F1F5F9] rounded-lg flex items-center justify-center">
                  <span className="text-[#6B7280] text-xl">--</span>
                </div>
              )}
            </div>
            <div className="flex items-center justify-center my-3">
              <p className="text-[#6B7280] text-xl tracking-wide">
                {lastUpdated ? `AS OF ${lastUpdated}` : "Loading..."}
              </p>
            </div>
          </div>

          {/* RIGHT */}
          <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-2/6 h-full p-6 rounded-2xl flex flex-col gap-2">
            {/* Weather Condition Card */}
            <div className="bg-gradient-to-r from-[#F8FAFC] to-[#E2E8F0] rounded-lg p-5 border-l-4 border-[#0A3D91]">
              <p className="font-medium text-[#6B7280] text-sm tracking-wide">
                CONDITION
              </p>
              <p className="font-bold text-[#222222] text-xl">
                {currentWeather
                  ? currentWeather.current.condition
                  : "Loading..."}
              </p>
            </div>

            {/* Key Metrics Grid */}
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-gradient-to-br from-[#F8FAFC] to-[#E2E8F0] rounded-lg p-5">
                <p className="font-medium text-[#6B7280] text-sm">Heat Index</p>
                <p className="font-bold text-[#F87171] text-xl">
                  {currentWeather
                    ? `${currentWeather.current.heatindex_c}°`
                    : "--°"}
                </p>
              </div>

              <div className="bg-gradient-to-br from-[#F8FAFC] to-[#E2E8F0] rounded-lg p-5">
                <p className="font-medium text-[#6B7280] text-sm">Wind Chill</p>
                <p className="font-bold text-[#0A3D91] text-xl">
                  {currentWeather
                    ? `${currentWeather.current.windchill_c}°`
                    : "--°"}
                </p>
              </div>
            </div>

            {/* Humidity with Compact Progress Bar */}
            <div className="bg-gradient-to-r from-[#F8FAFC] to-[#E2E8F0] rounded-lg p-5">
              <div className="flex items-center justify-between mb-1">
                <p className="font-medium text-[#6B7280] text-sm">Humidity</p>
              </div>
              <div className="w-full bg-[#E2E8F0] rounded-full h-2 flex items-center justify-between">
                <div
                  className="bg-gradient-to-r from-[#94A3B8] to-[#0A3D91] h-2 rounded-full transition-all duration-500"
                  style={{
                    width: currentWeather
                      ? `${currentWeather.current.humidity}%`
                      : "0%",
                  }}
                ></div>

                <p className="font-bold text-[#0A3D91] text-lg">
                  {currentWeather
                    ? `${currentWeather.current.humidity}%`
                    : "--%"}
                </p>
              </div>
            </div>

            {/* Wind & UV Compact Row */}
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-gradient-to-br from-[#F8FAFC] to-[#E2E8F0] rounded-lg p-5">
                <p className="font-medium text-[#6B7280] text-sm">Wind</p>
                <div className="flex items-center gap-1">
                  <p className="font-bold text-[#1E293B]">
                    {currentWeather
                      ? `${currentWeather.current.wind_kph}`
                      : "--"}
                  </p>
                  <span className="text-xs text-[#6B7280]">km/h</span>
                  <span className="text-xs px-1 py-0.5 rounded bg-[#94A3B8] text-[#222222] font-medium ml-1">
                    {currentWeather ? currentWeather.current.wind_dir : "--"}
                  </span>
                </div>
              </div>

              <div className="bg-gradient-to-br from-[#F8FAFC] to-[#E2E8F0] rounded-lg p-5">
                <p className="font-medium text-[#6B7280] text-sm">UV Index</p>
                <div className="flex items-center gap-1">
                  <p className="font-bold text-[#475569]">
                    {currentWeather ? currentWeather.current.uv : "--"}
                  </p>
                  <span className="text-xs px-1 py-0.5 rounded bg-[#8A8B8C] text-[#FBFCFD] font-medium">
                    {currentWeather && currentWeather.current.uv
                      ? currentWeather.current.uv <= 2
                        ? "Low"
                        : currentWeather.current.uv <= 5
                        ? "Mod"
                        : currentWeather.current.uv <= 7
                        ? "High"
                        : "V.High"
                      : "--"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* BOTTOM BENTO */}
        <div className="flex flex-row items-center justify-center mt-8 mx-8 gap-8 mb-8">
          {/* LEFT - TODAY'S FORECAST */}
          <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-4/6 h-full p-4 rounded-2xl">
            <div className="flex flex-row items-center gap-4 mb-4">
              <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
              <p className="font-medium text-xl text-[#1E293B]">
                TODAY'S FORECAST
              </p>
            </div>

            <div className="grid grid-cols-4 gap-6 mb-4">
              {(() => {
                const filteredData = getFilteredHourlyForecast();
                if (filteredData.length === 0) {
                  return (
                    <div className="col-span-4 text-center text-[#6B7280] py-4">
                      Loading hourly forecast...
                    </div>
                  );
                }

                return filteredData.map((interval, index) => (
                  <div
                    key={index}
                    className="flex flex-col items-center p-2 bg-[#F8FAFC] rounded-lg hover:bg-[#EEF2FF] transition-colors"
                  >
                    <p className="font-medium text-[#1E293B] mb-1 text-lg">
                      {interval.hour_12 || interval.time || `${index * 3}:00`}
                    </p>
                    <div className="mb-1">
                      {getWeatherIcon(interval.condition, true, 64)}
                    </div>
                    <p className="font-semibold text-[#0A3D91] text-xl">
                      {interval.temp_c || interval.temperature || "--"}°
                    </p>
                    <p className="text-[#6B7280] text-sm">
                      {interval.chance_of_rain || interval.precipitation || 0}%
                    </p>
                  </div>
                ));
              })()}
            </div>
          </div>

          {/* RIGHT - WEEKLY FORECAST */}
          <div className="bg-[#FBFCFD] shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] w-2/6 h-full p-4 rounded-2xl">
            <div className="flex flex-row items-center gap-4 mb-4">
              <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
              <p className="font-medium text-xl text-[#1E293B]">
                3-DAY FORECAST
              </p>
            </div>
            <div className="grid grid-cols-1 gap-12">
              {weeklyForecast && weeklyForecast.forecast_days ? (
                weeklyForecast.forecast_days
                  .filter((day) => !day.is_estimated)
                  .slice(0, 3)
                  .map((day, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 bg-[#F8FAFC] rounded-lg hover:bg-[#EEF2FF] transition-colors"
                    >
                      <div className="flex items-center gap-4">
                        <div className="flex items-center justify-center">
                          {getWeatherIcon(day.condition, true, 64)}
                        </div>
                        <div>
                          <p className="font-medium text-xl text-[#1E293B]">
                            {day.day_name}
                          </p>
                          <p className="text-[#6B7280]">{day.condition}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-6">
                        <div className="text-center">
                          <p className="text-[#6B7280] text-lg">Rain</p>
                          <p className="font-medium text-[#0A3D91] text-xl">
                            {day.chance_of_rain}%
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-[#6B7280] text-lg">High/Low</p>
                          <p className="font-semibold text-[#1E293B] text-xl">
                            {day.max_temp_c}°/{day.min_temp_c}°
                          </p>
                        </div>
                      </div>
                    </div>
                  ))
              ) : (
                <div className="text-center text-[#6B7280] py-8">
                  Loading 3-day forecast...
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-center pb-6">
          <p className="text-sm text-[#6B7280] font-medium">
            Weather data sourced from{" "}
            <span className="text-[#0A3D91]">WeatherAPI.com</span>.
          </p>
        </div>
      </div>
    </div>
  );
}
