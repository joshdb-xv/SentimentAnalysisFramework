"use client";

import { useState, useRef, useEffect } from "react";

export default function ClimateTrendsChart({ location, days }) {
  const [hoveredData, setHoveredData] = useState(null);
  const [animationProgress, setAnimationProgress] = useState(0);
  const [timeSeriesData, setTimeSeriesData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const chartRef = useRef(null);

  const API_BASE_URL = "http://localhost:8000";

  // Animate chart on mount and when data changes
  useEffect(() => {
    setAnimationProgress(0);
    const timer = setTimeout(() => {
      setAnimationProgress(1);
    }, 100);
    return () => clearTimeout(timer);
  }, [timeSeriesData]);

  // Fetch trends data from backend
  useEffect(() => {
    fetchTrendsData();
  }, [location, days]);

  const fetchTrendsData = async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (location) params.append("location", location);
      params.append("days", days);

      const response = await fetch(
        `${API_BASE_URL}/observations/trends?${params.toString()}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.status === "ok" && data.trends) {
        const aggregatedData = aggregateByMonth(data.trends);
        setTimeSeriesData(aggregatedData);
      } else {
        throw new Error(data.error || "Failed to fetch trends data");
      }
    } catch (err) {
      console.error("Error fetching trends data:", err);
      setError(err.message);
      setTimeSeriesData([]);
    } finally {
      setLoading(false);
    }
  };

  const aggregateByMonth = (dailyData) => {
    const monthlyMap = {};

    dailyData.forEach((day) => {
      const date = new Date(day.date);
      const monthKey = `${date.getFullYear()}-${String(
        date.getMonth() + 1
      ).padStart(2, "0")}`;
      const monthLabel = date.toLocaleDateString("en-US", {
        month: "short",
        year: "numeric",
      });

      if (!monthlyMap[monthKey]) {
        monthlyMap[monthKey] = {
          date: monthLabel,
          categories: {},
          total: 0,
        };
      }

      Object.entries(day.categories).forEach(([category, count]) => {
        if (!monthlyMap[monthKey].categories[category]) {
          monthlyMap[monthKey].categories[category] = 0;
        }
        monthlyMap[monthKey].categories[category] += count;
      });

      monthlyMap[monthKey].total += day.total;
    });

    return Object.keys(monthlyMap)
      .sort()
      .map((key) => {
        const month = monthlyMap[key];
        const result = { date: month.date };

        categories.forEach((cat) => {
          result[cat.key] = month.categories[cat.label] || 0;
        });

        return result;
      });
  };

  const categories = [
    {
      key: "seaLevelRise",
      label: "Sea Level Rise / Coastal Hazards",
      color: "#22D3EE",
    },
    { key: "extremeHeat", label: "Extreme Heat / Heatwaves", color: "#FB923C" },
    {
      key: "coldWeather",
      label: "Cold Weather / Temperature Drops",
      color: "#A855F7",
    },
    {
      key: "flooding",
      label: "Flooding and Extreme Precipitation",
      color: "#3B82F6",
    },
    {
      key: "storms",
      label: "Storms, Typhoons, and Wind Events",
      color: "#EF4444",
    },
    { key: "drought", label: "Drought and Water Scarcity", color: "#FACC15" },
    {
      key: "airPollution",
      label: "Air Pollution and Emissions",
      color: "#10B981",
    },
    {
      key: "environmental",
      label: "Environmental Degradation and Land Use",
      color: "#EC4899",
    },
    { key: "geological", label: "Geological Events", color: "#8B5CF6" },
  ];

  const stackedData = timeSeriesData.map((item) => {
    const stacked = { date: item.date };
    let cumulative = 0;

    categories.forEach((category) => {
      const animatedValue = (item[category.key] || 0) * animationProgress;
      stacked[category.key] = animatedValue;
      stacked[`${category.key}_start`] = cumulative;
      stacked[`${category.key}_end`] = cumulative + animatedValue;
      cumulative += animatedValue;
    });

    stacked.total = cumulative;
    return stacked;
  });

  const chartWidth = 1200;
  const chartHeight = 320;
  const margin = { top: 20, right: 30, bottom: 40, left: 30 };
  const innerWidth = chartWidth - margin.left - margin.right;
  const innerHeight = chartHeight - margin.top - margin.bottom;

  const maxTotal = Math.max(
    ...timeSeriesData.map((item) => {
      return categories.reduce((sum, cat) => sum + (item[cat.key] || 0), 0);
    }),
    1
  );
  const xScale = (index) =>
    stackedData.length > 1
      ? (index / (stackedData.length - 1)) * innerWidth
      : innerWidth / 2;
  const yScale = (value) => innerHeight - (value / maxTotal) * innerHeight;

  const generateSmoothAreaPath = (categoryKey) => {
    const points = stackedData.map((d, i) => ({
      x: xScale(i),
      yStart: yScale(d[`${categoryKey}_start`]),
      yEnd: yScale(d[`${categoryKey}_end`]),
    }));

    if (points.length === 0) return "";

    if (points.length === 1) {
      const p = points[0];
      const width = innerWidth * 1;
      const leftX = Math.max(0, p.x - width / 2);
      const rightX = Math.min(innerWidth, p.x + width / 2);

      return `M ${leftX} ${p.yStart} L ${leftX} ${p.yEnd} L ${rightX} ${p.yEnd} L ${rightX} ${p.yStart} Z`;
    }

    if (points.length === 2) {
      const [p1, p2] = points;
      return `M ${p1.x} ${p1.yEnd} L ${p2.x} ${p2.yEnd} L ${p2.x} ${p2.yStart} L ${p1.x} ${p1.yStart} Z`;
    }

    let topPath = `M ${points[0].x} ${points[0].yEnd}`;

    for (let i = 1; i < points.length; i++) {
      const prevPoint = points[i - 1];
      const currPoint = points[i];
      const cpx1 = prevPoint.x + (currPoint.x - prevPoint.x) * 0.4;
      const cpx2 = currPoint.x - (currPoint.x - prevPoint.x) * 0.4;

      topPath += ` C ${cpx1} ${prevPoint.yEnd}, ${cpx2} ${currPoint.yEnd}, ${currPoint.x} ${currPoint.yEnd}`;
    }

    for (let i = points.length - 1; i >= 0; i--) {
      if (i === points.length - 1) {
        topPath += ` L ${points[i].x} ${points[i].yStart}`;
      } else {
        const nextPoint = points[i + 1];
        const currPoint = points[i];
        const cpx1 = nextPoint.x - (nextPoint.x - currPoint.x) * 0.4;
        const cpx2 = currPoint.x + (nextPoint.x - currPoint.x) * 0.4;

        topPath += ` C ${cpx1} ${nextPoint.yStart}, ${cpx2} ${currPoint.yStart}, ${currPoint.x} ${currPoint.yStart}`;
      }
    }

    return topPath + " Z";
  };

  const handleMouseMove = (event) => {
    if (!chartRef.current || stackedData.length === 0) return;

    const rect = chartRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left - margin.left;

    if (x >= 0 && x <= innerWidth) {
      const dataIndex = Math.round((x / innerWidth) * (stackedData.length - 1));
      const dataPoint = timeSeriesData[dataIndex];

      if (dataPoint) {
        setHoveredData({
          date: dataPoint.date,
          data: categories
            .map((cat) => ({
              label: cat.label,
              value: dataPoint[cat.key] || 0,
              color: cat.color,
            }))
            .filter((item) => item.value > 0)
            .sort((a, b) => b.value - a.value),
          total: categories.reduce(
            (sum, cat) => sum + (dataPoint[cat.key] || 0),
            0
          ),
          x: event.clientX,
          y: event.clientY,
          index: dataIndex,
        });
      }
    }
  };

  const handleMouseLeave = () => {
    setHoveredData(null);
  };

  return (
    <div className="h-full flex flex-col">
      {/* Tooltip */}
      {hoveredData && (
        <div
          className="fixed bg-white text-gray-800 text-xs rounded-lg px-3 py-3 pointer-events-none z-50 shadow-xl border border-gray-200 max-w-xs"
          style={{
            left: hoveredData.x + 10,
            top: hoveredData.y - 10,
            transform: "translate(0, -100%)",
          }}
        >
          <div className="font-semibold text-sm mb-2 text-gray-900">
            {hoveredData.date}
          </div>
          {hoveredData.data.length > 0 ? (
            hoveredData.data.map((item, index) => (
              <div key={index} className="flex items-center gap-2 mb-1">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: item.color }}
                ></div>
                <span className="text-gray-600 text-xs">
                  {item.label}: {Math.round(item.value)}
                </span>
              </div>
            ))
          ) : (
            <div className="text-gray-500 text-xs italic">No data</div>
          )}
          <div className="border-t border-gray-200 pt-1 mt-2">
            <span className="font-semibold text-gray-900">
              Total: {Math.round(hoveredData.total)}
            </span>
          </div>
        </div>
      )}

      {/* Header with title and legend */}
      <div className="flex items-center mb-4 flex-shrink-0">
        <div className="flex items-center gap-3 mr-8">
          <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
          <p className="font-medium text-xl text-[#1E293B] tracking-wide">
            CLIMATE TRENDS
          </p>
        </div>

        {/* Ultra-compact legend */}
        <div className="flex items-center gap-3 flex-1">
          {categories.map((category) => (
            <div
              key={category.key}
              className="relative group cursor-pointer"
              title={category.label}
            >
              <div
                className="w-3 h-3 rounded-full border-2 border-white shadow-sm hover:scale-110 transition-transform duration-200"
                style={{ backgroundColor: category.color }}
              ></div>
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-white text-[9px] rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10">
                {category.label}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center flex-1">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#0A3D91] mx-auto mb-4"></div>
            <p className="text-[#6B7280]">Loading climate trends...</p>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="flex items-center justify-center flex-1">
          <div className="text-center">
            <p className="text-red-500 mb-4">Error: {error}</p>
            <button
              onClick={fetchTrendsData}
              className="px-4 py-2 bg-[#0A3D91] text-white rounded-lg hover:bg-[#1E293B]"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Chart Container */}
      {!loading && !error && (
        <div className="flex-1 flex justify-center items-center min-h-0">
          {timeSeriesData.length > 0 ? (
            <div className="relative w-full h-full">
              <svg
                ref={chartRef}
                width="100%"
                height="100%"
                viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                className="overflow-visible"
                onMouseMove={handleMouseMove}
                onMouseLeave={handleMouseLeave}
                preserveAspectRatio="xMidYMid meet"
              >
                <defs>
                  {categories.map((category) => (
                    <linearGradient
                      key={`grad-${category.key}`}
                      id={`gradient-${category.key}`}
                      x1="0%"
                      y1="0%"
                      x2="0%"
                      y2="100%"
                    >
                      <stop
                        offset="0%"
                        stopColor={category.color}
                        stopOpacity="0.8"
                      />
                      <stop
                        offset="100%"
                        stopColor={category.color}
                        stopOpacity="0.6"
                      />
                    </linearGradient>
                  ))}

                  <pattern
                    id="grid"
                    width="40"
                    height="30"
                    patternUnits="userSpaceOnUse"
                  >
                    <path
                      d="M 40 0 L 0 0 0 30"
                      fill="none"
                      stroke="#F1F5F9"
                      strokeWidth="0.5"
                    />
                  </pattern>
                </defs>

                <rect
                  x={margin.left}
                  y={margin.top}
                  width={innerWidth}
                  height={innerHeight}
                  fill="url(#grid)"
                />

                <g transform={`translate(${margin.left}, ${margin.top})`}>
                  {categories.map((category, index) => (
                    <path
                      key={category.key}
                      d={generateSmoothAreaPath(category.key)}
                      fill={`url(#gradient-${category.key})`}
                      className="transition-all duration-300 hover:opacity-90"
                      style={{
                        transformOrigin: "bottom",
                        animation: `slideUp 0.8s ease-out ${
                          index * 0.05
                        }s both`,
                      }}
                    />
                  ))}

                  {[
                    0,
                    maxTotal * 0.25,
                    maxTotal * 0.5,
                    maxTotal * 0.75,
                    maxTotal,
                  ].map((value, i) => (
                    <g key={i}>
                      <line
                        x1={-5}
                        y1={yScale(value)}
                        x2={innerWidth}
                        y2={yScale(value)}
                        stroke="#E2E8F0"
                        strokeWidth="0.5"
                      />
                      <text
                        x={-10}
                        y={yScale(value)}
                        textAnchor="end"
                        dominantBaseline="middle"
                        className="text-xs fill-[#6B7280] font-medium"
                      >
                        {value.toFixed(0)}
                      </text>
                    </g>
                  ))}

                  {stackedData.map((d, i) => {
                    const showLabel =
                      stackedData.length <= 12 ||
                      i % Math.ceil(stackedData.length / 12) === 0;
                    if (!showLabel) return null;

                    return (
                      <text
                        key={i}
                        x={xScale(i)}
                        y={innerHeight + 20}
                        textAnchor="middle"
                        className="text-xs fill-[#6B7280] font-medium"
                      >
                        {d.date.split(" ")[0]}
                      </text>
                    );
                  })}

                  {hoveredData && (
                    <line
                      x1={xScale(hoveredData.index)}
                      y1={0}
                      x2={xScale(hoveredData.index)}
                      y2={innerHeight}
                      stroke="#475569"
                      strokeWidth="2"
                      strokeDasharray="4,4"
                      opacity="0.6"
                    />
                  )}
                </g>
              </svg>
            </div>
          ) : (
            <div className="text-center text-[#6B7280]">
              <p>No climate trend data available for the selected period.</p>
            </div>
          )}
        </div>
      )}

      <style jsx>{`
        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(20px) scaleY(0.8);
          }
          to {
            opacity: 1;
            transform: translateY(0) scaleY(1);
          }
        }
      `}</style>
    </div>
  );
}
