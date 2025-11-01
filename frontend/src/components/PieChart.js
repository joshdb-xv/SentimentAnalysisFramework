import { useState, useEffect } from "react";

export default function SentimentDistributionChart({ location, days }) {
  const [hoveredData, setHoveredData] = useState(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dominantSentiment, setDominantSentiment] = useState("NEUTRAL");
  const [totalAnalyzed, setTotalAnalyzed] = useState(0);
  const [mostPositiveCategory, setMostPositiveCategory] = useState(null);
  const [mostNegativeCategory, setMostNegativeCategory] = useState(null);

  const API_BASE_URL = "http://localhost:8000";

  // Sentiment colors
  const sentimentColors = {
    POSITIVE: "#7DD3FC",
    NEUTRAL: "#94A3B8",
    NEGATIVE: "#F87171",
  };

  // Fetch sentiment data from backend
  useEffect(() => {
    fetchSentimentData();
    fetchCategoryData();
  }, [location, days]);

  const fetchSentimentData = async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (location) params.append("location", location);
      params.append("days", days);

      const response = await fetch(
        `${API_BASE_URL}/observations/sentiment?${params.toString()}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.status === "ok" && data.sentiment) {
        const formattedData = [
          {
            name: "POSITIVE",
            value: data.sentiment.positive || 0,
            color: sentimentColors.POSITIVE,
          },
          {
            name: "NEUTRAL",
            value: data.sentiment.neutral || 0,
            color: sentimentColors.NEUTRAL,
          },
          {
            name: "NEGATIVE",
            value: data.sentiment.negative || 0,
            color: sentimentColors.NEGATIVE,
          },
        ];

        setChartData(formattedData);
        setDominantSentiment(data.dominant || "NEUTRAL");
        setTotalAnalyzed(data.total_analyzed || 0);
      } else {
        throw new Error(data.error || "Failed to fetch sentiment data");
      }
    } catch (err) {
      console.error("Error fetching sentiment data:", err);
      setError(err.message);
      setChartData([
        { name: "POSITIVE", value: 0, color: sentimentColors.POSITIVE },
        { name: "NEUTRAL", value: 0, color: sentimentColors.NEUTRAL },
        { name: "NEGATIVE", value: 0, color: sentimentColors.NEGATIVE },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const fetchCategoryData = async () => {
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
        const significantCategories = data.categories.filter(
          (cat) => cat.total >= 5
        );

        let maxPositiveScore = 0;
        let maxNegativeScore = 0;
        let mostPositive = null;
        let mostNegative = null;

        significantCategories.forEach((cat) => {
          if (cat.total > 0) {
            const positivePercent = (cat.positive / cat.total) * 100;
            const negativePercent = (cat.negative / cat.total) * 100;

            const positiveScore = positivePercent * Math.log(cat.positive + 1);
            const negativeScore = negativePercent * Math.log(cat.negative + 1);

            if (positiveScore > maxPositiveScore && cat.positive > 0) {
              maxPositiveScore = positiveScore;
              mostPositive = {
                name: cat.name,
                percentage: positivePercent.toFixed(1),
                count: cat.positive,
              };
            }

            if (negativeScore > maxNegativeScore && cat.negative > 0) {
              maxNegativeScore = negativeScore;
              mostNegative = {
                name: cat.name,
                percentage: negativePercent.toFixed(1),
                count: cat.negative,
              };
            }
          }
        });

        setMostPositiveCategory(mostPositive);
        setMostNegativeCategory(mostNegative);
      }
    } catch (err) {
      console.error("Error fetching category data:", err);
      setMostPositiveCategory(null);
      setMostNegativeCategory(null);
    }
  };

  const handleMouseEnter = (item, event) => {
    const percentage =
      totalAnalyzed > 0
        ? ((item.value / totalAnalyzed) * 100).toFixed(1)
        : "0.0";

    setHoveredData({
      name: item.name,
      value: item.value,
      percentage: percentage,
      total: totalAnalyzed,
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

  // Create SVG path for pie slices
  const createPieSlice = (item, index, startAngle, endAngle, radius = 100) => {
    const centerX = 140;
    const centerY = 120;

    const angleDiff = endAngle - startAngle;
    const isFullCircle = angleDiff >= 2 * Math.PI - 0.001;

    if (isFullCircle) {
      return (
        <circle
          key={index}
          cx={centerX}
          cy={centerY}
          r={radius}
          fill={item.color}
          stroke="#ffffff"
          strokeWidth="2"
          style={{ cursor: "pointer" }}
          onMouseEnter={(event) => handleMouseEnter(item, event)}
          onMouseLeave={handleMouseLeave}
          className="hover:brightness-115 transition-all duration-200"
        />
      );
    }

    const x1 = centerX + radius * Math.cos(startAngle);
    const y1 = centerY + radius * Math.sin(startAngle);
    const x2 = centerX + radius * Math.cos(endAngle);
    const y2 = centerY + radius * Math.sin(endAngle);

    const largeArcFlag = angleDiff <= Math.PI ? "0" : "1";

    const pathData = [
      "M",
      centerX,
      centerY,
      "L",
      x1,
      y1,
      "A",
      radius,
      radius,
      0,
      largeArcFlag,
      1,
      x2,
      y2,
      "Z",
    ].join(" ");

    return (
      <path
        key={index}
        d={pathData}
        fill={item.color}
        stroke="#ffffff"
        strokeWidth="2"
        style={{ cursor: "pointer" }}
        onMouseEnter={(event) => handleMouseEnter(item, event)}
        onMouseLeave={handleMouseLeave}
        className="hover:brightness-115 transition-all duration-200"
      />
    );
  };

  let currentAngle = -Math.PI / 2;
  const slices = chartData
    .filter((item) => item.value > 0)
    .map((item, index) => {
      const sliceAngle =
        totalAnalyzed > 0 ? (item.value / totalAnalyzed) * 2 * Math.PI : 0;
      const startAngle = currentAngle;
      const endAngle = currentAngle + sliceAngle;
      currentAngle = endAngle;

      return createPieSlice(item, index, startAngle, endAngle);
    });

  return (
    <div className="" onMouseMove={handleMouseMove}>
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
          <div className="font-semibold">{hoveredData.name}</div>
          <div className="text-gray-200">
            Value: {hoveredData.value} ({hoveredData.percentage}%)
          </div>
          <div className="text-gray-300 text-xs">
            Total: {hoveredData.total}
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
        <p className="font-medium text-xl text-[#1E293B] tracking-wide">
          SENTIMENT DISTRIBUTION
        </p>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#0A3D91] mx-auto mb-4"></div>
            <p className="text-[#6B7280]">Loading sentiment data...</p>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <p className="text-red-500 mb-4">Error: {error}</p>
            <button
              onClick={fetchSentimentData}
              className="px-4 py-2 bg-[#0A3D91] text-white rounded-lg hover:bg-[#1E293B]"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Chart Content */}
      {!loading && !error && (
        <>
          <div className="flex flex-row">
            {/* Custom SVG Pie Chart */}
            <div className="flex items-center justify-center">
              {totalAnalyzed > 0 ? (
                <svg width="280" height="240" viewBox="0 0 280 240">
                  {slices}
                </svg>
              ) : (
                <div className="w-[280px] h-[240px] flex items-center justify-center text-[#6B7280]">
                  No data available
                </div>
              )}
            </div>

            {/* Custom legend area */}
            <div className="flex items-center justify-center">
              <div className="flex flex-col items-center justify-center gap-6">
                {chartData.map((entry) => (
                  <div key={entry.name} className="flex items-center gap-1">
                    <div
                      className="w-3 h-3 rounded-xl"
                      style={{ backgroundColor: entry.color }}
                    ></div>
                    <span className="font-medium text-[#6B7280] uppercase text-sm">
                      {entry.name}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex flex-col mx-10 gap-2">
              <div>
                <div className="text-sm font-semibold text-[#1E293B] mb-1">
                  MOST POSITIVE{" "}
                  {mostPositiveCategory &&
                    `(${mostPositiveCategory.percentage}%)`}
                </div>
                {mostPositiveCategory ? (
                  <div>
                    <div className="text-[#7DD3FC] font-semibold">
                      {mostPositiveCategory.name}
                    </div>
                  </div>
                ) : (
                  <div className="text-[#6B7280] text-sm italic">
                    No data available
                  </div>
                )}
              </div>

              <div>
                <div className="text-sm font-semibold text-[#1E293B] mb-1">
                  MOST NEGATIVE{" "}
                  {mostNegativeCategory &&
                    `(${mostNegativeCategory.percentage}%)`}
                </div>
                {mostNegativeCategory ? (
                  <div>
                    <div className="text-[#F87171] font-semibold">
                      {mostNegativeCategory.name}
                    </div>
                  </div>
                ) : (
                  <div className="text-[#6B7280] text-sm italic">
                    No data available
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
