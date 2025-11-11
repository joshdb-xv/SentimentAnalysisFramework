import { MdDownload, MdRefresh } from "react-icons/md";
import StatisticsCards from "./StatisticsCards";
import AverageScoresPanel from "./AverageScores";
import PreviewTable from "./PreviewTable";

export default function ResultsSection({
  results,
  onDownloadCSV,
  onReset,
  onShowJson,
}) {
  return (
    <div className="bg-bluish-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl">
      <div className="flex justify-between items-center mb-6">
        <div className="flex flex-row items-center gap-4">
          <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
          <p className="font-medium text-xl text-[#1E293B]">RESULTS</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={onDownloadCSV}
            className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-primary to-primary-dark text-white rounded-xl hover:from-primary-dark hover:to-black transition shadow-sm hover:shadow-md"
          >
            <MdDownload className="text-xl" />
            Download CSV
          </button>
          <button
            onClick={onReset}
            className="flex items-center gap-2 px-4 py-2.5 bg-gray-mid text-white rounded-xl hover:bg-primary-dark transition shadow-sm hover:shadow-md"
          >
            <MdRefresh className="text-xl" />
            Start Again
          </button>
        </div>
      </div>

      <StatisticsCards stats={results.stats} />
      <AverageScoresPanel stats={results.stats} />
      <PreviewTable preview={results.preview} />

      {/* Debug Info */}
      <div className="mt-6">
        <button
          onClick={onShowJson}
          className="px-4 py-2 text-sm border border-gray-light rounded-lg hover:bg-bluish-gray transition text-primary-dark font-medium"
        >
          Show raw JSON data
        </button>
      </div>
    </div>
  );
}
