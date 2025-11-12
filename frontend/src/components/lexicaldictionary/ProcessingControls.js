import { MdPlayArrow, MdRefresh, MdError } from "react-icons/md";

export default function ProcessingControls({
  dictionaryInfo,
  keywordsInfo,
  processing,
  status,
  startProcessing,
  resetAll,
}) {
  return (
    <div className="bg-bluish-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl flex flex-col min-h-[280px]">
      <div className="flex flex-row items-center gap-4">
        <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
        <p className="font-medium text-xl text-[#1E293B]">PROCESS DICTIONARY</p>
      </div>

      <div className="flex-1 flex flex-col justify-center py-6">
        <div className="mb-4 p-4 bg-primary/10 border border-primary rounded-xl">
          <p className="text-sm text-primary flex items-start gap-2">
            <span className="text-primary text-lg">⚡</span>
            <span>
              <strong>FastText embeddings are automatically used</strong> for
              semantic scoring. Models are pre-loaded at server startup for fast
              processing.
            </span>
          </p>
        </div>

        <div className="flex gap-3">
          <button
            onClick={startProcessing}
            disabled={!dictionaryInfo || processing}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-primary to-primary-dark text-white rounded-xl hover:from-primary-dark hover:to-black disabled:from-gray disabled:to-gray-light disabled:cursor-not-allowed transition shadow-sm hover:shadow-md"
          >
            <MdPlayArrow className="text-xl" />
            {processing ? "Processing..." : "Start Processing"}
          </button>

          <button
            onClick={resetAll}
            className="flex items-center gap-2 px-6 py-3 bg-bluish-gray text-gray-dark rounded-xl hover:bg-gray-light hover:text-white transition"
          >
            <MdRefresh className="text-xl" />
            Reset
          </button>
        </div>

        {!keywordsInfo && !processing && dictionaryInfo && (
          <p className="text-sm text-gray-mid mt-3 flex items-center gap-2">
            <span>💡</span>
            <span>
              No keywords loaded. Will use default from{" "}
              <code className="bg-bluish-gray px-1.5 py-0.5 rounded text-xs font-mono">
                data/weatherkeywords.csv
              </code>
            </span>
          </p>
        )}

        {status && (
          <div className="mt-4">
            <div className="flex justify-between text-sm text-primary-dark mb-2">
              <span className="font-medium">{status.message}</span>
              <span className="font-semibold text-primary">
                {status.progress}%
              </span>
            </div>
            <div className="w-full bg-bluish-gray rounded-full h-3 overflow-hidden">
              <div
                className="bg-gradient-to-r from-primary to-primary-dark h-3 rounded-full transition-all duration-300 ease-out"
                style={{ width: `${status.progress}%` }}
              />
            </div>

            {status.status === "error" && (
              <div className="mt-3 p-3 bg-red/10 border border-red rounded-xl flex items-start gap-2">
                <MdError className="text-red text-xl flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-red">
                    Error occurred
                  </p>
                  <p className="text-sm text-red mt-1">{status.message}</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
