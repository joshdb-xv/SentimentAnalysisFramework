import { MdUpload, MdCheckCircle } from "react-icons/md";

export default function KeywordsUpload({
  file,
  info,
  uploading,
  loading,
  onLoad,
  onUpload,
}) {
  return (
    <div className="bg-bluish-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl flex flex-col min-h-[450px]">
      <div className="flex flex-row items-center gap-4">
        <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
        <p className="font-medium text-xl text-[#1E293B]">CLIMATE KEYWORDS</p>
      </div>

      <div className="flex-1 flex flex-col justify-center py-6">
        {/* Load from default location */}
        <button
          onClick={onLoad}
          disabled={loading}
          className="relative w-full mb-4 flex items-center justify-center gap-2 border-2 border-primary bg-primary/10 rounded-xl p-4 hover:bg-primary/20 transition disabled:opacity-50 disabled:cursor-not-allowed group"
        >
          <MdUpload className="text-2xl text-primary" />
          <span className="text-primary font-medium">
            {loading ? "Loading..." : "Load from data/weatherkeywords.csv"}
          </span>

          {loading && (
            <div className="absolute inset-0 bg-primary/10 rounded-xl flex items-center justify-center">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
            </div>
          )}
        </button>

        {/* Or upload custom file */}
        <div className="relative my-4">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-bluish-gray"></div>
          </div>
          <div className="relative flex justify-center text-xs">
            <span className="px-2 bg-white text-gray">
              Or upload custom file
            </span>
          </div>
        </div>

        <label className="relative flex flex-col items-center justify-center border-2 border-dashed border-gray-light rounded-xl p-6 cursor-pointer hover:border-primary hover:bg-primary/5 transition group">
          <MdUpload className="text-4xl text-gray-light group-hover:text-primary mb-2 transition" />
          <span className="text-sm text-gray-mid">
            {uploading ? (
              <span className="text-primary font-medium">Uploading...</span>
            ) : file ? (
              file.name
            ) : (
              "Click to upload custom CSV"
            )}
          </span>
          <input
            type="file"
            accept=".csv"
            onChange={onUpload}
            disabled={uploading}
            className="hidden"
          />

          {uploading && (
            <div className="absolute inset-0 bg-white/80 rounded-xl flex items-center justify-center">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
            </div>
          )}
        </label>

        {info && (
          <div className="mt-4 p-4 bg-primary/10 border border-primary rounded-xl text-sm">
            <div className="flex items-start gap-2">
              <MdCheckCircle className="text-primary text-xl flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="font-semibold text-primary">
                  Keywords loaded successfully
                </p>
                <p className="text-primary-dark mt-1">
                  Keywords:{" "}
                  <span className="font-mono font-semibold">
                    {info.keyword_count}
                  </span>
                </p>
                <p className="text-xs text-gray-mid mt-1 font-mono truncate">
                  {info.file_path}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
