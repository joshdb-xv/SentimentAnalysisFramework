import { MdUpload, MdCheckCircle } from "react-icons/md";

export default function FileUploadSection({
  dictionaryFile,
  keywordsFile,
  dictionaryInfo,
  keywordsInfo,
  uploadingDict,
  uploadingKeywords,
  loadingKeywords,
  handleDictionaryUpload,
  handleKeywordsLoad,
  handleKeywordsUpload,
}) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Dictionary Upload */}
      <div className="bg-bluish-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl flex flex-col min-h-[450px]">
        <div className="flex flex-row items-center gap-4">
          <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
          <p className="font-medium text-xl text-[#1E293B]">
            UPLOAD DICTIONARY
          </p>
        </div>

        <div className="flex-1 flex flex-col justify-center py-6">
          <label className="relative flex flex-col items-center justify-center border-2 border-dashed border-gray-light rounded-xl p-8 cursor-pointer hover:border-primary hover:bg-primary/5 transition group">
            <MdUpload className="text-5xl text-gray-light group-hover:text-primary mb-2 transition" />
            <span className="text-sm text-gray-mid text-center">
              {uploadingDict ? (
                <span className="text-primary font-medium">Uploading...</span>
              ) : dictionaryFile ? (
                <span className="text-primary-dark">{dictionaryFile.name}</span>
              ) : (
                "Click to upload Excel file"
              )}
            </span>
            <input
              type="file"
              accept=".xlsx,.xls"
              onChange={handleDictionaryUpload}
              disabled={uploadingDict}
              className="hidden"
            />

            {uploadingDict && (
              <div className="absolute inset-0 bg-white/80 rounded-xl flex items-center justify-center">
                <div className="flex flex-col items-center gap-2">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                  <span className="text-sm text-primary font-medium">
                    Loading...
                  </span>
                </div>
              </div>
            )}
          </label>

          {dictionaryInfo && (
            <div className="mt-4 p-4 bg-primary/10 border border-primary rounded-xl text-sm">
              <div className="flex items-start gap-2">
                <MdCheckCircle className="text-primary text-xl flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="font-semibold text-primary">
                    Uploaded successfully
                  </p>
                  <p className="text-primary-dark mt-1">
                    Rows:{" "}
                    <span className="font-mono font-semibold">
                      {dictionaryInfo.rows.toLocaleString()}
                    </span>
                  </p>
                  <p className="text-primary-dark">
                    Columns:{" "}
                    <span className="font-mono text-xs">
                      {dictionaryInfo.columns.join(", ")}
                    </span>
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Keywords Load/Upload */}
      <div className="bg-bluish-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl flex flex-col min-h-[450px]">
        <div className="flex flex-row items-center gap-4">
          <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
          <p className="font-medium text-xl text-[#1E293B]">CLIMATE KEYWORDS</p>
        </div>

        <div className="flex-1 flex flex-col justify-center py-6">
          <button
            onClick={handleKeywordsLoad}
            disabled={loadingKeywords}
            className="relative w-full mb-4 flex items-center justify-center gap-2 border-2 border-primary bg-primary/10 rounded-xl p-4 hover:bg-primary/20 transition disabled:opacity-50 disabled:cursor-not-allowed group"
          >
            <MdUpload className="text-2xl text-primary" />
            <span className="text-primary font-medium">
              {loadingKeywords
                ? "Loading..."
                : "Load from data/weatherkeywords.csv"}
            </span>

            {loadingKeywords && (
              <div className="absolute inset-0 bg-primary/10 rounded-xl flex items-center justify-center">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
              </div>
            )}
          </button>

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
              {uploadingKeywords ? (
                <span className="text-primary font-medium">Uploading...</span>
              ) : keywordsFile ? (
                keywordsFile.name
              ) : (
                "Click to upload custom CSV"
              )}
            </span>
            <input
              type="file"
              accept=".csv"
              onChange={handleKeywordsUpload}
              disabled={uploadingKeywords}
              className="hidden"
            />

            {uploadingKeywords && (
              <div className="absolute inset-0 bg-white/80 rounded-xl flex items-center justify-center">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
              </div>
            )}
          </label>

          {keywordsInfo && (
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
                      {keywordsInfo.keyword_count}
                    </span>
                  </p>
                  <p className="text-xs text-gray-mid mt-1 font-mono truncate">
                    {keywordsInfo.file_path}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
