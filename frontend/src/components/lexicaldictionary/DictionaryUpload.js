import { MdUpload, MdCheckCircle } from "react-icons/md";

export default function DictionaryUpload({ file, info, uploading, onUpload }) {
  return (
    <div className="bg-bluish-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl flex flex-col min-h-[450px]">
      <div className="flex flex-row items-center gap-4">
        <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
        <p className="font-medium text-xl text-[#1E293B]">UPLOAD DICTIONARY</p>
      </div>

      <div className="flex-1 flex flex-col justify-center py-6">
        <label className="relative flex flex-col items-center justify-center border-2 border-dashed border-gray-light rounded-xl p-8 cursor-pointer hover:border-primary hover:bg-primary/5 transition group">
          <MdUpload className="text-5xl text-gray-light group-hover:text-primary mb-2 transition" />
          <span className="text-sm text-gray-mid text-center">
            {uploading ? (
              <span className="text-primary font-medium">Uploading...</span>
            ) : file ? (
              <span className="text-primary-dark">{file.name}</span>
            ) : (
              "Click to upload Excel file"
            )}
          </span>
          <input
            type="file"
            accept=".xlsx,.xls"
            onChange={onUpload}
            disabled={uploading}
            className="hidden"
          />

          {uploading && (
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

        {info && (
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
                    {info.rows.toLocaleString()}
                  </span>
                </p>
                <p className="text-primary-dark">
                  Columns:{" "}
                  <span className="font-mono text-xs">
                    {info.columns.join(", ")}
                  </span>
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
