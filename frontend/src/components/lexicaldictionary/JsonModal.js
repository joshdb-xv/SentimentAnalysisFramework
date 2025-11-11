export default function JsonModal({ results, onClose }) {
  return (
    <div
      className="absolute inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-4 border-b border-bluish-gray">
          <h3 className="text-lg font-semibold text-black">Raw JSON Data</h3>
          <button
            onClick={onClose}
            className="text-gray-light hover:text-gray-mid transition"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
        <div
          className="flex-1 overflow-y-auto p-4 bg-black"
          style={{
            scrollbarWidth: "none",
            msOverflowStyle: "none",
          }}
        >
          <style>{`
            .flex-1.overflow-y-auto::-webkit-scrollbar {
              display: none;
            }
          `}</style>
          <pre className="text-xs font-mono text-blue whitespace-pre">
            {JSON.stringify(results, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}
