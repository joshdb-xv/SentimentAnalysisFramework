import { MdSearch } from "react-icons/md";

export default function WordSearchSection({
  searchWord,
  setSearchWord,
  handleSearchWord,
  searching,
}) {
  return (
    <div className="bg-white shadow-[0px_2px_16px_0px_rgba(30,41,59,0.25)] p-8 rounded-2xl">
      <div className="flex flex-row items-center gap-4 mb-6">
        <div className="w-3 h-3 bg-gradient-to-r from-[#111111] via-[#1E293B] to-[#0A3D91] rounded-full"></div>
        <p className="font-medium text-xl text-[#1E293B]">SEARCH WORD</p>
      </div>

      <form onSubmit={handleSearchWord} className="flex gap-3">
        <div className="flex-1 relative">
          <input
            type="text"
            value={searchWord}
            onChange={(e) => setSearchWord(e.target.value)}
            placeholder="Enter a word to search..."
            className="w-full px-4 py-3 border-2 border-bluish-gray rounded-xl focus:border-primary focus:outline-none transition"
          />
          <MdSearch className="absolute right-4 top-1/2 -translate-y-1/2 text-2xl text-gray-light" />
        </div>
        <button
          type="submit"
          disabled={searching || !searchWord.trim()}
          className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-primary to-primary-dark text-white rounded-xl hover:from-primary-dark hover:to-black disabled:from-gray disabled:to-gray-light disabled:cursor-not-allowed transition shadow-sm hover:shadow-md"
        >
          <MdSearch className="text-xl" />
          {searching ? "Searching..." : "Search"}
        </button>
      </form>

      <p className="text-sm text-gray-mid mt-3">
        💡 Search for any word in the processed lexical dictionary to see its
        sentiment score and breakdown.
      </p>
    </div>
  );
}
