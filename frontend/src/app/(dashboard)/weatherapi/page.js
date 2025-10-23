import { MdChevronRight } from "react-icons/md";

async function getData() {
  await new Promise((resolve) => setTimeout(resolve, 800));
  return {};
}

export default async function WeatherAPI() {
  await getData();

  return (
    <div className="">
      <div className="px-6 w-full h-20 flex items-center border-b-2 border-primary-dark/10 gap-4">
        <div className="w-fit text-2xl font-bold bg-gradient-to-r from-[#222222] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent leading-snug">
          Sentiment Analysis Framework
        </div>
        <MdChevronRight className="text-3xl text-gray-dark" />
        <p className="text-gray-dark text-2xl font-medium">WeatherAPI</p>
      </div>
    </div>
  );
}
