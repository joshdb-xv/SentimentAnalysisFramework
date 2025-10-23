"use client";

import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  const handlePlayClick = () => {
    router.push("/home");
  };

  return (
    <div className="w-full h-screen flex flex-col justify-center px-64 bg-[#FBFCFD]">
      <p className="text-7xl font-bold bg-gradient-to-r from-[#222222] via-[#1E293B] to-[#0A3D91] bg-clip-text text-transparent leading-snug">
        A Sentiment Analysis Framework for Assessing Filipino Public Perception
        of Climate Change
      </p>

      <div className="flex flex-row gap-4 text-xl py-4">
        <p className="font-bold text-[#6B7280]">BARTOLOME</p>
        <p className="font-bold text-[#6B7280]">•</p>
        <p className="font-bold text-[#6B7280]">BUMANGLAG</p>
        <p className="font-bold text-[#6B7280]">•</p>
        <p className="font-bold text-[#6B7280]">ROLLE</p>
      </div>

      <button
        className="bg-[#0A3D91] text-2xl text-white font-semibold tracking-wider py-4 w-1/6 rounded-full mt-8 hover:bg-[#08316f] ease-in-out duration-200 active:bg-[#062857]"
        onClick={handlePlayClick}
      >
        Start
      </button>
    </div>
  );
}
