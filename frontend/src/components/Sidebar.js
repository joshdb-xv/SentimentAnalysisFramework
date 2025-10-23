"use client";

import {
  IoHome,
  IoHomeOutline,
  IoStatsChart,
  IoStatsChartOutline,
  IoCloudyNight,
  IoCloudyNightOutline,
  IoBook,
  IoBookOutline,
} from "react-icons/io5";
import { usePathname } from "next/navigation";
import Link from "next/link";

export default function Sidebar() {
  const pathname = usePathname();

  const navItems = [
    {
      name: "Home",
      path: "/home",
      activeIcon: IoHome,
      inactiveIcon: IoHomeOutline,
    },
    {
      name: "Observations",
      path: "/observations",
      activeIcon: IoStatsChart,
      inactiveIcon: IoStatsChartOutline,
    },
    {
      name: "WeatherAPI",
      path: "/weatherapi",
      activeIcon: IoCloudyNight,
      inactiveIcon: IoCloudyNightOutline,
    },
    {
      name: "Lexical Dictionary",
      path: "/lexicaldictionary",
      activeIcon: IoBook,
      inactiveIcon: IoBookOutline,
    },
  ];

  return (
    <div className="w-1/5 h-screen bg-[#1E293B]">
      <div className="px-4 h-20 flex items-center">
        <Link
          href="/"
          className="font-bold text-2xl text-[#FFFFFF] tracking-widest"
        >
          SAF
        </Link>
      </div>

      <div className="flex flex-col justify-center mx-6 mt-4 gap-4">
        {navItems.map((item) => {
          const isActive = pathname === item.path;
          const Icon = isActive ? item.activeIcon : item.inactiveIcon;

          return (
            <Link
              key={item.path}
              href={item.path}
              className={`flex gap-2 items-center transition-all duration-200 text-xl ${
                isActive
                  ? "text-[#FFFFFF] font-bold tracking-wide drop-shadow-[0_0_8px_rgba(255,255,255,0.5)]"
                  : "text-[#FBFCFD] hover:text-[#FFFFFF]"
              }`}
            >
              <Icon size={20} />
              {item.name}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
