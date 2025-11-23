"use client";

import { MdChevronRight } from "react-icons/md";
import { usePathname } from "next/navigation";
import Link from "next/link";

export default function Header() {
  const pathname = usePathname();

  const navItems = [
    { name: "Home", path: "/home" },
    { name: "Observations", path: "/observations" },
    { name: "WeatherAPI", path: "/weatherapi" },
    { name: "TwitterAPI", path: "/twitter-debug" },
    { name: "Lexical Dictionary", path: "/lexical-debug" },
    {
      name: "Naive Bayes - Climate Related Checker",
      path: "/training-debug",
    },
    { name: "Naive Bayes - Climate Domain Classifier", path: "/domain-debug" },
  ];

  // Find the current page based on the URL
  const currentPage = navItems.find((item) => pathname === item.path);

  return (
    <div className="bg-white px-6 w-full h-20 flex items-center justify-between border-b-2 border-primary-dark/5 fixed top-0 z-50">
      <div className="flex items-center gap-4">
        <Link
          href="/home"
          className="w-fit text-2xl font-bold bg-gradient-to-r from-black via-primary-dark to-primary bg-clip-text text-transparent leading-snug"
        >
          Sentiment Analysis Framework
        </Link>

        {currentPage && (
          <>
            <MdChevronRight className="text-3xl text-gray-light" />
            <p className="text-gray-mid text-2xl font-medium">
              {currentPage.name}
            </p>
          </>
        )}
      </div>
    </div>
  );
}
