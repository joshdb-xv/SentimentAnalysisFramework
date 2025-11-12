import { Suspense } from "react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";

export default function DashboardLayout({ children }) {
  return (
    <div className="flex">
      <Sidebar />
      <div className="ml-[20%] w-[80%]">
        <Header />
        {children}
      </div>
    </div>
  );
}
