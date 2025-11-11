import { Suspense } from "react";
import Sidebar from "@/components/Sidebar";

export default function DashboardLayout({ children }) {
  return (
    <div className="flex">
      <Sidebar />
      <div className="ml-[20%] w-[80%]">{children}</div>
    </div>
  );
}
