import { Poppins } from "next/font/google";
import "./globals.css";
import ProgressBar from "@/components/ProgressBar";

const poppins = Poppins({
  variable: "--font-poppins",
  subsets: ["latin"],
  weight: ["100", "200", "300", "400", "500", "600", "700", "800", "900"],
});

export const metadata = {
  title: "Sentiment Analysis Framework",
  description:
    "A Sentiment Analysis Framework for Assessing Filipino Public Perception of Climate Change",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning={true}>
      <ProgressBar />
      <body
        className={`${poppins.variable} antialiased`}
        suppressHydrationWarning={true}
      >
        {children}
      </body>
    </html>
  );
}
