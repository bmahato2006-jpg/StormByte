import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CYCLONE-X | AI Tropical Cyclone Intelligence & Disaster Platform",
  description:
    "AI-Powered Tropical Cyclone Intelligence, Digital Twin, and Disaster Decision-Support Platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[linear-gradient(90deg,#93c5fd_0%,#bfdbfe_14%,#dbeafe_26%,#eff6ff_38%,#ffffff_48%,#ffffff_52%,#eff6ff_62%,#dbeafe_74%,#bfdbfe_86%,#93c5fd_100%)] text-slate-800 antialiased selection:bg-blue-100 selection:text-blue-900">
        {children}
      </body>
    </html>
  );
}

