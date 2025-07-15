  // frontend/app/layout.tsx
  import type { Metadata } from "next";
  import { Inter } from "next/font/google";
  import "./globals.css";
  import { Navbar } from "@/components/Navbar";
  import { Toaster } from "@/components/ui/sonner" // For notifications

  const inter = Inter({ subsets: ["latin"] });

  export const metadata: Metadata = {
    title: "HSI DNN Strategy",
    description: "Run and analyze the HSI DNN Alpha Yield Strategy",
  };

  export default function RootLayout({
    children,
  }: Readonly<{
    children: React.ReactNode;
  }>) {
    return (
      <html lang="en">
        <body className={inter.className}>
          <Navbar />
          <main className="container mx-auto p-4 mt-4">
              {children}
          </main>
           <Toaster /> {/* Add toaster for notifications */}
        </body>
      </html>
    );
  }