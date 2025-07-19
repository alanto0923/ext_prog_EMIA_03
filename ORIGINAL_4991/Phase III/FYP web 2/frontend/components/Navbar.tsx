// frontend/components/Navbar.tsx
import Link from 'next/link';
import React from 'react';

export function Navbar() {
  return (
    <nav className="bg-gray-800 text-white p-4">
      <div className="container mx-auto flex justify-between items-center">
        <Link href="/" className="text-xl font-bold">
          HSI DNN Strategy
        </Link>
        <div className="space-x-4">
          <Link href="/configure" className="hover:text-gray-300">
            Configure & Run
          </Link>
          {/* This link is correct */}
          <Link href="/history" className="hover:text-gray-300">
             History
           </Link>
          {/* Add more links as needed */}
        </div>
      </div>
    </nav>
  );
}