// frontend/components/ChartDisplay.tsx
import Image from 'next/image';
import React from 'react';

interface ChartDisplayProps {
  chartUrl: string | null;
  altText: string;
  title?: string;
}

export function ChartDisplay({ chartUrl, altText, title }: ChartDisplayProps) {
  if (!chartUrl) {
    return <p className="text-center text-gray-500">Chart not available.</p>;
  }

  return (
    <div className="my-4 p-2 border rounded">
       {title && <h4 className="text-md font-semibold mb-2 text-center">{title}</h4>}
      <div className="relative w-full h-auto" style={{ minHeight: '400px' }}> {/* Adjust minHeight as needed */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={chartUrl}
          alt={altText}
          style={{ width: '100%', height: 'auto', objectFit: 'contain' }} // Let image dictate aspect ratio
          loading="lazy" // Lazy load images
        />
      </div>
    </div>
  );
}