 // frontend/components/LogViewer.tsx
 import React from 'react';

 interface LogViewerProps {
   logContent: string | null;
   isLoading: boolean;
   error: string | null;
 }

 export function LogViewer({ logContent, isLoading, error }: LogViewerProps) {
   if (isLoading) return <p>Loading logs...</p>;
   if (error) return <p className="text-red-500">Error loading logs: {error}</p>;
   if (!logContent) return <p>No log content available.</p>;

   return (
     <pre className="bg-gray-900 text-gray-200 p-4 rounded-md text-xs overflow-auto h-96">
       {logContent}
     </pre>
   );
 }