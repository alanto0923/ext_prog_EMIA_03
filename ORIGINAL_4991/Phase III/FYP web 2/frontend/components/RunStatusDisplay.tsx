 // frontend/components/RunStatusDisplay.tsx
 'use client';

 import React from 'react';
 import { WorkflowStatusResponse } from '@/models/workflow';
 import { Progress } from "@/components/ui/progress";
 import { Badge } from "@/components/ui/badge";

 interface RunStatusDisplayProps {
   statusData: WorkflowStatusResponse | null;
   error: string | null;
 }

 export function RunStatusDisplay({ statusData, error }: RunStatusDisplayProps) {
   if (error) {
     return <p className="text-red-500">Error checking status: {error}</p>;
   }

   if (!statusData) {
     return <p>Loading status...</p>;
   }

   const getBadgeVariant = (status: string): "default" | "secondary" | "destructive" | "outline" => {
       switch (status) {
           case 'SUCCESS': return 'default'; // Greenish in default theme
           case 'FAILURE': return 'destructive';
           case 'PENDING': return 'secondary';
           case 'STARTED':
           case 'PROGRESS':
           case 'RETRY':
               return 'outline'; // Yellowish/Blueish depending on theme
           default: return 'secondary';
       }
   }

   const progress = statusData.details?.progress ?? (statusData.status === 'SUCCESS' ? 100 : 0);
   const step = statusData.details?.step ?? statusData.details?.message ?? statusData.status;

   return (
     <div className="p-4 border rounded-lg space-y-3">
       <div className="flex justify-between items-center">
         <h3 className="text-lg font-semibold">Run Status (ID: {statusData.run_id})</h3>
         <Badge variant={getBadgeVariant(statusData.status)}>{statusData.status}</Badge>
       </div>
       <p>Details: {step}</p>
       {(statusData.status === 'PROGRESS' || statusData.status === 'STARTED' || statusData.status === 'SUCCESS') && (
         <Progress value={progress} className="w-full" />
       )}
        {statusData.status === 'FAILURE' && statusData.details?.error && (
             <div className="mt-2 p-3 bg-red-100 border border-red-300 rounded">
                 <p className="text-red-700 font-semibold">Error:</p>
                 <pre className="text-red-600 text-xs whitespace-pre-wrap">
                    {statusData.details.error}
                    {statusData.details.traceback ? `\n\nTraceback:\n${statusData.details.traceback}` : ''}
                 </pre>
             </div>
         )}
     </div>
   );
 }