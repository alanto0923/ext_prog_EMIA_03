// frontend/app/results/[runId]/logs/page.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { getWorkflowResults, fetchResultFileContent } from '@/lib/api';
import { WorkflowResultResponse } from '@/models/workflow'; // Import models
import { LogViewer } from '@/components/LogViewer';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'; // For error display

export default function ResultLogsPage() {
  const params = useParams();
  const runId = params.runId as string;
  const [logContent, setLogContent] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true); // Tracks loading of log file itself
  const [error, setError] = useState<string | null>(null); // For errors fetching results OR log content

  useEffect(() => {
    let isMounted = true; // Flag to prevent state updates on unmounted component
    if (!runId) {
        setError("Run ID is missing.");
        setIsLoading(false);
        return;
    };

    const fetchLog = async () => {
      if (!isMounted) return;
      setIsLoading(true);
      setError(null);
      setLogContent(null);

      try {
        // First get results to find the log file path
        const resultsData = await getWorkflowResults(runId);
        if (!isMounted) return; // Check after await

        if (resultsData.files.log_file) {
           const logUrl = `${resultsData.output_dir_url}/${resultsData.files.log_file}`;
           try {
               // Then fetch the actual log content
               const content = await fetchResultFileContent(logUrl);
               if (!isMounted) return;
               setLogContent(content);
           } catch (logFetchErr: any) {
               console.error("Error fetching log file:", logFetchErr);
               if (isMounted) setError(`Failed to load log file: ${logFetchErr.message}`);
           }
        } else {
             if (isMounted) setError('Log file path not found in results.');
        }

         // Set main error only if the run itself failed, separate from log fetch error
         if (resultsData.status !== 'SUCCESS' && !error) { // Avoid overwriting log fetch error
             if (isMounted) setError(resultsData.message || 'Run failed.');
         }

      } catch (err: any) {
        console.error("Error fetching results/log info:", err);
        if (isMounted) {
            if (err.message?.includes("not finished yet")) { setError("Run is still in progress. Logs not available yet."); }
            else if (err.message?.includes("not found")) { setError(`Run ID ${runId} not found.`); }
            else { setError(err.message || 'Failed to fetch results/log info.'); }
        }
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };

    fetchLog();

    // Cleanup function
    return () => {
        isMounted = false;
    };
  }, [runId]); // Re-fetch if runId changes

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Run Logs</h2>
      <LogViewer logContent={logContent} isLoading={isLoading} error={error} />
    </div>
  );
}