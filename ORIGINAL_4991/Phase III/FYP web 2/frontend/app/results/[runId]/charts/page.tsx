// frontend/app/results/[runId]/charts/page.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { getWorkflowResults } from '@/lib/api';
import { WorkflowResultResponse } from '@/models/workflow'; // Import models
import { ChartDisplay } from '@/components/ChartDisplay';
import { getResultFileUrl } from '@/lib/utils';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'; // For error display
import Link from 'next/link';

export default function ResultChartsPage() {
    const params = useParams();
    const runId = params.runId as string;
    const [results, setResults] = useState<WorkflowResultResponse | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

     useEffect(() => {
        let isMounted = true; // Mount check
        if (!runId) {
            setError("Run ID is missing.");
            setIsLoading(false);
            return;
        }
        setIsLoading(true);
        setError(null);
        setResults(null);

        getWorkflowResults(runId)
            .then(data => {
                if (!isMounted) return;
                setResults(data); // Store results object
                if (data.status !== 'SUCCESS') {
                    setError(data.message || 'Run did not complete successfully. Charts not available.');
                }
            })
            .catch(err => {
                if (!isMounted) return;
                console.error("Error fetching results:", err);
                 if (err.message?.includes("not finished yet")) { setError("Run is still in progress. Charts not available yet."); }
                 else if (err.message?.includes("not found")) { setError(`Run ID ${runId} not found.`); }
                 else { setError(err.message || 'Failed to fetch results data.'); }
            })
            .finally(() => {
                if (isMounted) setIsLoading(false);
            });

        // Cleanup
        return () => { isMounted = false; };
    }, [runId]); // Re-fetch if runId changes

    if (isLoading) return <div className="text-center p-4">Loading chart information...</div>;

    // Display error if fetching failed or run was not successful
    if (error) return (
       <Card className="border-destructive mt-4">
            <CardHeader><CardTitle className="text-destructive">Charts Not Available</CardTitle></CardHeader>
            <CardContent>
                <p>{error}</p>
                {results?.files?.log_file && ( // Provide log link even if run failed
                    <Link href={`/results/${runId}/logs`} className="text-sm text-destructive/90 hover:underline mt-2 block">View Logs</Link>
                 )}
           </CardContent>
       </Card>
   );

    // Should only reach here if results are loaded and status is SUCCESS
    if (!results || results.status !== 'SUCCESS') {
        return <p>Could not load chart data. Run may have failed.</p>;
    }


    const files = results.files;
    const baseUrl = results.output_dir_url;

  return (
    <div className="space-y-6">
       <h2 className="text-xl font-semibold">Generated Charts</h2>
        {/* Conditionally render charts based on file availability */}
        <ChartDisplay
            chartUrl={getResultFileUrl(baseUrl, files.full_period_plot)}
            altText="Full Period Cumulative Return Chart"
            title="Full Period Cumulative Return (vs Comparisons)"
        />
        <ChartDisplay
            chartUrl={getResultFileUrl(baseUrl, files.equity_curve_plot)}
            altText="Strategy Dollar Equity Curve Chart"
            title="Strategy Dollar Equity Curve"
         />
         <ChartDisplay
             chartUrl={getResultFileUrl(baseUrl, files.training_history_plot)}
             altText="DNN Training History Plot"
             title="DNN Model Training History (Loss & MAE)"
         />
          <ChartDisplay // Use ChartDisplay even if optional, it handles null URL
               chartUrl={getResultFileUrl(baseUrl, files.test_period_plot)}
               altText="Test Period Cumulative Return Chart"
               title="Test Period Cumulative Return (vs Comparisons)"
           />
    </div>
  );
}