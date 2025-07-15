// frontend/app/results/[runId]/page.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { getWorkflowResults, fetchCsvData } from '@/lib/api';
import { WorkflowResultResponse, MetricData } from '@/models/workflow'; // Import models
import { MetricsTable } from '@/components/MetricsTable';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import Link from 'next/link';

export default function ResultSummaryPage() {
  const params = useParams();
  const runId = params.runId as string;
  const [results, setResults] = useState<WorkflowResultResponse | null>(null);
  const [metricsData, setMetricsData] = useState<MetricData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null); // For results fetch error
  const [csvError, setCsvError] = useState<string | null>(null); // For CSV fetch/parse error

  useEffect(() => {
    let isMounted = true; // Flag to prevent state updates on unmounted component
    if (!runId) {
        setError("Run ID is missing.");
        setIsLoading(false);
        return;
    };

    const fetchResults = async () => {
      if (!isMounted) return;
      setIsLoading(true);
      setError(null);
      setCsvError(null);
      setMetricsData([]);
      setResults(null);

      try {
        const data = await getWorkflowResults(runId);
        if (!isMounted) return;
        setResults(data);

        if (data.status === 'SUCCESS') {
            if (data.files.metrics_csv) {
                const metricsUrl = `${data.output_dir_url}/${data.files.metrics_csv}`;
                try {
                    const csvData = await fetchCsvData(metricsUrl);
                    if (!isMounted) return;
                    if (Array.isArray(csvData) && (csvData.length === 0 || (typeof csvData[0] === 'object' && 'Metric' in csvData[0]))){
                       setMetricsData(csvData as MetricData[]);
                    } else {
                        console.error("Parsed CSV data is not in expected format:", csvData);
                        setCsvError('Failed to parse metrics: Invalid data structure.');
                    }
                } catch (csvErr: any) {
                    console.error("Error fetching/parsing metrics CSV:", csvErr);
                    if (isMounted) setCsvError(`Failed to load metrics data: ${csvErr.message}`);
                }
            } else {
                 if (isMounted) setCsvError('Metrics file not found in results for this successful run.');
            }
        } else { // Handle FAILURE status from results endpoint
            if (isMounted) setError(data.message || `Run ${runId} failed.`);
        }

      } catch (err: any) {
        console.error("Error fetching results:", err);
        if (isMounted) {
            if (err.message?.includes("not finished yet")) { setError("Run is still in progress. Please wait and refresh."); }
            else if (err.message?.includes("not found")) { setError(`Run ID ${runId} not found.`); }
            else { setError(err.message || 'Failed to fetch results.'); }
        }
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };

    fetchResults();

    // Cleanup function
    return () => {
        isMounted = false;
    };
  }, [runId]); // Re-fetch if runId changes

  if (isLoading) return <div className="text-center p-4">Loading results summary...</div>;

  // Display primary error first
  if (error) return (
      <Card className="border-destructive">
           <CardHeader><CardTitle className="text-destructive">Error Loading Results</CardTitle></CardHeader>
           <CardContent><p>{error}</p></CardContent>
      </Card>
  );

  // If results loaded but run failed
  if (!results || results.status === 'FAILURE') return (
        <Card className="border-destructive">
           <CardHeader><CardTitle className="text-destructive">Run Failed</CardTitle></CardHeader>
           <CardContent>
               <p>{results?.message || "The workflow run failed."}</p>
                {results?.files?.log_file && (
                   <Link href={`/results/${runId}/logs`} className="text-sm text-destructive/90 hover:underline mt-2 block">View Logs for details</Link>
                )}
           </CardContent>
      </Card>
  );


  // If successful run, display metrics table (and handle CSV error)
  return (
    <div>
       <Card>
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
            {/* Display best k/n if available in results */}
            {results?.best_k !== undefined && results?.best_k !== null && results?.best_n !== undefined && results?.best_n !== null && (
               <CardDescription>
                   Displaying results for best parameters found: k={results.best_k}, n={results.best_n}
               </CardDescription>
            )}
          </CardHeader>
          <CardContent>
             <MetricsTable data={metricsData} isLoading={false} error={csvError} />
             {/* isLoading is false because main loading is done, table handles csvError */}
          </CardContent>
       </Card>
       {/* Add more summary information here if needed, e.g., links to files */}
    </div>
  );
}