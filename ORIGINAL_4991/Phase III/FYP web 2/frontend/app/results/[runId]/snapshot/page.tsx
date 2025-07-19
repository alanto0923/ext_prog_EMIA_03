// frontend/app/results/[runId]/snapshot/page.tsx
'use client';

import React, { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { getWorkflowResults, fetchCsvData } from '@/lib/api';
import { WorkflowResultResponse, SnapshotData } from '@/models/workflow'; // Import models
import { SnapshotTable } from '@/components/SnapshotTable';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'; // For error display
import Link from 'next/link';

export default function ResultSnapshotPage() {
  const params = useParams();
  const runId = params.runId as string;
  const [results, setResults] = useState<WorkflowResultResponse | null>(null); // Store full results
  const [snapshotData, setSnapshotData] = useState<SnapshotData[]>([]);
  const [isLoading, setIsLoading] = useState(true); // Tracks loading of results *and* CSV
  const [error, setError] = useState<string | null>(null); // For results fetch error
  const [csvError, setCsvError] = useState<string | null>(null); // For CSV fetch/parse error

  useEffect(() => {
    let isMounted = true; // Flag to prevent state updates on unmounted component
    if (!runId) {
        setError("Run ID is missing.");
        setIsLoading(false);
        return;
    }

    const fetchResultsAndSnapshot = async () => {
      if (!isMounted) return;
      setIsLoading(true);
      setError(null);
      setCsvError(null);
      setSnapshotData([]);
      setResults(null);

      try {
        const data = await getWorkflowResults(runId);
        if (!isMounted) return;
        setResults(data);

        if (data.status === 'SUCCESS') {
            if (data.files.snapshot_csv) {
               const snapshotUrl = `${data.output_dir_url}/${data.files.snapshot_csv}`;
               try {
                   const csvData = await fetchCsvData(snapshotUrl);
                   if (!isMounted) return;
                   // Basic validation and type conversion
                   if (Array.isArray(csvData) && (csvData.length === 0 || (typeof csvData[0] === 'object' && 'Category' in csvData[0]))){
                        const processedData: SnapshotData[] = csvData.map((row: any) => ({
                           Date: row.Date || '', // Should be ISO string
                           Category: row.Category || 'Unknown',
                           Rank: row.Rank !== undefined && row.Rank !== '' && !isNaN(parseInt(row.Rank)) ? parseInt(row.Rank) : null, // Parse Rank safely
                           StockID: row.StockID || 'Unknown',
                           'Cumulative Return': row['Cumulative Return'] !== undefined && row['Cumulative Return'] !== '' && !isNaN(parseFloat(row['Cumulative Return'])) ? parseFloat(row['Cumulative Return']) : null,
                           'Snapshot Date Price': row['Snapshot Date Price'] !== undefined && row['Snapshot Date Price'] !== '' && !isNaN(parseFloat(row['Snapshot Date Price'])) ? parseFloat(row['Snapshot Date Price']) : null,
                           'Snapshot Date Daily Return': row['Snapshot Date Daily Return'] !== undefined && row['Snapshot Date Daily Return'] !== '' && !isNaN(parseFloat(row['Snapshot Date Daily Return'])) ? parseFloat(row['Snapshot Date Daily Return']) : null,
                           'Snapshot Date Score': row['Snapshot Date Score'] !== undefined && row['Snapshot Date Score'] !== '' && !isNaN(parseFloat(row['Snapshot Date Score'])) ? parseFloat(row['Snapshot Date Score']) : null,
                        }));
                        setSnapshotData(processedData);
                   } else {
                        console.error("Parsed snapshot CSV data is not in expected format:", csvData);
                        setCsvError('Failed to parse snapshot: Invalid data structure.');
                   }
               } catch (csvErr: any) {
                   console.error("Error fetching/parsing snapshot CSV:", csvErr);
                   if (isMounted) setCsvError(`Failed to load snapshot data: ${csvErr.message}`);
               }
            } else {
                 if (isMounted) setCsvError('Snapshot file not found in results for this successful run.');
            }
        } else { // Handle FAILURE status from results endpoint
            if (isMounted) setError(data.message || `Run ${runId} failed. Snapshot not available.`);
        }

      } catch (err: any) {
        console.error("Error fetching results:", err);
        if (isMounted) {
             if (err.message?.includes("not finished yet")) { setError("Run is still in progress. Snapshot not available yet."); }
             else if (err.message?.includes("not found")) { setError(`Run ID ${runId} not found.`); }
             else { setError(err.message || 'Failed to fetch results.'); }
        }
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };

    fetchResultsAndSnapshot();

    // Cleanup function
    return () => {
        isMounted = false;
    };
  }, [runId]); // Re-fetch if runId changes

  if (isLoading) return <div className="text-center p-4">Loading snapshot data...</div>;

  // Show primary error first
  if (error) return (
       <Card className="border-destructive mt-4">
            <CardHeader><CardTitle className="text-destructive">Error</CardTitle></CardHeader>
            <CardContent><p>{error}</p></CardContent>
       </Card>
   );

  // If results loaded but run failed or snapshot wasn't generated/loaded
  if (!results || results.status === 'FAILURE' || csvError) {
      return (
         <Card className="mt-4 border-orange-500">
            <CardHeader><CardTitle className="text-orange-600">Snapshot Not Available</CardTitle></CardHeader>
            <CardContent>
                 <p>{csvError || results?.message || "Snapshot data could not be loaded or was not generated for this run."}</p>
                 {results?.status === 'FAILURE' && results?.files?.log_file && (
                     <Link href={`/results/${runId}/logs`} className="text-sm text-orange-700 hover:underline mt-2 block">View Logs</Link>
                  )}
            </CardContent>
       </Card>
      );
  }

  // If successful and snapshot CSV exists (and loaded potentially), show table
  return (
    <div>
      <SnapshotTable data={snapshotData} isLoading={false} error={null} />
      {/* Pass error=null to the table as the page handles the fetch/parse error */}
    </div>
  );
}