// frontend/app/configure/page.tsx
'use client'; // Needs client-side interaction

import React, { useState, useEffect } from 'react'; // Added useEffect
import { ParameterForm } from '@/components/ParameterForm';
import { RunStatusDisplay } from '@/components/RunStatusDisplay';
import { startWorkflow, getWorkflowStatus } from '@/lib/api';
import { WorkflowConfig, WorkflowStatusResponse } from '@/models/workflow';
import { useRouter } from 'next/navigation';
import { toast } from "sonner";

export default function ConfigurePage() {
      const [isLoading, setIsLoading] = useState(false);
      const [runId, setRunId] = useState<string | null>(null);
      const [error, setError] = useState<string | null>(null);
      const [currentStatus, setCurrentStatus] = useState<WorkflowStatusResponse | null>(null);
      const router = useRouter();
      const [intervalId, setIntervalId] = useState<NodeJS.Timeout | null>(null); // Store interval ID in state

      const handleRunSubmit = async (config: WorkflowConfig) => {
        setIsLoading(true);
        setError(null);
        setRunId(null);
        setCurrentStatus(null);
        if (intervalId) clearInterval(intervalId); // Clear previous interval if any

        try {
          toast.info("Starting workflow run...");
          const response = await startWorkflow(config);
          setRunId(response.run_id);
          toast.success(`Workflow started with Run ID: ${response.run_id}`);

          // Start polling for status immediately
          pollStatus(response.run_id);

        } catch (err: any) {
          console.error("Failed to start workflow:", err);
          const errMsg = `Failed to start workflow: ${err.message}`;
          setError(errMsg);
          toast.error(errMsg);
          setIsLoading(false); // Stop loading only on start failure
        }
        // Don't set isLoading to false here; keep it true while polling
      };

       const pollStatus = (id: string) => {
          // Clear any existing interval before starting a new one
          if (intervalId) clearInterval(intervalId);

         const newIntervalId = setInterval(async () => {
           try {
             const statusResponse = await getWorkflowStatus(id);
             setCurrentStatus(statusResponse); // Update status state

             const statusToastId = `status-${id}`;

             if (statusResponse.status === 'SUCCESS') {
               toast.success(`Run ${id} completed successfully! Redirecting...`, { id: statusToastId });
               clearInterval(newIntervalId);
               setIntervalId(null); // Clear interval ID from state
               setIsLoading(false); // Stop loading on success
               router.push(`/results/${id}`); // Redirect to results page
             } else if (statusResponse.status === 'FAILURE') {
                const failureMsg = `Run ${id} failed: ${statusResponse.details?.error || 'Check logs.'}`;
                toast.error(failureMsg, { id: statusToastId });
                clearInterval(newIntervalId);
                setIntervalId(null);
                // Keep error state from handleRunSubmit if it failed initially
                // Otherwise, set error based on polling result
                if (!error) setError(`Run failed: ${statusResponse.details?.error || 'Check logs for details.'}`);
                setIsLoading(false); // Stop loading on failure
             } else if (statusResponse.status === 'PENDING' || statusResponse.status === 'STARTED' || statusResponse.status === 'PROGRESS' || statusResponse.status === 'RETRY') {
                // Update status toast
                toast.info(`Run ${id}: ${statusResponse.details?.step || statusResponse.status} (${statusResponse.details?.progress ?? 0}%)`, { id: statusToastId, duration: 10000 }); // Keep toast longer
             } else {
                  // Handle unexpected states (REVOKED, etc.)
                  const unexpectedMsg = `Run ${id}: Unexpected status ${statusResponse.status}`;
                  toast.warning(unexpectedMsg, { id: statusToastId });
                  clearInterval(newIntervalId);
                  setIntervalId(null);
                  if (!error) setError(unexpectedMsg); // Set error if not already set
                  setIsLoading(false); // Stop loading
             }
           } catch (pollError: any) {
             console.error("Polling error:", pollError);
             // Don't stop polling on transient network errors, but maybe show a warning
             // Only update toast if it's not already showing a failure
             if (currentStatus?.status !== 'FAILURE') {
                 toast.warning(`Could not get status update for ${id}: ${pollError.message}`, { id: `status-${id}` });
             }
             // Optionally implement logic to stop polling after several consecutive errors
           }
         }, 5000); // Poll every 5 seconds

         setIntervalId(newIntervalId); // Store the new interval ID
       };

        // Cleanup interval on component unmount
        useEffect(() => {
          return () => {
             if (intervalId) {
                 clearInterval(intervalId);
             }
          };
       }, [intervalId]); // Depend on intervalId


      return (
        <div className="space-y-6">
          <h1 className="text-3xl font-bold text-center mb-6">Configure and Run Workflow</h1>

          {/* Only show the form if a run hasn't been started yet in this session */}
          {!runId && <ParameterForm onSubmit={handleRunSubmit} isLoading={isLoading} />}

          {/* Show status display if a run has been initiated OR if there was an error starting */}
          {(runId || error && !isLoading) && (
            <div className="mt-8">
                 {/* Pass error only if it's specifically a start error */}
                <RunStatusDisplay statusData={currentStatus} error={!runId ? error : null} />
            </div>
          )}

           {/* Persistent failure message */}
           {runId && !isLoading && currentStatus?.status === 'FAILURE' && (
               <div className="mt-6 p-4 border border-destructive bg-destructive/10 rounded-md text-destructive">
                   <p className="font-semibold">Run Failed</p>
                   <p>Details: {currentStatus.details?.error || currentStatus.details?.message || 'Unknown failure. Please check logs.'}</p>
                   {/* Optionally add a link to logs page */}
                   <a href={`/results/${runId}/logs`} className="text-sm underline hover:text-destructive/80 mt-2 inline-block">View Logs</a>
               </div>
           )}

        </div>
      );
    }