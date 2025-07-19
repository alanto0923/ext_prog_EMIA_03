// frontend/lib/api.ts
import {
  WorkflowConfig,
  RunHistoryResponse,
  WorkflowRunResponse,
  WorkflowStatusResponse,
  WorkflowResultResponse,
  DefaultConfigResponse // Ensure this is imported
} from '@/models/workflow';

// Define backend URL (use environment variable for flexibility)
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// --- Workflow API Calls ---

export async function startWorkflow(config: WorkflowConfig): Promise<WorkflowRunResponse> {
const response = await fetch(`${API_BASE_URL}/workflow/run`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(config),
});
if (!response.ok) {
  const errorData = await response.json().catch(() => ({ detail: 'Failed to start workflow: Invalid response from server' }));
  throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
}
return await response.json();
}

export async function getWorkflowStatus(runId: string): Promise<WorkflowStatusResponse> {
const response = await fetch(`${API_BASE_URL}/workflow/status/${runId}`);
if (!response.ok && response.status !== 202) {
   const errorData = await response.json().catch(() => ({ detail: `Failed to get status for ${runId}: Invalid response` }));
   throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
}
 try {
     const data = await response.json();
     return data as WorkflowStatusResponse;
 } catch (e) {
     if (response.status === 202) {
         return { task_id: runId, run_id: runId, status: 'PROGRESS', details: { message: 'Processing...' } };
     }
     console.error(`Failed to parse status response for ${runId} (Status: ${response.status})`, e);
     throw new Error(`Failed to parse status response for ${runId}. Status: ${response.status}`);
 }
}

export async function getWorkflowResults(runId: string): Promise<WorkflowResultResponse> {
const response = await fetch(`${API_BASE_URL}/workflow/results/${runId}`);
if (!response.ok) {
    if (response.status === 202) { throw new Error(`Workflow run '${runId}' is not finished yet.`); }
    else if (response.status === 404) { throw new Error(`Workflow run '${runId}' not found or results are unavailable.`); }
    const errorData = await response.json().catch(() => ({ detail: `Failed to fetch results for ${runId}: Invalid server response (Status: ${response.status})` }));
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
}
return await response.json();
}

export async function getDefaultConfig(): Promise<WorkflowConfig> {
  const response = await fetch(`${API_BASE_URL}/workflow/config/defaults`);
  if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch default config: Invalid response from server' }));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }
  const data: DefaultConfigResponse = await response.json(); // Use the imported type
  if (data && typeof data === 'object' && data.defaults) {
      return data.defaults as WorkflowConfig; // Return the nested 'defaults' object
  } else {
      console.error("Invalid format for default config response:", data);
      throw new Error("Invalid format received for default configuration.");
  }
}

export async function getRunHistory(limit: number = 50): Promise<RunHistoryResponse> {
  const response = await fetch(`${API_BASE_URL}/workflow/history?limit=${limit}`);
  if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch run history: Invalid response from server' }));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }
  return await response.json();
}


// --- Functions to fetch content of result files ---
export async function fetchResultFileContent(fileUrl: string): Promise<string> {
  if (!fileUrl) throw new Error("File URL is required.");
  const response = await fetch(fileUrl);
  if (!response.ok) {
      const errorText = await response.text().catch(() => `Status ${response.status}`);
      console.error(`Failed to fetch file content from ${fileUrl}. Response: ${errorText}`);
      throw new Error(`Failed to fetch file content from ${fileUrl}. Status: ${response.status}`);
  }
  return await response.text();
}

export async function fetchCsvData(fileUrl: string): Promise<any[]> {
   if (!fileUrl) throw new Error("File URL is required.");
  const textContent = await fetchResultFileContent(fileUrl);
  const lines = textContent.trim().replace(/\r\n/g, '\n').split('\n');
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim());
  const data = lines.slice(1).map(line => {
      const values: string[] = []; // Explicit type
      let currentVal = ''; let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
          const char = line[i];
          if (char === '"' && (i === 0 || line[i-1] !== '\\')) { inQuotes = !inQuotes; }
          else if (char === ',' && !inQuotes) { values.push(currentVal.trim()); currentVal = ''; }
          else { currentVal += char; }
      }
      values.push(currentVal.trim());
      if (values.length !== headers.length) { console.warn(`CSV row length mismatch: Expected ${headers.length}, got ${values.length}. Row: "${line}"`); }
      const row: { [key: string]: any } = {};
      headers.forEach((header, index) => { row[header] = values[index] !== undefined ? values[index].replace(/^"|"$/g, '') : ''; });
      return row;
  });
  return data;
}