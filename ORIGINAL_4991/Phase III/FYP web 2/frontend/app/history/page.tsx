// frontend/app/history/page.tsx
'use client';

import React, { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import { getRunHistory } from '@/lib/api';
// Import models explicitly
import { RunHistoryItem, RunHistoryResponse } from '@/models/workflow';
import { format, parseISO } from 'date-fns'; // For date formatting
import { Badge } from "@/components/ui/badge";
import { Button } from '@/components/ui/button';
import { ArrowUpDown } from 'lucide-react';

import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
  SortingState,
  getSortedRowModel,
} from "@tanstack/react-table";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

// Helper to format dates/times or return 'N/A'
const formatDateTime = (dateTimeString?: string | Date | null): string => {
    if (!dateTimeString) return 'N/A';
    try {
        // Try parsing assuming ISO string format from backend (like Celery's date_done)
        const date = typeof dateTimeString === 'string' ? parseISO(dateTimeString) : dateTimeString;
         if (isNaN(date.getTime())) return 'Invalid Date';
        return format(date, 'yyyy-MM-dd HH:mm:ss'); // Format as desired
    } catch (e) {
        // Handle potential errors if date string is not ISO parseable
        console.warn("Date formatting error:", e);
        return 'Invalid Date';
    }
};

// Helper for status badge variant
const getStatusBadgeVariant = (status: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (status?.toUpperCase()) { // Make comparison case-insensitive
        case 'SUCCESS': return 'default'; // Greenish in default theme
        case 'FAILURE': return 'destructive'; // Red
        case 'PENDING': return 'secondary'; // Grey
        case 'STARTED': return 'outline';   // Blueish/Yellowish outline
        case 'PROGRESS': return 'outline';
        case 'RETRY': return 'outline';
        case 'REVOKED': return 'secondary';
        default: return 'secondary'; // Default grey for UNKNOWN or other states
    }
}


export default function HistoryPage() {
  const [history, setHistory] = useState<RunHistoryItem[]>([]); // Use imported type
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sorting, setSorting] = React.useState<SortingState>([]); // State for TanStack Table sorting


  useEffect(() => {
    let isMounted = true; // Mount check
    const fetchHistory = async () => {
      if (!isMounted) return;
      setIsLoading(true);
      setError(null);
      try {
        const response: RunHistoryResponse = await getRunHistory(); // Use imported type
        if (isMounted) setHistory(response.history);
      } catch (err: any) {
        if (isMounted) setError(`Failed to load run history: ${err.message}`);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };
    fetchHistory();
    // Cleanup
    return () => { isMounted = false; };
  }, []); // Empty dependency array means run once on mount

   // Define columns for TanStack Table
   const columns = useMemo<ColumnDef<RunHistoryItem>[]>(() => [ // Use imported type
     {
       accessorKey: "run_id",
       header: "Run ID",
       cell: ({ row }) => (
         // Link to the specific results page for this run
         <Link href={`/results/${row.getValue("run_id")}`} className="text-blue-600 hover:underline font-mono">
           {/* Display first 8 chars of ID for brevity */}
           {(row.getValue("run_id") as string).substring(0, 8)}...
         </Link>
       ),
        enableSorting: false, // Usually don't sort by ID link
     },
      {
       accessorKey: "end_time", // Use end_time (completion time) for sorting by default
       header: ({ column }) => ( // Make header clickable for sorting
         <Button variant="ghost" onClick={() => column.toggleSorting(column.getIsSorted() === "asc")} className='px-1 text-left justify-start'>
           Completed At
           <ArrowUpDown className="ml-2 h-4 w-4" />
         </Button>
       ),
       cell: ({ row }) => formatDateTime(row.getValue("end_time")), // Format the date/time
     },
     {
       accessorKey: "status",
       header: "Status",
       cell: ({ row }) => {
           const status = row.getValue("status") as string;
           // Use Badge component for visual status
           return <Badge variant={getStatusBadgeVariant(status)}>{status || 'UNKNOWN'}</Badge>;
       },
        enableSorting: false, // Sorting by badge might not be intuitive
     },
     {
        accessorKey: "best_k",
        header: "Best K",
        cell: ({ row }) => row.getValue("best_k") ?? '-', // Show '-' if null/undefined
     },
     {
        accessorKey: "best_n",
        header: "Best N",
         cell: ({ row }) => row.getValue("best_n") ?? '-',
     },
     {
        accessorKey: "message",
        header: "Summary / Message",
        // Make message column wider if needed via TableHead className or style
        cell: ({ row }) => <div className="text-xs truncate max-w-xs" title={row.getValue("message") as string | undefined}>{row.getValue("message") || '-'}</div>,
        enableSorting: false,
     },
   ], []); // Empty dependency array for columns

   // Setup TanStack Table instance
   const table = useReactTable({
     data: history, // Use imported type
     columns,
     state: { sorting }, // Control sorting state
     onSortingChange: setSorting, // Function to update sorting state
     getCoreRowModel: getCoreRowModel(),
     getSortedRowModel: getSortedRowModel(), // Enable sorting model
     // Default sort: newest completed runs first
     initialState: {
         sorting: [{ id: 'end_time', desc: true }],
     },
   });


  // Render the component
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Workflow Run History</h1>

      {isLoading && <p className="text-center p-4">Loading history...</p>}
      {error && <p className="text-red-500 p-4">{error}</p>}

      {!isLoading && !error && (
         <div className="rounded-md border">
         <Table>
             <TableHeader>
             {table.getHeaderGroups().map((headerGroup) => (
                 <TableRow key={headerGroup.id}>
                 {headerGroup.headers.map((header) => (
                     <TableHead key={header.id}>
                         {header.isPlaceholder ? null : flexRender( header.column.columnDef.header, header.getContext() )}
                     </TableHead>
                 ))}
                 </TableRow>
             ))}
             </TableHeader>
             <TableBody>
             {table.getRowModel().rows?.length ? (
                 table.getRowModel().rows.map((row) => (
                 <TableRow key={row.id} data-state={row.getIsSelected() && "selected"}>
                     {row.getVisibleCells().map((cell) => (
                     <TableCell key={cell.id}>
                         {flexRender(cell.column.columnDef.cell, cell.getContext())}
                     </TableCell>
                     ))}
                 </TableRow>
                 ))
             ) : (
                 <TableRow>
                 <TableCell colSpan={columns.length} className="h-24 text-center">
                     No run history found. Start a new run from the Configure page.
                 </TableCell>
                 </TableRow>
             )}
             </TableBody>
         </Table>
         </div>
      )}
    </div>
  );
}