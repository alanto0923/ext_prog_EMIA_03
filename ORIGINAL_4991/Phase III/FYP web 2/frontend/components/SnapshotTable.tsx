  // frontend/components/SnapshotTable.tsx
  'use client';
  import React, { useMemo } from 'react';
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
  import { SnapshotData } from '@/models/workflow';
  import { formatNumber, formatDate } from '@/lib/utils';
  import { Button } from './ui/button'; // For sorting indicator
  import { ArrowUpDown } from 'lucide-react'; // Icon for sorting

  interface SnapshotTableProps {
    data: SnapshotData[];
    isLoading: boolean;
    error: string | null;
  }

  export function SnapshotTable({ data, isLoading, error }: SnapshotTableProps) {
    const [sorting, setSorting] = React.useState<SortingState>([]);

    const columns = useMemo<ColumnDef<SnapshotData>[]>(() => [
      // Keep Date column simple if it's always the same
      // {
      //     accessorKey: "Date",
      //     header: "Date",
      //     cell: ({ row }) => formatDate(row.getValue("Date")),
      // },
      {
          accessorKey: "Category",
          header: "Category",
      },
      {
          accessorKey: "Rank",
          header: "Rank",
           cell: ({ row }) => row.getValue("Rank") ?? '-', // Show '-' if Rank is null/undefined
      },
      {
          accessorKey: "StockID",
          header: "Stock ID",
           cell: ({ row }) => <div className="font-mono">{row.getValue("StockID")}</div>,
      },
      {
          accessorKey: "Cumulative Return",
          header: ({ column }) => (
              <Button
                  variant="ghost"
                  onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
                  className="text-right w-full justify-end px-0 hover:bg-transparent"
              >
                  Cumul. Return
                  <ArrowUpDown className="ml-2 h-4 w-4" />
              </Button>
          ),
          cell: ({ row }) => <div className="text-right">{formatNumber(row.getValue("Cumulative Return"), { style: 'percent' })}</div>,
      },
      {
          accessorKey: "Snapshot Date Price",
           header: ({ column }) => (
              <Button
                  variant="ghost"
                  onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
                  className="text-right w-full justify-end px-0 hover:bg-transparent"
               >
                   Price
                   <ArrowUpDown className="ml-2 h-4 w-4" />
               </Button>
           ),
           cell: ({ row }) => <div className="text-right">{formatNumber(row.getValue("Snapshot Date Price"), { style: 'currency', currency: 'HKD', minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>, // Assuming HKD
      },
      {
          accessorKey: "Snapshot Date Daily Return",
           header: ({ column }) => (
              <Button
                   variant="ghost"
                   onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
                   className="text-right w-full justify-end px-0 hover:bg-transparent"
               >
                   Daily Return
                   <ArrowUpDown className="ml-2 h-4 w-4" />
               </Button>
           ),
           cell: ({ row }) => <div className="text-right">{formatNumber(row.getValue("Snapshot Date Daily Return"), { style: 'percent' })}</div>,
      },
       {
          accessorKey: "Snapshot Date Score",
           header: ({ column }) => (
              <Button
                   variant="ghost"
                   onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
                   className="text-right w-full justify-end px-0 hover:bg-transparent"
               >
                   Score
                   <ArrowUpDown className="ml-2 h-4 w-4" />
               </Button>
           ),
           cell: ({ row }) => <div className="text-right">{formatNumber(row.getValue("Snapshot Date Score"), { maximumFractionDigits: 5 })}</div>,
      },
    ], []);

    const table = useReactTable({
      data: data,
      columns,
      state: { sorting },
      onSortingChange: setSorting,
      getCoreRowModel: getCoreRowModel(),
      getSortedRowModel: getSortedRowModel(),
    });

    if (isLoading) return <p>Loading snapshot data...</p>;
    if (error) return <p className="text-red-500">Error loading snapshot: {error}</p>;
    if (!data || data.length === 0) return <p>No snapshot data available.</p>;

    const snapshotDate = data.length > 0 ? formatDate(data[0].Date) : 'N/A';

    return (
      <div>
          <h3 className="text-lg font-semibold mb-2">Portfolio Snapshot for {snapshotDate}</h3>
          <div className="rounded-md border">
          <Table>
              <TableHeader>
              {table.getHeaderGroups().map((headerGroup) => (
                  <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => {
                      return (
                      <TableHead key={header.id}>
                          {header.isPlaceholder
                          ? null
                          : flexRender(
                              header.column.columnDef.header,
                              header.getContext()
                              )}
                      </TableHead>
                      );
                  })}
                  </TableRow>
              ))}
              </TableHeader>
              <TableBody>
              {table.getRowModel().rows?.length ? (
                  table.getRowModel().rows.map((row) => (
                  <TableRow
                      key={row.id}
                      data-state={row.getIsSelected() && "selected"}
                  >
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
                      No results.
                  </TableCell>
                  </TableRow>
              )}
              </TableBody>
          </Table>
          </div>
       </div>
    );
  }