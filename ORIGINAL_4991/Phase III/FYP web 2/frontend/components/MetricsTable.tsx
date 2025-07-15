// frontend/components/MetricsTable.tsx
'use client';
import React, { useMemo, useState, useEffect } from 'react';
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
  Row,
} from "@tanstack/react-table";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { MetricData } from '@/models/workflow';
import { formatNumber } from '@/lib/utils';
import { cn } from '@/lib/utils';

interface MetricsTableProps {
  data: MetricData[];
  isLoading: boolean;
  error: string | null;
}

// --- Formatting and Comparison Helpers (Keep as before) ---
const formatMetricValue = (metricName: string, value: any): string => {
    if (metricName.includes('Positive Days %')) { return formatNumber(value, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + '%'; }
    const isPercent = ['Cumulative Return', 'Annualized Return', 'Annualized Volatility', 'Max Drawdown', 'Alpha (Jensen, ann.)', 'Tracking Error (ann.)'].some(pm => metricName.includes(pm));
    const precision = ['Sharpe Ratio', 'Sortino Ratio'].some(pm => metricName.includes(pm)) ? 3 : 4;
    if (metricName.includes('Beta') || metricName.includes('Correlation')) { return formatNumber(value, { maximumFractionDigits: 4 }); }
    if (metricName.includes('Trading Days')) { return formatNumber(value, { maximumFractionDigits: 0 }); }
    return formatNumber(value, { style: isPercent ? 'percent' : 'decimal', maximumFractionDigits: precision, minimumFractionDigits: isPercent ? 2 : (precision > 0 ? precision : 0), });
};
const getComparisonClass = (row: Row<MetricData>, benchmarkColName: string): string => {
    const metricName = row.original.Metric; let stratValue: number | null = null; let benchValue: number | null = null;
    try { stratValue = parseFloat(row.original.Strategy as string); } catch { stratValue = null; }
    try { benchValue = parseFloat(row.original[benchmarkColName] as string); } catch { benchValue = null; }
    if (stratValue === null || benchValue === null || isNaN(stratValue) || isNaN(benchValue)) { return ''; }
    const higherIsBetterMetrics = ['Cumulative Return', 'Annualized Return', 'Sharpe Ratio', 'Sortino Ratio', 'Positive Days %', 'Alpha (Jensen)'];
    const lowerIsBetterMetrics = ['Annualized Volatility', 'Max Drawdown', 'Tracking Error']; let isBetter = false;
    if (higherIsBetterMetrics.some(m => metricName.includes(m))) { isBetter = stratValue > benchValue; }
    else if (lowerIsBetterMetrics.some(m => metricName.includes(m))) { isBetter = metricName.includes('Max Drawdown') ? stratValue > benchValue : stratValue < benchValue; }
    else { return ''; }
    return isBetter ? 'text-green-600 dark:text-green-400 font-medium' : 'text-red-600 dark:text-red-400';
};
// --- End Helpers ---

export function MetricsTable({ data, isLoading, error }: MetricsTableProps) {

   const columns = useMemo<ColumnDef<MetricData>[]>(() => {
        if (!data || data.length === 0) return [];
        const headers = Object.keys(data[0]);
        const benchmarkColName = headers.find(h => h.toLowerCase().includes('benchmark')) || 'Benchmark';
        const trackerColName = headers.find(h => h.toLowerCase().includes('tracker')) || 'Tracker';
        const strategyColName = 'Strategy';
        if (!headers.includes(strategyColName)) { console.error("MetricsTable: 'Strategy' column missing!"); return []; }

        const definedColumns: ColumnDef<MetricData>[] = [
            { accessorKey: "Metric", header: "Metric", cell: ({ row }) => <div className="font-medium">{row.getValue("Metric")}</div> },
            { // Strategy Column
                accessorKey: strategyColName, header: () => <div className="text-right">{strategyColName}</div>,
                cell: ({ row }) => {
                    const metricName = row.original.Metric;
                    // Use row.getValue for simple keys like "Strategy"
                    const value = row.getValue(strategyColName);
                    const comparisonClass = headers.includes(benchmarkColName) ? getComparisonClass(row, benchmarkColName) : '';
                    return <div className={cn("text-right", comparisonClass)}>{formatMetricValue(metricName, value)}</div>;
                }
            }
        ];
        if (headers.includes(benchmarkColName)) { // Add Benchmark if exists
            definedColumns.push({
                accessorKey: benchmarkColName, // Keep accessor simple if possible, but header can be complex
                header: () => <div className="text-right">{benchmarkColName}</div>, // Display full name
                cell: ({ row }) => {
                    const metricName = row.original.Metric;
                    // **** Access directly via original data using the found name ****
                    const value = row.original[benchmarkColName];
                    // ***************************************************************
                    return <div className="text-right">{formatMetricValue(metricName, value)}</div>
                }
            });
        }
        if (headers.includes(trackerColName)) { // Add Tracker if exists
            definedColumns.push({
                accessorKey: trackerColName, // Keep accessor simple if possible
                header: () => <div className="text-right">{trackerColName}</div>, // Display full name
                cell: ({ row }) => {
                    const metricName = row.original.Metric;
                    // **** FIX: Access directly via original data using the found name ****
                    const value = row.original[trackerColName];
                    // ********************************************************************
                    const isComparisonMetric = ['Beta', 'Alpha (Jensen)', 'Correlation', 'Tracking Error'].some(m => metricName.includes(m));
                    const displayValue = isComparisonMetric ? "N/A" : formatMetricValue(metricName, value);
                    return <div className="text-right">{displayValue}</div>;
                }
            });
        }
        return definedColumns;
    }, [data]);

  const [filteredData, setFilteredData] = useState<MetricData[]>([]);

  useEffect(() => {
      const metricsOnly = data.filter(row =>
         !['Strategy_Name', 'Parameter_K', 'Parameter_N', 'Period_Start', 'Period_End'].includes(row.Metric)
       );
       setFilteredData(metricsOnly);
  }, [data]);

   const table = useReactTable({
     data: filteredData,
     columns,
     getCoreRowModel: getCoreRowModel(),
   });

  if (isLoading) return <p>Loading metrics...</p>;
  if (error) return <p className="text-red-500">Error loading metrics: {error}</p>;

  const headerInfo = data.reduce((acc, row) => {
       if (['Strategy_Name', 'Parameter_K', 'Parameter_N', 'Period_Start', 'Period_End'].includes(row.Metric)) {
           const valueKey = Object.keys(row).find(k => k !== 'Metric' && row[k] !== 'N/A' && row[k] !== '' && row[k] !== null);
           acc[row.Metric] = valueKey ? row[valueKey] : 'N/A';
       }
       return acc;
   }, {} as Record<string, any>);

  return (
     <div>
        <div className="mb-4 p-3 border rounded bg-muted/50 text-sm">
            <p><strong>Strategy:</strong> {headerInfo.Strategy_Name || 'N/A'}</p>
            <p><strong>Params (Best):</strong> k={headerInfo.Parameter_K || 'N/A'}, n={headerInfo.Parameter_N || 'N/A'}</p>
            <p><strong>Period:</strong> {headerInfo.Period_Start?.substring(0,10) || 'N/A'} to {headerInfo.Period_End?.substring(0,10) || 'N/A'}</p>
        </div>
        <div className="rounded-md border">
        <Table>
            <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                    <TableHead key={header.id}>{header.isPlaceholder ? null : flexRender( header.column.columnDef.header, header.getContext() )}</TableHead>
                ))}
                </TableRow>
            ))}
            </TableHeader>
            <TableBody>
            {table.getRowModel().rows?.length ? (
                table.getRowModel().rows.map((row) => (
                <TableRow key={row.id} data-state={row.getIsSelected() && "selected"}>
                    {row.getVisibleCells().map((cell) => ( <TableCell key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</TableCell> ))}
                </TableRow>
                )) ) : ( <TableRow><TableCell colSpan={columns.length} className="h-24 text-center">No metric results.</TableCell></TableRow> )}
            </TableBody>
        </Table>
        </div>
    </div>
  );
}