// frontend/app/results/[runId]/layout.tsx
'use client'; // Keep as client component for usePathname

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import React, { useState, useEffect } from 'react';
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

// Define Tab values
const TABS = {
    SUMMARY: 'summary',
    CHARTS: 'charts',
    SNAPSHOT: 'snapshot',
    LOGS: 'logs',
} as const;

type TabValue = typeof TABS[keyof typeof TABS];

export default function ResultLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: { runId: string };
}) {
  const pathname = usePathname();
  const runId = params.runId;
  const [activeTab, setActiveTab] = useState<TabValue>(TABS.SUMMARY);

  // Effect to update the active tab based on the current path
  useEffect(() => {
    if (pathname.endsWith(`/${TABS.CHARTS}`)) { setActiveTab(TABS.CHARTS); }
    else if (pathname.endsWith(`/${TABS.SNAPSHOT}`)) { setActiveTab(TABS.SNAPSHOT); }
    else if (pathname.endsWith(`/${TABS.LOGS}`)) { setActiveTab(TABS.LOGS); }
    else if (pathname === `/results/${runId}`) { setActiveTab(TABS.SUMMARY); }
    else { setActiveTab(TABS.SUMMARY); } // Default fallback
  }, [pathname, runId]);

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Workflow Results: <span className="font-mono text-xl break-all">{runId}</span></h1>

      <Tabs value={activeTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value={TABS.SUMMARY} asChild>
             <Link href={`/results/${runId}`}>Summary</Link>
          </TabsTrigger>
          <TabsTrigger value={TABS.CHARTS} asChild>
             <Link href={`/results/${runId}/${TABS.CHARTS}`}>Charts</Link>
          </TabsTrigger>
          <TabsTrigger value={TABS.SNAPSHOT} asChild>
             <Link href={`/results/${runId}/${TABS.SNAPSHOT}`}>Snapshot</Link>
          </TabsTrigger>
          <TabsTrigger value={TABS.LOGS} asChild>
             <Link href={`/results/${runId}/${TABS.LOGS}`}>Logs</Link>
          </TabsTrigger>
        </TabsList>
         <div className="mt-4 rounded-md border bg-card text-card-foreground shadow-sm p-4 md:p-6">
             {children}
         </div>
      </Tabs>
    </div>
  );
}