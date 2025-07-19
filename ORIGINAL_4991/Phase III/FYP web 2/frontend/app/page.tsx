    // frontend/app/page.tsx
    import Link from 'next/link';
    import { Button } from "@/components/ui/button";
    import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";

    export default function HomePage() {
      return (
        <div className="space-y-6">
          <h1 className="text-3xl font-bold">HSI DNN Strategy Dashboard</h1>
          <p>Welcome to the strategy runner application.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
             <Card>
                <CardHeader>
                  <CardTitle>Configure & Run</CardTitle>
                  <CardDescription>Set parameters and start a new workflow run.</CardDescription>
                </CardHeader>
                 <CardContent>
                     <Link href="/configure">
                        <Button>Go to Configuration</Button>
                     </Link>
                 </CardContent>
             </Card>
              <Card>
                 <CardHeader>
                   <CardTitle>Run History (Not Implemented)</CardTitle>
                   <CardDescription>View past workflow runs and their results.</CardDescription>
                 </CardHeader>
                  <CardContent>
                      <Button disabled>View History</Button>
                  </CardContent>
              </Card>
          </div>
        </div>
      );
    }