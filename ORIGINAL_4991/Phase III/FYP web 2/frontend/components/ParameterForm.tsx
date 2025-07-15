// frontend/components/ParameterForm.tsx
'use client'; // Mark as client component

import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { getDefaultConfig } from '@/lib/api';
import { WorkflowConfig } from '@/models/workflow'; // Import model
import { Textarea } from './ui/textarea'; // For ticker list

interface ParameterFormProps {
  onSubmit: (config: WorkflowConfig) => void;
  isLoading: boolean; // To disable button during submission
}

export function ParameterForm({ onSubmit, isLoading }: ParameterFormProps) {
  // State specifically for the form's current values (strings initially)
  const [formState, setFormState] = useState<Record<string, any>>({});
  const [loadingDefaults, setLoadingDefaults] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadDefaults() {
      try {
        setError(null);
        setLoadingDefaults(true);
        const defaults = await getDefaultConfig(); // Fetches WorkflowConfig structure

        // Prepare initial form state from defaults, converting lists to comma-separated strings for display
        const initialFormState: Record<string, any> = { ...defaults };
        if (defaults.tickers && Array.isArray(defaults.tickers)) {
            initialFormState.tickers = defaults.tickers.join(', ');
        }
         if (defaults.k_values && Array.isArray(defaults.k_values)) {
            initialFormState.k_values = defaults.k_values.join(', '); // Use k_values for form state key
        }
         if (defaults.n_values && Array.isArray(defaults.n_values)) {
            initialFormState.n_values = defaults.n_values.join(', '); // Use n_values for form state key
        }
        // Ensure numeric defaults are strings for the input fields initially
        if (typeof initialFormState.seed === 'number') initialFormState.seed = String(initialFormState.seed);
        if (typeof initialFormState.total_strategy_capital === 'number') initialFormState.total_strategy_capital = String(initialFormState.total_strategy_capital);

        setFormState(initialFormState);

      } catch (err: any) {
        setError(`Failed to load default config: ${err.message}`);
        setFormState({}); // Reset or keep empty on error
      } finally {
        setLoadingDefaults(false);
      }
    }
    loadDefaults();
  }, []); // Load defaults only once on mount

  // Handles changes for all input types (text, number, date, textarea)
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormState(prev => ({ ...prev, [name]: value })); // Store raw string value from input
  };

  // Processes form state and calls the onSubmit prop
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null); // Clear previous submission errors

    // Convert form state strings back into the structure expected by WorkflowConfig
    // Initialize optional array fields to prevent potential undefined errors later
    const configToSend: WorkflowConfig = {
        tickers: [],
        k_values: [],
        n_values: [],
        // Initialize other optional fields as null or based on your preference
        strategy_name: null,
        seed: null,
        start_date_str: null,
        validation_end_date: null,
        extend_simulation_end_date_str: null,
        lookback_date_str: null,
        total_strategy_capital: null,
    };

    try {
        for (const key in formState) {
            const value = formState[key]; // This is the string value from the form input

            // Skip if value is effectively empty after trimming
            if (value === null || value === undefined || String(value).trim() === '') {
                // If empty, ensure the corresponding key in configToSend remains null or default []
                continue;
            }

            // Process specific keys
            if (key === 'tickers' && typeof value === 'string') {
                configToSend.tickers = value.split(',').map((t: string) => t.trim()).filter(t => t); // Add string type for 't'
            } else if (key === 'k_values' && typeof value === 'string') {
                const parsedValues = value.split(',').map(v => parseInt(v.trim()));
                if (parsedValues.some(isNaN)) throw new Error("Invalid non-numeric value found in K Values. Please use comma-separated numbers.");
                configToSend.k_values = parsedValues.filter(v => !isNaN(v)); // Ensure only numbers are included
            } else if (key === 'n_values' && typeof value === 'string') {
                const parsedValues = value.split(',').map(v => parseInt(v.trim()));
                if (parsedValues.some(isNaN)) throw new Error("Invalid non-numeric value found in N Values. Please use comma-separated numbers.");
                configToSend.n_values = parsedValues.filter(v => !isNaN(v));
            } else if (key === 'seed' && typeof value === 'string') {
                const num = parseInt(value, 10); // Use parseInt for seed
                if (!isNaN(num)) configToSend.seed = num;
                else throw new Error("Invalid Seed value. Please enter an integer.");
            } else if (key === 'total_strategy_capital' && typeof value === 'string') {
                const num = parseFloat(value); // Use parseFloat for capital
                if (!isNaN(num)) configToSend.total_strategy_capital = num;
                 else throw new Error("Invalid Starting Capital value. Please enter a number.");
            } else {
                 // Directly assign other fields (like strategy_name, dates which are already strings)
                 // Check if the key is a valid key of WorkflowConfig before assigning
                 if (key in configToSend) { // Check against initialized keys
                     configToSend[key as keyof WorkflowConfig] = value;
                 }
            }
        }

        // Optional: Remove empty arrays if backend prefers null/undefined
        // if (configToSend.tickers?.length === 0) configToSend.tickers = null;
        // if (configToSend.k_values?.length === 0) configToSend.k_values = null;
        // if (configToSend.n_values?.length === 0) configToSend.n_values = null;

        onSubmit(configToSend); // Pass the structured and typed config object

    } catch (parseError: any) {
        console.error("Error processing form fields:", parseError);
        setError(`Error processing form fields: ${parseError.message}. Please check inputs.`);
    }
  };

  if (loadingDefaults) return <div className="text-center p-4">Loading default configuration...</div>;

  // Render Form
  return (
    <Card className="w-full max-w-3xl mx-auto">
      <CardHeader>
        <CardTitle>Configuration Parameters</CardTitle>
        <CardDescription>Adjust parameters or use defaults. Empty fields will use server defaults.</CardDescription>
      </CardHeader>
      <form onSubmit={handleSubmit}>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
           {/* General Settings */}
          <div className="space-y-1">
            <Label htmlFor="strategy_name">Run Name (Optional)</Label>
            <Input id="strategy_name" name="strategy_name" value={formState.strategy_name || ''} onChange={handleChange} placeholder="e.g., My Test Run"/>
          </div>
           <div className="space-y-1">
            <Label htmlFor="seed">Random Seed</Label>
            {/* Keep type="text" for input, parse in handleSubmit */}
            <Input id="seed" name="seed" type="text" inputMode='numeric' pattern='[0-9]*' value={formState.seed ?? ''} onChange={handleChange} placeholder="e.g., 42"/>
          </div>

          {/* Date Settings - Input type="date" handles formatting */}
          <div className="space-y-1">
            <Label htmlFor="start_date_str">History Start Date</Label>
            <Input id="start_date_str" name="start_date_str" type="date" value={formState.start_date_str || ''} onChange={handleChange} />
          </div>
          <div className="space-y-1">
             <Label htmlFor="validation_end_date">Validation End Date</Label>
             <Input id="validation_end_date" name="validation_end_date" type="date" value={formState.validation_end_date || ''} onChange={handleChange} />
           </div>
           <div className="space-y-1">
             <Label htmlFor="extend_simulation_end_date_str">Simulation End Date</Label>
             <Input id="extend_simulation_end_date_str" name="extend_simulation_end_date_str" type="date" value={formState.extend_simulation_end_date_str || ''} onChange={handleChange} />
           </div>
           <div className="space-y-1">
             <Label htmlFor="lookback_date_str">Snapshot Report Date</Label>
             <Input id="lookback_date_str" name="lookback_date_str" type="date" value={formState.lookback_date_str || ''} onChange={handleChange} />
           </div>

            {/* Ticker List */}
             <div className="space-y-1 md:col-span-2">
                <Label htmlFor="tickers">Tickers (comma-separated)</Label>
                <Textarea id="tickers" name="tickers" value={formState.tickers || ''} onChange={handleChange} rows={4} placeholder="0001.HK, 0005.HK, 0700.HK, ..."/>
             </div>

             {/* Grid Search Parameters */}
             <div className="space-y-1">
                <Label htmlFor="k_values">K Values (comma-separated integers)</Label>
                <Input id="k_values" name="k_values" value={formState.k_values || ''} onChange={handleChange} placeholder="e.g., 50, 51, 52"/>
             </div>
             <div className="space-y-1">
                 <Label htmlFor="n_values">N Values (comma-separated integers)</Label>
                 <Input id="n_values" name="n_values" value={formState.n_values || ''} onChange={handleChange} placeholder="e.g., 1, 2, 3" />
              </div>

             {/* Simulation Capital */}
             <div className="space-y-1">
                 <Label htmlFor="total_strategy_capital">Starting Capital (Base Currency)</Label>
                 {/* Keep type="text" for input, parse in handleSubmit */}
                 <Input id="total_strategy_capital" name="total_strategy_capital" type="text" inputMode='numeric' value={formState.total_strategy_capital ?? ''} onChange={handleChange} placeholder="e.g., 1000000"/>
              </div>

          {/* Add more fields here if needed */}

        </CardContent>
        <CardFooter className="flex flex-col sm:flex-row justify-between items-center pt-4"> {/* Adjust footer layout */}
           {/* Display error prominently */}
           {error && <p className="text-red-500 text-sm mb-2 sm:mb-0 sm:mr-4">{error}</p>}
          <Button type="submit" disabled={isLoading} className="w-full sm:w-auto sm:ml-auto"> {/* Full width on small, auto on larger */}
            {isLoading ? 'Starting...' : 'Run Workflow'}
          </Button>
        </CardFooter>
      </form>
    </Card>
  );
}