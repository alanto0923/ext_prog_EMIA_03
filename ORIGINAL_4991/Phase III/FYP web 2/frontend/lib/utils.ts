// frontend/lib/utils.ts
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import { format } from 'date-fns'; // Use date-fns for reliable date formatting

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Function to format dates nicely
export function formatDate(dateString?: string | Date | null): string {
  if (!dateString) return 'N/A';
  try {
    const date = new Date(dateString);
    // Check if date is valid after parsing
    if (isNaN(date.getTime())) {
        return 'Invalid Date';
    }
    return format(date, 'yyyy-MM-dd');
  } catch (error) {
    return 'Invalid Date';
  }
}

// Function to format numbers (basic)
export function formatNumber(
    value?: number | string | null,
    options: Intl.NumberFormatOptions = {}
): string {
    if (value === null || value === undefined || value === '' || value === 'N/A') return 'N/A';
    const num = Number(value);
    if (isNaN(num)) return 'N/A'; // Was not a number

    // Sensible defaults
    const defaultOptions: Intl.NumberFormatOptions = {
        maximumFractionDigits: 4, // Default max decimals
        ...options, // User options override defaults
    };

    // Specific handling for percentages if needed via options
    if (defaultOptions.style === 'percent' && defaultOptions.maximumFractionDigits === undefined) {
        defaultOptions.maximumFractionDigits = 2; // Sensible default for percentages
    }
     // Specific handling for currency if needed via options
     if (defaultOptions.style === 'currency' && defaultOptions.maximumFractionDigits === undefined) {
        defaultOptions.maximumFractionDigits = 0; // Sensible default for currency (no cents)
    }


    try {
        return new Intl.NumberFormat('en-US', defaultOptions).format(num);
    } catch (e) {
        console.error("Formatting error:", e);
        return 'N/A';
    }
}

// Function to construct full URL for result files
export function getResultFileUrl(baseUrl: string | undefined | null, relativePath: string | undefined | null): string | null {
    if (!baseUrl || !relativePath) {
        return null;
    }
    // Ensure no double slashes
    const cleanBase = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
    const cleanRelative = relativePath.startsWith('/') ? relativePath.slice(1) : relativePath;
    return `${cleanBase}/${cleanRelative}`;
}