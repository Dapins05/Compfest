import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Gabungkan kelas Tailwind dengan resolusi konflik yang benar. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
