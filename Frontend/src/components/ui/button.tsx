/**
 * Tombol dasar.
 *
 * Komponen di folder ini mengikuti pola shadcn/ui: kodenya berada di dalam repo
 * dan bukan dependensi runtime, sehingga varian dan warnanya mengikuti token
 * pada `globals.css` alih-alih palet bawaan pustaka. Berkasnya ditulis langsung
 * karena penyiapan lewat CLI bersifat interaktif dan tidak dapat dijalankan
 * pada lingkungan tanpa terminal interaktif.
 */

import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-lg font-medium " +
    "transition-colors disabled:pointer-events-none disabled:opacity-45 " +
    "[&_svg]:pointer-events-none [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        primary: "bg-brand text-white hover:bg-brand/90 shadow-sm",
        outline:
          "border border-line-strong bg-surface text-ink hover:bg-surface-sunken",
        ghost: "text-ink-soft hover:bg-surface-sunken hover:text-ink",
        danger: "bg-reject text-white hover:bg-reject/90 shadow-sm",
      },
      size: {
        sm: "h-9 px-3 text-sm [&_svg]:size-4",
        md: "h-11 px-4 text-sm [&_svg]:size-4",
        lg: "h-13 px-6 text-base [&_svg]:size-5",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants>;

export function Button({ className, variant, size, ...props }: ButtonProps) {
  return <button className={cn(buttonVariants({ variant, size }), className)} {...props} />;
}
