import { cva, type VariantProps } from "class-variance-authority";
import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-medium",
  {
    variants: {
      tone: {
        neutral: "border-line bg-surface-sunken text-ink-soft",
        brand: "border-brand-line bg-brand-soft text-brand",
        pass: "border-pass-line bg-pass-soft text-pass",
        reject: "border-reject-line bg-reject-soft text-reject",
        hold: "border-hold-line bg-hold-soft text-hold",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export type BadgeProps = HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>;

export function Badge({ className, tone, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}
