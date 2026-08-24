import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "@/lib/utils";

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-xl border border-line bg-surface shadow-[0_1px_2px_rgba(15,21,31,0.04)]",
        className,
      )}
      {...props}
    />
  );
}

interface CardHeaderProps {
  title: string;
  description?: string;
  icon?: ReactNode;
  action?: ReactNode;
  className?: string;
}

export function CardHeader({ title, description, icon, action, className }: CardHeaderProps) {
  return (
    <div
      className={cn(
        "flex items-start justify-between gap-3 border-b border-line px-4 py-4 sm:px-5",
        className,
      )}
    >
      <div className="flex min-w-0 items-start gap-3">
        {icon ? <span className="mt-0.5 text-ink-faint">{icon}</span> : null}
        <div className="min-w-0">
          <h2 className="text-sm font-semibold tracking-tight text-ink">{title}</h2>
          {description ? (
            <p className="mt-0.5 text-xs leading-relaxed text-ink-faint">{description}</p>
          ) : null}
        </div>
      </div>
      {action}
    </div>
  );
}

export function CardBody({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-4 sm:p-5", className)} {...props} />;
}
