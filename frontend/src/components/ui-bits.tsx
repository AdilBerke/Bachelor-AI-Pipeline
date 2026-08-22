import type { ReactNode } from "react";

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-8 flex flex-col gap-4 border-b hairline pb-6 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <div className="mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
          Lofi Studio
        </div>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-foreground sm:text-3xl">
          {title}
        </h1>
        {description && (
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
            {description}
          </p>
        )}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

export function StatCard({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "default" | "primary" | "accent" | "muted";
}) {
  const border =
    tone === "primary" ? "border-warm/40" : tone === "accent" ? "border-cool/40" : "hairline";
  return (
    <div className={`rounded-2xl border ${border} bg-surface/50 p-5 transition hover:bg-surface-2/50`}>
      <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-foreground">{value}</div>
      {hint && <div className="mt-2 text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}

export function InfoNote({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-warm/30 bg-warm/10 px-4 py-3 text-xs leading-relaxed text-foreground/80">
      {children}
    </div>
  );
}

const STATUS_STYLES: Record<string, string> = {
  queued: "bg-muted text-muted-foreground",
  running: "bg-warm/15 text-warm",
  completed: "bg-emerald-500/15 text-emerald-300",
  failed: "bg-destructive/20 text-destructive-foreground",
  cancelled: "bg-muted text-muted-foreground",
  online: "bg-emerald-500/15 text-emerald-300",
  offline: "bg-muted text-muted-foreground",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={`mono inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[10px] uppercase tracking-widest ${
        STATUS_STYLES[status] ?? "bg-muted text-muted-foreground"
      }`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current opacity-80" />
      {status}
    </span>
  );
}
