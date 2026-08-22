import { Link, useRouterState } from "@tanstack/react-router";
import {
  ArrowLeft,
  AudioLines,
  Cpu,
  Film,
  Library,
  Link as LinkIcon,
  ListChecks,
  Waves,
} from "lucide-react";

type NavItem = {
  to:
    | "/studio/import"
    | "/studio/generate"
    | "/studio/video"
    | "/studio/library"
    | "/studio/jobs"
    | "/studio/model";
  label: string;
  icon: typeof LinkIcon;
  exact?: boolean;
};

const nav: NavItem[] = [
  { to: "/studio/import", label: "Quellen", icon: LinkIcon, exact: true },
  { to: "/studio/generate", label: "Audio", icon: Waves },
  { to: "/studio/video", label: "Video", icon: Film },
  { to: "/studio/library", label: "Ausgaben", icon: Library },
  { to: "/studio/jobs", label: "Läufe", icon: ListChecks },
  { to: "/studio/model", label: "Steuerung", icon: Cpu },
];

export function StudioSidebar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  return (
    <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r hairline bg-surface/30 px-4 py-6 md:flex">
      <Link to="/studio/import" className="mono flex items-center gap-2 px-1 text-sm font-semibold">
        <span className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-warm/40 bg-warm/10 text-warm">
          <AudioLines className="h-3.5 w-3.5" />
        </span>
        <span>
          LOFI.GEN
          <span className="block text-[10px] font-normal normal-case tracking-normal text-muted-foreground">
            Studio
          </span>
        </span>
      </Link>

      <nav className="mt-8 flex flex-1 flex-col gap-1">
        {nav.map((item) => {
          const active = item.exact ? pathname === item.to : pathname.startsWith(item.to);
          const Icon = item.icon;
          return (
            <Link
              key={item.to}
              to={item.to}
              className={[
                "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition",
                active
                  ? "border border-warm/40 bg-warm/10 text-warm"
                  : "border border-transparent text-muted-foreground hover:bg-surface-2/60 hover:text-foreground",
              ].join(" ")}
            >
              <Icon className="h-4 w-4" />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="mt-6 space-y-3 border-t hairline pt-4">
        <div className="mono flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-muted-foreground">
          <span className="h-1.5 w-1.5 rounded-full bg-warm animate-pulse-soft" />
          lokal · MusicGen + LoRA
        </div>
        <Link
          to="/"
          className="mono flex items-center gap-2 text-[11px] uppercase tracking-widest text-muted-foreground transition hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Zur Website
        </Link>
      </div>
    </aside>
  );
}

export function StudioMobileNav() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  return (
    <div className="sticky top-0 z-30 flex items-center gap-1 overflow-x-auto border-b hairline bg-background/95 px-3 py-2 backdrop-blur md:hidden">
      <Link
        to="/"
        className="mr-1 inline-flex shrink-0 items-center gap-1 rounded-full border hairline px-2.5 py-1.5 text-xs text-muted-foreground"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
      </Link>
      {nav.map((item) => {
        const active = item.exact ? pathname === item.to : pathname.startsWith(item.to);
        const Icon = item.icon;
        return (
          <Link
            key={item.to}
            to={item.to}
            className={[
              "mono inline-flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] uppercase tracking-widest",
              active ? "bg-warm/15 text-warm" : "text-muted-foreground",
            ].join(" ")}
          >
            <Icon className="h-3.5 w-3.5" />
            <span>{item.label}</span>
          </Link>
        );
      })}
    </div>
  );
}
