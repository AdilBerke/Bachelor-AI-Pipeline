import { createFileRoute } from "@tanstack/react-router";
import { Film } from "lucide-react";
import { PageHeader } from "../components/ui-bits";

export const Route = createFileRoute("/studio/video")({
  head: () => ({
    meta: [
      { title: "LOFI.GEN · Video" },
      { name: "description", content: "Video-Generierung (LTX) – in Arbeit." },
    ],
  }),
  component: VideoPage,
});

function VideoPage() {
  return (
    <div>
      <PageHeader title="Video" description="Video-Generierung für musikalische Szenarien." />
      <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed hairline bg-surface/40 px-6 py-16 text-center">
        <Film className="h-8 w-8 text-warm" />
        <p className="text-sm text-muted-foreground">
          Noch nicht angebunden. Die LTX-Video-Pipeline aus dem Bachelor-Projekt landet hier, sobald sie
          integriert ist.
        </p>
      </div>
    </div>
  );
}
