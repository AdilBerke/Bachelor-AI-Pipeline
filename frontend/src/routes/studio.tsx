import { Outlet, createFileRoute } from "@tanstack/react-router";
import { StudioSidebar, StudioMobileNav } from "../components/StudioSidebar";

export const Route = createFileRoute("/studio")({
  head: () => ({
    meta: [
      { title: "LOFI.GEN · Studio" },
      {
        name: "description",
        content: "Lokale Steuerung der Lo-Fi Pipeline: Generierung, Training, Bewertung, Import.",
      },
    ],
  }),
  component: StudioLayout,
});

function StudioLayout() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <StudioMobileNav />
      <div className="mx-auto flex max-w-[1600px]">
        <StudioSidebar />
        <main className="min-w-0 flex-1 px-5 py-8 sm:px-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
