## Ziel

Alles wieder privat: Agent-Integration (MCP) entfernen, Anmeldung zurücknehmen, Seite nicht öffentlich veröffentlichen. Die Website selbst (Hero, Pipeline, Evaluation, Audio-Player, Team) bleibt inhaltlich unverändert.

## Was passiert

**1. MCP-Server komplett entfernen**
- Löschen: `src/lib/mcp/` (Server-Definition, `data.ts`, alle drei Tools)
- Löschen der generierten Routen: `src/routes/mcp.ts`, `src/routes/[.mcp]/list-tools.ts`, `src/routes/[.mcp]/invoke-tool/$tool.ts`, `src/routes/[.well-known]/oauth-protected-resource.ts`
- Löschen: `.lovable/mcp/manifest.json`
- `vite.config.ts`: `mcpPlugin()` und dessen Import entfernen
- Pakete `@lovable.dev/mcp-js` entfernen (sowie der Eintrag in `bunfig.toml`, falls gesetzt)

**2. Anmeldung zurücknehmen**
- Löschen: `src/routes/auth.tsx`, `src/routes/[.]lovable.oauth.consent.tsx`, `src/components/AuthNavButton.tsx`
- `src/routes/index.tsx`: den Anmelde-Button aus der Navigation entfernen (Rest der Nav bleibt)
- `src/start.ts`: `attachSupabaseAuth` aus `functionMiddleware` entfernen (CSRF- und Error-Middleware bleiben)

Hinweis: Die Backend-Verbindung selbst (Lovable Cloud) bleibt technisch bestehen, wird aber von der App nicht mehr genutzt – es gibt keinen Login und keine Datenbanktabellen. Wenn du willst, kann ich zusätzlich die Google-Anmeldung im Backend deaktivieren.

**3. Nicht öffentlich veröffentlichen**
- Die Seite ist aktuell nicht veröffentlicht – es passiert also nichts weiter. Sie bleibt nur über die Vorschau erreichbar. Falls du später veröffentlichst und die Sichtbarkeit „privat“ (nur Workspace-Mitglieder) willst, benötigt das einen Business/Enterprise-Plan; darauf weise ich dann hin.

## Prüfung

- TypeScript-Check und Route-Generierung müssen sauber durchlaufen
- Vorschau öffnen: Startseite rendert vollständig, keine Konsolenfehler, `/mcp` und `/auth` existieren nicht mehr
