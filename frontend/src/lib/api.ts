// Central API layer for LofiGen AI.
// All HTTP calls live here. The backend runs locally on the AI machine.
// If no backend is reachable, callers fall back to mock data (see mockData.ts).

import { getSettings } from "./settings";
import * as mock from "./mockData";

export type PromptProfile = string;

export const PROFILE_LABELS: Record<PromptProfile, string> = {
  jazz_lofi: "Jazz Lofi",
  chillhop_lofi: "Chillhop Lofi",
  dreamy_lofi: "Dreamy Lofi",
  study_lofi: "Study Lofi",
  guitar_lofi: "Guitar Lofi",
  chill_lofi: "Chill Lofi",
};

export function profileLabel(profile: PromptProfile | string): string {
  return PROFILE_LABELS[profile as PromptProfile] ?? profile.replaceAll("_", " ");
}

export type OutputFormat = "wav" | "mp3" | "wav_mp3";

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type JobStep =
  | "model_loading"
  | "segment_generation"
  | "segment_check"
  | "crossfade"
  | "export"
  | "done";

export interface ModelStatus {
  online: boolean;
  baseModel: string;
  activeAdapter: string;
  activeCheckpoint: string;
  device: string;
  vramUsedMb?: number;
  training: { active: boolean; step?: number; loss?: number };
}

export interface PromptProfileInfo {
  id: PromptProfile;
  label: string;
  description: string;
  basePrompt: string;
}

export interface ProjectGenre {
  id: string;
  label: string;
  description: string;
  basePrompt: string;
}

export interface ProjectDatasetGenre {
  selected?: number;
  target?: number;
  missing?: number;
  rejected?: number;
  [key: string]: unknown;
}

export interface ProjectDataset {
  path: string;
  ready: boolean;
  selectedTotal: number;
  targetTotal: number;
  missingTotal: number;
  rejectedTotal: number;
  genres: Record<string, ProjectDatasetGenre>;
}

export interface ProjectLora {
  adapter: string;
  adapterExists: boolean;
  status: string;
  checkpointStep?: number | string | null;
  runName?: string | null;
}

export interface ProjectOverview {
  status: ModelStatus;
  dataset: ProjectDataset;
  lora: ProjectLora;
  genres: ProjectGenre[];
}

export interface ProjectRunResponse {
  jobId: string;
  status: JobStatus;
}

export interface Top10Candidate {
  rang?: number;
  genre?: string;
  titel?: string;
  kanal?: string;
  url?: string;
  video_id?: string;
  gesamt_score?: number;
  view_count?: number;
  like_count?: number;
  dauer?: string;
  status?: string;
}

export interface CheckpointInfo {
  id: string;
  step: number;
  createdAt: string;
  recommended?: boolean;
  notes?: string;
}

export const SCORE_CATEGORIES = [
  "Technische Audioqualität",
  "Musikalische Kohärenz",
  "Genre-Treue",
  "Übergangsqualität",
  "Referenzähnlichkeit",
] as const;

export type ScoreCategory = (typeof SCORE_CATEGORIES)[number];
export type ScoreMap = Record<string, number>;

export interface AudioScoreDifference {
  Kategorie: string;
  Genrestandard: number;
  "Generierte Audio": number;
  Differenz: number;
}

export interface AudioScoreRating {
  status: string;
  genre: string;
  genre_key?: string;
  kategorien?: string[];
  genrestandard_scores: ScoreMap;
  generierte_audio_scores: ScoreMap;
  groesste_unterschiede?: AudioScoreDifference[];
  auswertung?: string;
  html?: string;
  json?: string;
  csv?: string;
}

export interface GenerateRequest {
  targetDurationMin: number;
  durationInput?: string;
  promptProfile: PromptProfile;
  customPrompt?: string;
  targetBpm?: number;
  instruments?: string;
  segmentDurationSec: 30 | 60 | 90 | 120;
  crossfadeSec: number;
  seed?: number;
  normalize: boolean;
  outputFormat: OutputFormat;
  adapterPath: string;
  githubPush?: boolean;
}

export interface Job {
  jobId: string;
  status: JobStatus;
  promptProfile: PromptProfile;
  targetDurationMin: number;
  outputFormat: OutputFormat;
  startedAt: string;
  finishedAt?: string;
  progress: number; // 0..1
  step: JobStep;
  etaSec?: number;
  audioId?: string;
  error?: string;
  kind?: string;
  logPath?: string;
  lastLines?: string[];
}

export interface AudioFile {
  id: string;
  title: string;
  durationSec: number;
  createdAt: string;
  promptProfile: PromptProfile;
  prompt: string;
  checkpoint: string;
  formats: OutputFormat;
  wavUrl?: string;
  mp3Url?: string;
  reportUrl?: string;
  metadata?: Record<string, string | number>;
  scoreRating?: AudioScoreRating;
  score_rating?: AudioScoreRating;
}

export interface YoutubeMp3ImportRequest {
  url: string;
  genre?: string;
  title?: string;
  rightsConfirmed: boolean;
  datasetName?: string;
  caption?: string;
  maxClips?: number;
  clipHopSec?: number;
  seed?: number;
}

export interface YoutubeMp3ImportJob {
  job_id: string;
  status: JobStatus;
  stage: string;
  progress_percent: number;
  clip_total: number;
  clip_current: number;
  clip_accepted: number;
  clip_rejected: number;
  clip_last_status?: string;
  clip_last_reason?: string;
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  video_id?: string;
  url?: string;
  title?: string;
  audio_path?: string;
  audio_exists?: boolean;
  file_size_bytes?: number;
  metadata_path?: string;
  build_clips: boolean;
  dataset_path?: string;
  dataset_summary?: Record<string, unknown>;
  error?: string;
}

export interface SourceAudio {
  videoId: string;
  title: string;
  url: string;
  audioExists: boolean;
  audioPath: string;
  fileSizeBytes: number;
  downloadedAt: string;
  clipDirs: number;
  manual: boolean;
}

export interface DeleteSourceResult {
  videoId: string;
  removedAudioFiles: number;
  removedMetadataEntries: number;
  removedClipDirs: number;
  removedManifestRows: number;
}

export interface Top10StatusEntry {
  rang: number | null;
  videoId: string;
  titel: string;
  kanal: string;
  url: string;
  viewCount: number | null;
  likeCount: number | null;
  status: "im_datensatz" | "manuell_noetig" | "offen";
  grund: string | null;
}

export interface Top10StatusGenre {
  genreKey: string;
  label: string;
  entries: Top10StatusEntry[];
}

export interface PruneNonTop10Result {
  removedCount: number;
  removedVideoIds: string[];
  keptManual: number;
  keptTop10: number;
  aborted?: boolean;
  grund?: string;
}

export interface UploadSourceResult {
  videoId: string;
  title: string;
  audioPath: string;
  fileSizeBytes: number;
}

// ---- transport ----

function baseUrl(): string {
  return getSettings().apiUrl.replace(/\/+$/, "");
}

function withToken(url: string): string {
  const token = getSettings().apiToken;
  if (!token || !url.startsWith("/api/")) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}token=${encodeURIComponent(token)}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const settings = getSettings();
  const url = `${baseUrl()}${path}`;
  const tokenHeader = settings.apiToken ? { "X-Api-Key": settings.apiToken } : {};
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...tokenHeader, ...(init?.headers ?? {}) },
  });
  if (!res.ok) throw new Error(`API ${res.status} on ${path}`);
  return (await res.json()) as T;
}

// Wrap a request so a network failure transparently falls back to mock data.
// This keeps the UI usable while the local backend is offline. Real data
// always wins when the backend is reachable.
async function withMock<T>(fn: () => Promise<T>, fallback: () => T): Promise<T> {
  try {
    return await fn();
  } catch (err) {
    console.warn("[api] backend unreachable, using mock data:", (err as Error).message);
    return fallback();
  }
}

// ---- endpoints ----

export const api = {
  absoluteUrl: (path: string) => (path.startsWith("/api/") ? `${baseUrl()}${withToken(path)}` : path),

  status: () =>
    withMock(
      () => request<ModelStatus>("/api/status"),
      () => mock.mockStatus,
    ),

  models: () =>
    withMock(
      () => request<{ baseModel: string; checkpoints: CheckpointInfo[] }>("/api/models"),
      () => ({ baseModel: mock.mockStatus.baseModel, checkpoints: mock.mockCheckpoints }),
    ),

  promptProfiles: () =>
    withMock(
      () => request<PromptProfileInfo[]>("/api/prompt-profiles"),
      () => mock.mockProfiles,
    ),

  projectOverview: () =>
    withMock(
      () => request<ProjectOverview>("/api/project/overview"),
      () => ({
        status: mock.mockStatus,
        dataset: {
          path: "daten/processed/lora_training",
          ready: false,
          selectedTotal: 0,
          targetTotal: 5000,
          missingTotal: 5000,
          rejectedTotal: 0,
          genres: {},
        },
        lora: {
          adapter: mock.mockStatus.activeAdapter,
          adapterExists: false,
          status: "offline",
        },
        genres: mock.mockProfiles,
      }),
    ),

  addProjectGenre: (body: {
    label: string;
    caption?: string;
    keywords?: string;
    required?: string;
  }) =>
    request<{ ok: boolean; genre: Record<string, unknown> }>("/api/project/genres", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  runProjectAction: (action: string, body: Record<string, unknown> = {}) =>
    request<ProjectRunResponse>(`/api/project/run/${action}`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  top10Results: () =>
    withMock(
      () => request<Top10Candidate[]>("/api/project/top10-results"),
      () => [],
    ),

  generate: (body: GenerateRequest) =>
    withMock(
      () =>
        request<{ jobId: string; status: JobStatus }>("/api/generate", {
          method: "POST",
          body: JSON.stringify(body),
        }),
      () => mock.createMockJob(body),
    ),

  jobs: () =>
    withMock(
      () => request<Job[]>("/api/jobs"),
      () => mock.listMockJobs(),
    ),
  job: (id: string) =>
    withMock(
      () => request<Job>(`/api/jobs/${id}?client=lofigen`),
      () => mock.getMockJob(id),
    ),
  cancelJob: (id: string) =>
    withMock(
      () => request<{ ok: true }>(`/api/jobs/${id}/cancel`, { method: "POST" }),
      () => {
        mock.cancelMockJob(id);
        return { ok: true as const };
      },
    ),

  audio: () =>
    withMock(
      () => request<AudioFile[]>("/api/audio"),
      () => mock.mockAudio,
    ),
  audioDownloadUrl: (id: string, fmt: "wav" | "mp3") =>
    `${baseUrl()}${withToken(`/api/audio/${id}/download?format=${fmt}`)}`,

  importYoutubeMp3: (body: YoutubeMp3ImportRequest) =>
    request<{ job_id: string; job: YoutubeMp3ImportJob }>("/api/musicgen/youtube-mp3", {
      method: "POST",
      body: JSON.stringify({
        url: body.url,
        genre: body.genre,
        title: body.title,
        rightsConfirmed: body.rightsConfirmed,
        buildClips: body.buildClips,
        datasetName: body.datasetName,
        caption: body.caption,
        maxClips: body.maxClips,
        clipHopSec: body.clipHopSec,
        seed: body.seed,
      }),
    }),

  youtubeMp3Job: (id: string) =>
    request<{ job_id: string; job: YoutubeMp3ImportJob }>(`/api/musicgen/youtube-mp3/${id}`),

  sources: () =>
    withMock(
      () => request<SourceAudio[]>("/api/sources"),
      () => [],
    ),
  deleteSource: (videoId: string) =>
    request<DeleteSourceResult>(`/api/sources/${videoId}/delete`, { method: "POST" }),
  pruneNonTop10: () =>
    request<PruneNonTop10Result>("/api/sources/prune-non-top10", { method: "POST" }),
  top10Status: () =>
    withMock(
      () => request<Top10StatusGenre[]>("/api/sources/top10-status"),
      () => [],
    ),
  uploadSource: async (file: File, genre: string): Promise<UploadSourceResult> => {
    const settings = getSettings();
    const params = new URLSearchParams({ filename: file.name, genre });
    const url = `${baseUrl()}${withToken(`/api/sources/upload?${params.toString()}`)}`;
    const tokenHeader = settings.apiToken ? { "X-Api-Key": settings.apiToken } : {};
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": file.type || "application/octet-stream", ...tokenHeader },
      body: file,
    });
    if (!res.ok) throw new Error(`API ${res.status} on /api/sources/upload`);
    return (await res.json()) as UploadSourceResult;
  },
};
