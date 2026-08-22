// Mock data — used only when the real backend is unreachable.
// Clearly labeled so it is never confused with real model output.

import type {
  AudioFile,
  CheckpointInfo,
  GenerateRequest,
  Job,
  ModelStatus,
  PromptProfileInfo,
} from "./api";

export const mockStatus: ModelStatus = {
  online: false,
  baseModel: "facebook/musicgen-melody-large",
  activeAdapter: "training/musicgen/lora_training/adapter.pt",
  activeCheckpoint: "LoRA",
  device: "cuda:0 (offline)",
  training: { active: false },
};

export const mockCheckpoints: CheckpointInfo[] = [
  {
    id: "lora",
    step: 2500,
    createdAt: "2026-07-26T20:06:00Z",
    recommended: true,
    notes: "Aktueller freigegebener LoRA-Stand mit 4000 genrebalancierten Clips.",
  },
];

export const mockProfiles: PromptProfileInfo[] = [
  {
    id: "jazz_lofi",
    label: "Jazz Lofi",
    description: "Jazz-Akkorde, Piano/Rhodes, weiche Drums, kontrollierter Bass.",
    basePrompt: "warm jazz lofi instrumental, mellow piano chords, brushed snare, controlled bass",
  },
  {
    id: "chillhop_lofi",
    label: "Chillhop Lofi",
    description: "Ruhiger Groove, klare Drums, wenig Shaker, entspannte Bewegung.",
    basePrompt: "calm chillhop lofi beat, soft drums, warm bass, relaxed groove, subtle texture",
  },
  {
    id: "dreamy_lofi",
    label: "Dreamy Lofi",
    description: "Weiche Flächen, sanfte Melodie, ruhige Atmosphäre.",
    basePrompt: "dreamy lofi instrumental, mellow keys, soft drums, nostalgic texture, calm mood",
  },
  {
    id: "study_lofi",
    label: "Study Lofi",
    description: "Konzentriert, gleichmäßig, wenig Störgeräusche.",
    basePrompt: "study lofi beat, calm piano, controlled bass, soft drums, minimal shaker",
  },
  {
    id: "guitar_lofi",
    label: "Guitar Lofi",
    description: "Gitarrenakzente, warmer Bass, entspannter Lofi-Groove.",
    basePrompt: "guitar lofi instrumental, clean guitar accents, warm bass, soft drums, smooth groove",
  },
  {
    id: "chill_lofi",
    label: "Chill Lofi",
    description: "Ruhig, kontrolliert, weich und musikalisch stabil.",
    basePrompt: "chill lofi instrumental, mellow piano, warm bass, soft drums, calm relaxed evening mood",
  },
];

const jobs: Job[] = [
  {
    jobId: "job_001",
    status: "completed",
    promptProfile: "chill_lofi",
    targetDurationMin: 60,
    outputFormat: "wav_mp3",
    startedAt: "2025-05-29T20:14:00Z",
    finishedAt: "2025-05-29T21:02:00Z",
    progress: 1,
    step: "done",
    audioId: "audio_001",
  },
  {
    jobId: "job_002",
    status: "completed",
    promptProfile: "jazz_lofi",
    targetDurationMin: 30,
    outputFormat: "wav",
    startedAt: "2025-05-30T11:00:00Z",
    finishedAt: "2025-05-30T11:24:00Z",
    progress: 1,
    step: "done",
    audioId: "audio_002",
  },
  {
    jobId: "job_003",
    status: "failed",
    promptProfile: "dreamy_lofi",
    targetDurationMin: 120,
    outputFormat: "mp3",
    startedAt: "2025-06-01T09:10:00Z",
    progress: 0.34,
    step: "segment_generation",
    error: "CUDA out of memory bei Segment 17",
  },
];

export function listMockJobs(): Job[] {
  return [...jobs].sort((a, b) => (a.startedAt < b.startedAt ? 1 : -1));
}
export function getMockJob(id: string): Job {
  const j = jobs.find((x) => x.jobId === id);
  if (!j) throw new Error("Job not found");
  return j;
}
export function cancelMockJob(id: string) {
  const j = jobs.find((x) => x.jobId === id);
  if (j && (j.status === "queued" || j.status === "running")) {
    j.status = "cancelled";
  }
}
export function createMockJob(body: GenerateRequest) {
  const jobId = `job_${String(jobs.length + 1).padStart(3, "0")}_mock`;
  const job: Job = {
    jobId,
    status: "queued",
    promptProfile: body.promptProfile,
    targetDurationMin: body.targetDurationMin,
    outputFormat: body.outputFormat,
    startedAt: new Date().toISOString(),
    progress: 0,
    step: "model_loading",
  };
  jobs.unshift(job);
  // Simulate progression in mock mode only.
  simulateProgress(job);
  return { jobId, status: job.status };
}

function simulateProgress(job: Job) {
  if (typeof window === "undefined") return;
  const steps: Job["step"][] = [
    "model_loading",
    "segment_generation",
    "segment_check",
    "crossfade",
    "export",
    "done",
  ];
  let i = 0;
  const tick = () => {
    if (job.status === "cancelled" || job.status === "failed") return;
    job.status = "running";
    job.progress = Math.min(1, job.progress + 0.07 + Math.random() * 0.06);
    if (job.progress > (i + 1) / steps.length) i = Math.min(steps.length - 1, i + 1);
    job.step = steps[i];
    if (job.progress >= 1) {
      job.status = "completed";
      job.progress = 1;
      job.step = "done";
      job.finishedAt = new Date().toISOString();
      return;
    }
    setTimeout(tick, 1500);
  };
  setTimeout(tick, 800);
}

export const mockAudio: AudioFile[] = [
  {
    id: "audio_001",
    title: "Chill Lofi · 20 min · LoRA",
    durationSec: 1200,
    createdAt: "2026-06-29T14:01:00Z",
    promptProfile: "chill_lofi",
    prompt: "chill lofi instrumental, mellow piano, warm bass, soft drums",
    checkpoint: "LoRA",
    formats: "wav_mp3",
    metadata: { segmentSec: 30, crossfadeSec: 3, seed: 0, normalized: 1 },
    scoreRating: {
      status: "finished",
      genre: "Chill Lofi",
      genre_key: "chillhop_lofi",
      genrestandard_scores: {
        "Technische Audioqualität": 92,
        "Musikalische Kohärenz": 87,
        "Genre-Treue": 90,
        "Übergangsqualität": 82,
        "Referenzähnlichkeit": 88,
      },
      generierte_audio_scores: {
        "Technische Audioqualität": 86,
        "Musikalische Kohärenz": 78,
        "Genre-Treue": 81,
        "Übergangsqualität": 74,
        "Referenzähnlichkeit": 79,
      },
      auswertung:
        "Der größte Unterschied liegt bei Musikalische Kohärenz: die generierte Audio liegt 9.0 Punkte unter dem Genrestandard.",
    },
  },
  {
    id: "audio_002",
    title: "Jazz Lofi · 5 min · LoRA",
    durationSec: 300,
    createdAt: "2026-06-29T13:38:00Z",
    promptProfile: "jazz_lofi",
    prompt: "warm jazz lofi instrumental, mellow piano chords, soft drums",
    checkpoint: "LoRA",
    formats: "wav",
    metadata: { segmentSec: 30, crossfadeSec: 3, normalized: 1 },
    scoreRating: {
      status: "finished",
      genre: "Jazz Lofi",
      genre_key: "jazz_lofi",
      genrestandard_scores: {
        "Technische Audioqualität": 90,
        "Musikalische Kohärenz": 84,
        "Genre-Treue": 93,
        "Übergangsqualität": 78,
        "Referenzähnlichkeit": 89,
      },
      generierte_audio_scores: {
        "Technische Audioqualität": 88,
        "Musikalische Kohärenz": 80,
        "Genre-Treue": 86,
        "Übergangsqualität": 70,
        "Referenzähnlichkeit": 82,
      },
      auswertung:
        "Der größte Unterschied liegt bei Übergangsqualität: die generierte Audio liegt 8.0 Punkte unter dem Genrestandard.",
    },
  },
];

