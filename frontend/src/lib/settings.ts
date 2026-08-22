// Persisted UI settings (localStorage). Backend URL + generation defaults.
import type { OutputFormat, PromptProfile } from "./api";

export interface AppSettings {
  apiUrl: string;
  apiToken: string;
  defaultProfile: PromptProfile;
  defaultDurationMin: number;
  defaultFormat: OutputFormat;
  defaultSegmentSec: 30 | 60 | 90 | 120;
  defaultCrossfadeSec: number;
  autoNormalize: boolean;
  enableMp3: boolean;
}

const KEY = "lofigen.settings.v1";
const VALID_PROFILES = [
  "jazz_lofi",
  "chillhop_lofi",
  "dreamy_lofi",
  "study_lofi",
  "guitar_lofi",
  "chill_lofi",
];

const DEFAULTS: AppSettings = {
  apiUrl: "http://127.0.0.1:8000",
  apiToken: "",
  defaultProfile: "chillhop_lofi",
  defaultDurationMin: 20,
  defaultFormat: "wav_mp3",
  defaultSegmentSec: 30,
  defaultCrossfadeSec: 3,
  autoNormalize: true,
  enableMp3: true,
};

export function getSettings(): AppSettings {
  if (typeof window === "undefined") return DEFAULTS;
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    const parsed = { ...DEFAULTS, ...JSON.parse(raw) };
    if (!VALID_PROFILES.includes(parsed.defaultProfile)) {
      parsed.defaultProfile = DEFAULTS.defaultProfile;
    }
    return parsed;
  } catch {
    return DEFAULTS;
  }
}

export function saveSettings(s: AppSettings) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(KEY, JSON.stringify(s));
  window.dispatchEvent(new CustomEvent("lofigen:settings"));
}

export function resetSettings() {
  saveSettings(DEFAULTS);
}
