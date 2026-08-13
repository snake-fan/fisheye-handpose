import { useCallback, useSyncExternalStore } from "react";

export interface UrlState {
  run: string;
  run_offset: string;
  frame: string;
  offset: string;
  stage: string;
  track: string;
  status: string;
  q: string;
}

const keys = ["run", "run_offset", "frame", "offset", "stage", "track", "status", "q"] as const;

function readUrlState(): UrlState {
  const params = new URLSearchParams(window.location.search);
  return Object.fromEntries(keys.map((key) => [key, params.get(key) ?? ""])) as unknown as UrlState;
}

function snapshot(): string {
  return window.location.href;
}

function subscribe(listener: () => void): () => void {
  window.addEventListener("popstate", listener);
  window.addEventListener("urlstatechange", listener);
  return () => {
    window.removeEventListener("popstate", listener);
    window.removeEventListener("urlstatechange", listener);
  };
}

export function useUrlState(): [UrlState, (patch: Partial<UrlState>) => void] {
  useSyncExternalStore(subscribe, snapshot, snapshot);
  const state = readUrlState();

  const update = useCallback((patch: Partial<UrlState>) => {
    const url = new URL(window.location.href);
    for (const [key, value] of Object.entries(patch)) {
      if (value) url.searchParams.set(key, value);
      else url.searchParams.delete(key);
    }
    window.history.pushState({}, "", url);
    window.dispatchEvent(new Event("urlstatechange"));
  }, []);

  return [state, update];
}
