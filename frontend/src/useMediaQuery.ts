import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (notify: () => void) => {
      const list = window.matchMedia(query);
      list.addEventListener("change", notify);
      return () => list.removeEventListener("change", notify);
    },
    [query],
  );
  return useSyncExternalStore(subscribe, () => window.matchMedia(query).matches);
}

/** Ignore gestures that belong to an overlay with its own scrolling. */
const inOverlay = (target: EventTarget | null) =>
  target instanceof Element && target.closest(".sheet-backdrop, .dialog-backdrop, .review-screen") !== null;

/**
 * Show something (the header) when the user scrolls or swipes upwards, then
 * hide it again after `holdMs`. Calling `reveal` restarts the countdown.
 * When disabled it is always shown.
 */
export function useRevealOnScrollUp(enabled: boolean, holdMs = 3000) {
  const [shown, setShown] = useState(true);
  const timer = useRef<number | undefined>(undefined);

  const reveal = useCallback(() => {
    setShown(true);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setShown(false), holdMs);
  }, [holdMs]);

  useEffect(() => {
    if (!enabled) {
      window.clearTimeout(timer.current);
      setShown(true);
      return;
    }
    reveal(); // Show it briefly at first so people learn it's there.

    let touchY: number | null = null;
    let lastScroll = window.scrollY;
    const onWheel = (event: WheelEvent) => {
      if (event.deltaY < 0 && !inOverlay(event.target)) reveal();
    };
    const onTouchStart = (event: TouchEvent) => {
      touchY = inOverlay(event.target) ? null : event.touches[0].clientY;
    };
    const onTouchMove = (event: TouchEvent) => {
      if (touchY !== null && event.touches[0].clientY - touchY > 24) {
        touchY = null;
        reveal();
      }
    };
    const onScroll = () => {
      if (window.scrollY < lastScroll) reveal();
      lastScroll = window.scrollY;
    };

    window.addEventListener("wheel", onWheel, { passive: true });
    window.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchmove", onTouchMove, { passive: true });
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.clearTimeout(timer.current);
      window.removeEventListener("wheel", onWheel);
      window.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("scroll", onScroll);
    };
  }, [enabled, reveal]);

  return [shown, reveal] as const;
}
