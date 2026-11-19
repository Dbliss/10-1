import { send } from "../transport";
import type { Situation } from "./types";

/** Deal one random bidding situation. The answer comes with it, kept hidden by the UI. */
export async function fetchSituation(): Promise<Situation> {
  const { status, data } = await send("/api/bid-game/round", "POST");
  if (status >= 400) throw new Error(data?.error ?? "The game server could not deal a hand.");
  return data as Situation;
}
