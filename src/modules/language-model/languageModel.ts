import { ANSWER_LENGTH, getBaseCandidates, type Candidate, type PromptId } from "./languageModelData";

export const TEMPERATURES = [
  { id: "accountant", name: "Revisorn", value: 0.5, description: "Håller sig till det väntade", symbol: "▥" },
  { id: "narrator", name: "Berättaren", value: 1, description: "En lagom dos variation", symbol: "≋" },
  { id: "artist", name: "Konstnären", value: 2, description: "Tar gärna ut svängarna", symbol: "✳" }
] as const;
export type TemperatureId = (typeof TEMPERATURES)[number]["id"];
export type Phase = "ready" | "prepared" | "sampled";
export interface Draw {
  token: string;
  probability: number;
  context: readonly string[];
  temperature: TemperatureId;
}
export interface Generation {
  phase: Phase;
  tokens: readonly string[];
  input: readonly string[];
  draws: readonly Draw[];
}

export function startGeneration(): Generation {
  return { phase: "ready", tokens: [], input: [], draws: [] };
}

/** softmax(log(p) / T); at T=1 the hand-authored base distribution is unchanged. */
export function applyTemperature(candidates: readonly Candidate[], temperature: number): Candidate[] {
  if (!Number.isFinite(temperature) || temperature <= 0) throw new RangeError("Temperaturen måste vara positiv.");
  if (candidates.length === 0 || candidates.some(({ probability }) => !Number.isFinite(probability) || probability <= 0)) {
    throw new RangeError("Sannolikheterna måste vara positiva.");
  }
  const logits = candidates.map(({ probability }) => Math.log(probability) / temperature);
  const max = Math.max(...logits);
  const weights = logits.map((logit) => Math.exp(logit - max));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return candidates.map(({ token }, i) => ({ token, probability: weights[i] / total }));
}

export function candidatesAt(promptId: PromptId, tokens: readonly string[], temperatureId: TemperatureId): Candidate[] {
  const temperature = TEMPERATURES.find(({ id }) => id === temperatureId);
  if (!temperature) throw new RangeError("Okänd temperatur.");
  return applyTemperature(getBaseCandidates(promptId, tokens), temperature.value);
}

export function sampleToken(candidates: readonly Candidate[], random = Math.random): Candidate {
  const draw = random();
  if (!Number.isFinite(draw) || draw < 0 || draw >= 1) throw new RangeError("Slumptalet måste ligga i intervallet [0, 1).");
  let cumulative = 0;
  for (const candidate of candidates) {
    cumulative += candidate.probability;
    if (draw < cumulative) return candidate;
  }
  // Only covers floating-point rounding at the upper edge of a normalized table.
  return candidates[candidates.length - 1];
}

/** A click alternates between sending the complete context and drawing one token. */
export function advanceGeneration(generation: Generation, promptId: PromptId, temperature: TemperatureId, random = Math.random): Generation {
  if (generation.tokens.length === ANSWER_LENGTH) return generation;
  if (generation.phase !== "prepared") {
    return { ...generation, phase: "prepared", input: [...generation.tokens] };
  }
  const selected = sampleToken(candidatesAt(promptId, generation.input, temperature), random);
  return {
    phase: "sampled",
    input: generation.input,
    tokens: [...generation.tokens, selected.token],
    draws: [...generation.draws, { ...selected, context: [...generation.input], temperature }]
  };
}

export function joinTokens(tokens: readonly string[]): string {
  return tokens.join(" ").replace(/\s+([.,?!])/g, "$1");
}

/** Largest-remainder rounding: the five displayed percentages add to exactly 100.0. */
export function percentageLabels(candidates: readonly Candidate[]): string[] {
  const units = candidates.map(({ probability }) => Math.floor(probability * 1000));
  const ranking = candidates.map(({ probability }, index) => ({ index, remainder: probability * 1000 - units[index] }))
    .sort((a, b) => b.remainder - a.remainder);
  const remaining = 1000 - units.reduce((sum, value) => sum + value, 0);
  for (let i = 0; i < remaining; i += 1) units[ranking[i % ranking.length].index] += 1;
  return units.map((unit) => unit === 0 ? "< 0,1 %" : `${(unit / 10).toLocaleString("sv-SE", { minimumFractionDigits: 1, maximumFractionDigits: 1 })} %`);
}
