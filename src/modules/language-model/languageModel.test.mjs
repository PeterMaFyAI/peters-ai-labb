import assert from "node:assert/strict";
import test from "node:test";
import { registerHooks } from "node:module";
import { ANSWER_LENGTH, PROMPTS, getBaseCandidates } from "./languageModelData.ts";

// Vite resolves extensionless TS imports; give Node's native TS test runner the
// same resolution for this module without changing the application's tsconfig.
registerHooks({
  resolve(specifier, context, nextResolve) {
    return nextResolve(specifier === "./languageModelData" ? "./languageModelData.ts" : specifier, context);
  }
});
const {
  TEMPERATURES, advanceGeneration, applyTemperature, candidatesAt, joinTokens,
  percentageLabels, sampleToken, startGeneration
} = await import("./languageModel.ts");

const close = (a, b, tolerance = 1e-12) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
const entropy = (candidates) => -candidates.reduce((sum, { probability: p }) => sum + p * Math.log(p), 0);

function everyContext(visit) {
  let contexts = 0;
  let leaves = 0;
  function walk(id, prefix) {
    if (prefix.length === ANSWER_LENGTH) { leaves += 1; return; }
    const candidates = getBaseCandidates(id, prefix);
    contexts += 1;
    visit(candidates, id, prefix);
    for (const { token } of candidates) walk(id, [...prefix, token]);
  }
  for (const { id } of PROMPTS) walk(id, []);
  return { contexts, leaves };
}

test("the complete database covers 2,343 contexts and 9,375 five-token answers", () => {
  const counts = everyContext((candidates, id, prefix) => {
    assert.equal(candidates.length, 5, `${id}: ${prefix}`);
    assert.equal(new Set(candidates.map(({ token }) => token)).size, 5);
    close(candidates.reduce((sum, { probability }) => sum + probability, 0), 1);
    for (const { token, probability } of candidates) {
      assert.match(token, /^(?:\p{L}+|\d+|[.,?!])$/u);
      assert.ok(Number.isFinite(probability) && probability > 0 && probability < 1);
    }
  });
  assert.deepEqual(counts, { contexts: 2343, leaves: 9375 });
  assert.throws(() => getBaseCandidates("invalid", []), RangeError);
  assert.throws(() => getBaseCandidates("capital", ["saknas"]), RangeError);
  assert.throws(() => getBaseCandidates("capital", Array(5).fill("Stockholm")), RangeError);
});

test("higher temperature flattens every reachable table while keeping its token ranking", () => {
  everyContext((candidates) => {
    const transformed = TEMPERATURES.map(({ value }) => applyTemperature(candidates, value));
    assert.ok(entropy(transformed[0]) < entropy(transformed[1]));
    assert.ok(entropy(transformed[1]) < entropy(transformed[2]));
    const ordering = (items) => [...items].sort((a, b) => b.probability - a.probability).map(({ token }) => token);
    for (const items of transformed) {
      close(items.reduce((sum, { probability }) => sum + probability, 0), 1);
      assert.deepEqual(ordering(items), ordering(candidates));
    }
    candidates.forEach(({ probability }, i) => close(transformed[1][i].probability, probability));
  });
  for (const temperature of [0, -1, NaN, Infinity]) assert.throws(() => applyTemperature(getBaseCandidates("capital", []), temperature), RangeError);
});

test("sampling uses all five probability intervals, including very unlikely tokens", () => {
  everyContext((candidates) => {
    for (const { value } of TEMPERATURES) {
      const transformed = applyTemperature(candidates, value);
      let start = 0;
      for (const candidate of transformed) {
        assert.equal(sampleToken(transformed, () => start + candidate.probability / 2).token, candidate.token);
        start += candidate.probability;
      }
    }
  });
  for (const draw of [-0.01, 1, NaN]) assert.throws(() => sampleToken(getBaseCandidates("capital", []), () => draw), RangeError);
});

test("repeated draws follow the distribution and can produce different answers", () => {
  const candidates = candidatesAt("capital", [], "narrator");
  const counts = new Map(candidates.map(({ token }) => [token, 0]));
  // Stratified draws verify the actual distribution without a flaky random test.
  for (let i = 0; i < 10000; i += 1) {
    const { token } = sampleToken(candidates, () => (i + 0.5) / 10000);
    counts.set(token, counts.get(token) + 1);
  }
  for (const { token, probability } of candidates) close(counts.get(token) / 10000, probability, 0.00011);
  assert.equal([...counts.values()].filter((count) => count > 0).length, 5);
  const choices = TEMPERATURES.map(({ id }) => sampleToken(candidatesAt("capital", [], id), () => 0.75).token);
  assert.deepEqual(choices, ["Stockholm", "Sveriges", "Huvudstaden"]);
});

test("ten clicks alternate full-context input and random output, then stop at five tokens", () => {
  for (const { id } of PROMPTS) {
    let state = startGeneration();
    let draws = 0;
    for (let step = 0; step < ANSWER_LENGTH; step += 1) {
      const previous = [...state.tokens];
      state = advanceGeneration(state, id, "narrator", () => { throw new Error("Input must not draw a token"); });
      assert.equal(state.phase, "prepared");
      assert.deepEqual(state.input, previous);
      assert.deepEqual(state.tokens, previous);
      state = advanceGeneration(state, id, "narrator", () => { draws += 1; return 0; });
      assert.equal(state.phase, "sampled");
      assert.equal(state.tokens.length, step + 1);
      assert.deepEqual(state.draws[step].context, previous);
      assert.deepEqual(state.input, previous, "The newly emitted token is only fed back on the next input step");
    }
    assert.equal(draws, 5);
    assert.equal(advanceGeneration(state, id, "artist"), state);
    assert.deepEqual(startGeneration(), { phase: "ready", tokens: [], input: [], draws: [] });
  }
});

test("the current full prefix and chosen temperature determine the next draw", () => {
  let state = advanceGeneration(startGeneration(), "capital", "narrator");
  state = advanceGeneration(state, "capital", "accountant", () => 0.75);
  assert.equal(state.tokens[0], "Stockholm");
  assert.equal(state.draws[0].temperature, "accountant");
  state = advanceGeneration(state, "capital", "artist");
  assert.deepEqual(state.input, ["Stockholm"]);
  assert.notDeepEqual(candidatesAt("capital", state.input, "artist"), candidatesAt("capital", [], "artist"));
  const before = structuredClone(state);
  state = advanceGeneration(state, "capital", "artist", () => 0);
  assert.equal(state.tokens[1], "är");
  assert.deepEqual(before.tokens, ["Stockholm"]);
  assert.equal(state.draws[0].temperature, "accountant");
  assert.equal(state.draws[1].temperature, "artist");
  assert.notDeepEqual(getBaseCandidates("story", ["hittade", "en", "katt", "som"]), getBaseCandidates("story", ["hittade", "en", "robot", "som"]));
  const mathAfterPeriod = getBaseCandidates("sum", ["Svaret", "är", "4", "."]).map(({ token }) => token);
  assert.ok(mathAfterPeriod.includes("Summan"));
  assert.ok(!mathAfterPeriod.includes("Staden"));
});

test("token formatting handles punctuation and tiny probabilities honestly", () => {
  assert.equal(joinTokens(["Stockholm", ",", "Sveriges", "huvudstad", "."]), "Stockholm, Sveriges huvudstad.");
  assert.equal(joinTokens(["4", "!", "Det", "är", "rätt"]), "4! Det är rätt");
  const labels = percentageLabels(candidatesAt("capital", [], "narrator"));
  assert.deepEqual(labels, ["68,0 %", "18,0 %", "9,0 %", "4,0 %", "1,0 %"]);
  everyContext((candidates) => {
    const adjusted = applyTemperature(candidates, 0.5);
    const formatted = percentageLabels(adjusted);
    assert.equal(formatted.length, 5);
    assert.ok(formatted.every((label) => label !== "0,0 %"));
    // A label < 0.1% represents a positive probability rounded to zero units.
    close(formatted.reduce((sum, label) => sum + (label.startsWith("<") ? 0 : Number(label.replace(" %", "").replace(",", "."))), 0), 100, 1e-9);
  });
});
