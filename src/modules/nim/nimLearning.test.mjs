import assert from "node:assert/strict";
import test from "node:test";
import {
  INITIAL_STICKS,
  chooseAction,
  createQTable,
  createVisitTable,
  greedyActions,
  optimalActions,
  performTrainingStep,
  startTrainingEpisode,
  validActions,
} from "./nimLearning.ts";

const params = { alpha: 0.25, gamma: 0.9, epsilon: 0 };
const episodeAt = (state) => ({ ...startTrainingEpisode("ai"), state });
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);

function sequence(...numbers) {
  let index = 0;
  return () => {
    assert.ok(index < numbers.length, "Unexpected random draw");
    return numbers[index++];
  };
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

test("tables start at zero with independent rows and only 1–3 legal sticks", () => {
  assert.equal(INITIAL_STICKS, 12);
  const table = createQTable();
  const visits = createVisitTable();
  assert.deepEqual(table, Array.from({ length: 13 }, () => [0, 0, 0]));
  assert.deepEqual(visits, table);
  table[1][0] = 9;
  assert.equal(table[2][0], 0);
  assert.equal(visits[1][0], 0);
  for (let state = 0; state <= 12; state += 1) {
    assert.deepEqual(validActions(state), [1, 2, 3].filter((action) => action <= state));
  }
  for (const state of [-1, 13, 1.5, NaN]) assert.throws(() => validActions(state), RangeError);
  assert.throws(() => chooseAction(table, 0), RangeError);
});

test("the gold strategy agrees with exhaustive misère minimax in every state", () => {
  function winningActions(state) {
    return validActions(state).filter((action) => action < state && winningActions(state - action).length === 0);
  }
  for (let state = 0; state <= 12; state += 1) {
    assert.deepEqual(optimalActions(state), winningActions(state), `state ${state}`);
  }
  for (const state of [1, 5, 9]) assert.deepEqual(optimalActions(state), []);
  assert.deepEqual(optimalActions(12), [3]);
});

test("greedy play ignores illegal cells, randomizes ties, and epsilon explores legal moves", () => {
  const table = createQTable();
  table[2] = [-0.7, -0.7, 999];
  assert.deepEqual(greedyActions(table, 2), [1, 2]);
  assert.equal(chooseAction(table, 2, 0, sequence(0)), 1);
  assert.equal(chooseAction(table, 2, 0, sequence(0.99)), 2);
  table[12] = [0, 1, 0];
  assert.equal(chooseAction(table, 12, 0.2, sequence(0.5)), 2);
  assert.equal(chooseAction(table, 12, 0.2, sequence(0.1, 0.99)), 3);
  assert.equal(chooseAction(table, 1, 1, sequence(0.5)), 1);
});

test("either player can start and the opening opponent move is uniformly selected", () => {
  assert.deepEqual(startTrainingEpisode("ai", sequence()), {
    state: 12, done: false, winner: null, starts: "ai", openingOpponentAction: null,
  });
  [0, 0.4, 0.99].forEach((draw, index) => {
    const episode = startTrainingEpisode("opponent", sequence(draw));
    assert.equal(episode.openingOpponentAction, index + 1);
    assert.equal(episode.state, 11 - index);
    assert.equal(episode.starts, "opponent");
    assert.equal(episode.done, false);
  });
});

test("taking the last stick loses and a terminal target never bootstraps", () => {
  const table = createQTable();
  table[0] = [999, 999, 999];
  table[1][0] = 0.4;
  const result = performTrainingStep(table, createVisitTable(), episodeAt(1), params, sequence());
  assert.equal(result.episode.winner, "opponent");
  assert.equal(result.episode.done, true);
  assert.equal(result.episode.state, 0);
  assert.equal(result.calculation.opponentAction, null);
  assert.equal(result.calculation.reward, -1);
  assert.equal(result.calculation.maxNextQ, 0);
  close(result.calculation.newQ, 0.05);
  assert.equal(result.visits[1][0], 1);
});

test("an opponent taking the last stick gives the AI a +1 terminal reward", () => {
  const table = createQTable();
  table[3] = [0.4, 0, 0];
  const result = performTrainingStep(table, createVisitTable(), episodeAt(3), params, sequence(0.99));
  assert.equal(result.episode.winner, "ai");
  assert.equal(result.calculation.action, 1);
  assert.equal(result.calculation.opponentAction, 2);
  assert.equal(result.calculation.nextState, 0);
  assert.equal(result.calculation.reward, 1);
  assert.equal(result.calculation.maxNextQ, 0);
  close(result.calculation.newQ, 0.55);
});

test("Bellman update spans both turns and uses the next AI state's best legal Q", () => {
  const table = createQTable();
  table[7] = [0, 0, 0.5];
  table[2] = [0.8, 0.3, 999];
  table[4] = [0.99, 0.99, 0.99];
  const visits = createVisitTable();
  const episode = episodeAt(7);
  const oldTable = structuredClone(table);
  const oldVisits = structuredClone(visits);
  const result = performTrainingStep(table, visits, episode, params, sequence(0.4));
  assert.deepEqual(result.calculation, {
    state: 7, action: 3, opponentAction: 2, nextState: 2, terminal: false,
    reward: 0, oldQ: 0.5, maxNextQ: 0.8, newQ: 0.555,
    ...params, explored: false,
  });
  assert.equal(result.episode.state, 2);
  assert.equal(result.episode.winner, null);
  assert.equal(result.episode.done, false);
  assert.equal(result.table[7][2], 0.555);
  assert.equal(result.visits[7][2], 1);
  assert.deepEqual(table, oldTable);
  assert.deepEqual(visits, oldVisits);
  assert.equal(episode.state, 7);
  assert.notEqual(result.table, table);
  assert.notEqual(result.table[7], table[7]);
  assert.notEqual(result.visits[7], visits[7]);
});

test("calculation exposes exploration and exactly reproduces each updated value", () => {
  const table = createQTable();
  table[4] = [0.7, 0, 0];
  const result = performTrainingStep(
    table, createVisitTable(), episodeAt(4), { ...params, epsilon: 1 }, sequence(0.2, 0.5, 0),
  );
  const calc = result.calculation;
  assert.equal(calc.action, 2);
  assert.equal(calc.explored, true);
  assert.equal(calc.newQ, calc.oldQ + calc.alpha * (calc.reward + calc.gamma * calc.maxNextQ - calc.oldQ));
  assert.equal(result.table[calc.state][calc.action - 1], calc.newQ);
});

test("a completed episode is a safe no-op and invalid learning parameters are rejected", () => {
  const table = createQTable();
  const visits = createVisitTable();
  const episode = { ...episodeAt(0), done: true, winner: "ai" };
  assert.deepEqual(performTrainingStep(table, visits, episode, params, sequence()), {
    table, visits, episode, calculation: null,
  });
  for (const key of ["alpha", "gamma", "epsilon"]) {
    for (const value of [-0.1, 1.1, NaN]) {
      assert.throws(() => performTrainingStep(table, visits, episodeAt(12), { ...params, [key]: value }), RangeError);
    }
  }
});

test("seeded training against random replies learns all forced-winning moves and can continue", () => {
  const random = seededRandom(20261018);
  const learning = { alpha: 0.04, gamma: 0.95, epsilon: 0.3 };
  let table = createQTable();
  let visits = createVisitTable();
  let updates = 0;
  for (let game = 0; game < 30000; game += 1) {
    let episode = startTrainingEpisode(game % 2 === 0 ? "ai" : "opponent", random);
    while (!episode.done) {
      const result = performTrainingStep(table, visits, episode, learning, random);
      ({ table, visits, episode } = result);
      updates += 1;
      assert.ok(validActions(result.calculation.state).includes(result.calculation.action));
      assert.ok(episode.state >= 0);
    }
  }
  for (let state = 1; state <= 12; state += 1) {
    for (const action of validActions(state)) assert.ok(visits[state][action - 1] > 0);
    const optimal = optimalActions(state);
    if (optimal.length) assert.deepEqual(greedyActions(table, state), optimal, `state ${state}`);
  }
  assert.equal(visits.flat().reduce((sum, count) => sum + count, 0), updates);
  assert.ok(table[1][0] < -0.99);
  assert.ok(table[2][0] > 0.99);
  const result = performTrainingStep(table, visits, startTrainingEpisode("ai"), learning, random);
  assert.equal(result.visits.flat().reduce((sum, count) => sum + count, 0), updates + 1);
  assert.deepEqual(createQTable(), Array.from({ length: 13 }, () => [0, 0, 0]));
});
