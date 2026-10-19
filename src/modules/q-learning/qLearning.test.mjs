import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_GOAL,
  DEFAULT_ITEMS,
  DEFAULT_START,
  DEFAULT_WALKABLE,
  ALL_ACTIONS,
  Q_ENVIRONMENTS,
  buildLabelMap,
  createAppleIndexByKey,
  createZeroQTable,
  deriveAppleKeys,
  encodeStateKey,
  getValidActions,
  normalizeAppleMask,
  nextEpsilon,
  performStep,
  pickAction,
  previewStep,
  roundQValue,
  shouldExpandAppleStates,
  traceGreedyPath
} from "./qLearning.ts";

function setup(environment = Q_ENVIRONMENTS[0]) {
  const { walkableKeys, startKey, goalKey, cellItems, enabledActions } = environment;
  const walkableSet = new Set(walkableKeys);
  const labelMap = buildLabelMap(walkableKeys, startKey, goalKey);
  const appleKeys = deriveAppleKeys(cellItems, labelMap);
  const appleIndexByKey = createAppleIndexByKey(appleKeys);
  const expandedStateSpace = shouldExpandAppleStates({
    ...environment,
    walkableSet,
    appleKeys
  });
  const positions = walkableKeys.filter((key) => key !== goalKey && cellItems[key] !== "bird");
  const stateKeys = [...new Set(positions.flatMap((key) =>
    Array.from({ length: expandedStateSpace ? 1 << appleKeys.length : 1 }, (_, mask) =>
      encodeStateKey(key, normalizeAppleMask(mask, key, appleIndexByKey, appleKeys.length), expandedStateSpace)
    )
  ))];
  return {
    startKey,
    qTable: createZeroQTable(stateKeys),
    agentKey: startKey,
    agentAppleMask: 0,
    episodeScore: 0,
    stepCount: 0,
    walkableSet,
    goalKey,
    cellItems,
    labelMap,
    appleKeys,
    appleIndexByKey,
    expandedStateSpace,
    validActionsByState: Object.fromEntries(walkableKeys.map((key) =>
      [key, getValidActions(key, walkableSet, enabledActions)]
    )),
    alpha: 0.5,
    gamma: 0.9,
    epsilon: 0.9,
    stepCost: environment.stepCost
  };
}

function continueFrom(params, outcome) {
  return {
    ...params,
    qTable: outcome.qTable,
    agentKey: outcome.agentKey,
    agentAppleMask: outcome.agentAppleMask,
    episodeScore: outcome.episodeScore,
    stepCount: outcome.stepCount
  };
}

test("the first five presets keep their small focused tables and zero step cost", () => {
  assert.equal(Q_ENVIRONMENTS.length, 6);
  const expected = [
    [["0,0", "1,0", "2,0"], "1,0", "2,0", { "0,0": "bird" }, ["left", "right"]],
    [["0,0", "1,0", "2,0", "3,0"], "0,0", "3,0", {}, ["right"]],
    [["0,0", "1,0", "0,1", "1,1"], "0,0", "1,1", { "0,1": "apple" }, ["right", "down"]],
    [["0,0", "1,0", "2,0", "1,1", "2,1"], "0,0", "2,1", { "1,1": "bird" }, ["right", "down"]],
    [["0,0", "1,0", "2,0", "0,1", "0,2"], "0,0", "2,0", { "0,1": "apple", "0,2": "bird" }, ["right", "down"]]
  ];
  Q_ENVIRONMENTS.slice(0, 5).forEach((environment, index) => {
    assert.deepEqual(
      [environment.walkableKeys, environment.startKey, environment.goalKey, environment.cellItems, environment.enabledActions],
      expected[index]
    );
    assert.ok(environment.description.length > 0);
    assert.equal(setup(environment).expandedStateSpace, false);
    assert.equal(environment.stepCost, 0);
  });
  assert.deepEqual(Q_ENVIRONMENTS.slice(0, 5).map((environment) => Object.keys(setup(environment).qTable).length), [1, 3, 3, 3, 3]);
  assert.deepEqual([DEFAULT_WALKABLE, DEFAULT_START, DEFAULT_GOAL, DEFAULT_ITEMS], expected[0].slice(0, 4));
});

test("the sixth preset is a full 4 by 4 automatic environment with apple memory", () => {
  const environment = Q_ENVIRONMENTS[5];
  const params = setup(environment);
  assert.equal(environment.walkableKeys.length, 16);
  assert.deepEqual([...params.walkableSet].sort(), Array.from({ length: 16 }, (_, index) =>
    `${index % 4},${Math.floor(index / 4)}`).sort());
  assert.equal(environment.startKey, "0,0");
  assert.equal(environment.goalKey, "3,3");
  assert.deepEqual(environment.enabledActions, ALL_ACTIONS);
  assert.equal(Object.values(environment.cellItems).filter(item => item === "bird").length, 2);
  assert.equal(params.appleKeys.length, 1);
  assert.equal(environment.stepCost, -0.1);
  assert.equal(environment.automaticOnly, true);
  assert.equal(params.expandedStateSpace, true);
  assert.equal(Object.keys(params.qTable).length, 25);
});

test("step cost applies to neutral moves, apples, revisits, birds and the goal", () => {
  const params = setup(Q_ENVIRONMENTS[5]);
  const neutral = performStep({ ...params, action: "right" });
  assert.equal(neutral.reward, -0.1);
  assert.equal(neutral.calculation.newQ, -0.05);
  const bird = performStep({ ...continueFrom(params, neutral), action: "down" });
  assert.equal(bird.reward, -2.1);
  assert.equal(bird.episodeScore, -2.2);
  assert.equal(bird.done, true);

  let current = params;
  const rewards = [];
  for (const action of ["right", "right", "down", "up", "down", "right", "down", "down"]) {
    const preview = previewStep({ ...current, action });
    const outcome = performStep({ ...current, action });
    assert.equal(outcome.reward, preview.reward);
    rewards.push(outcome.reward);
    current = continueFrom(params, outcome);
  }
  assert.deepEqual(rewards, [-0.1, -0.1, 0.9, -0.1, -0.1, -0.1, -0.1, 1.9]);
  assert.equal(current.agentKey, params.goalKey);
  assert.equal(current.episodeScore, 2.2);
});

test("epsilon schedules decay per episode, respect their floor and never increase epsilon", () => {
  assert.equal(nextEpsilon(0.9, "constant", 0.5, 1), 0.9);
  assert.equal(nextEpsilon(0.8, "multiply", 0.5, 0.1), 0.4);
  assert.equal(nextEpsilon(0.8, "subtract", 0.2, 0.1), 0.6000000000000001);
  assert.equal(nextEpsilon(0.11, "subtract", 0.2, 0.1), 0.1);
  assert.equal(nextEpsilon(0.11, "multiply", 0.5, 0.1), 0.1);
  assert.equal(nextEpsilon(0.02, "multiply", 0.5, 0.1), 0.02);
  assert.equal(nextEpsilon(0.9, "multiply", 1, 0), 0.9);
  let epsilon = 0.9;
  for (let index = 0; index < 200; index += 1) epsilon = nextEpsilon(epsilon, "multiply", 0.98, 0.05);
  assert.equal(epsilon, 0.05);
});

test("the displayed path follows current legal Q maxima without randomness or learning", (t) => {
  t.mock.method(Math, "random", () => assert.fail("Path display must never sample randomness"));
  const params = setup();
  params.qTable[params.startKey] = { left: 1, right: 2, up: 999, down: 999 };
  const original = structuredClone(params);
  const goalPath = traceGreedyPath(params);
  assert.equal(goalPath.ending, "goal");
  assert.equal(goalPath.score, 2);
  assert.deepEqual(goalPath.steps.map(step => step.action), ["right"]);
  assert.deepEqual(params, original);

  // The displayed policy must change even when the new Q maximum leads to a worse outcome.
  params.qTable[params.startKey].left = 3;
  const birdPath = traceGreedyPath(params);
  assert.equal(birdPath.ending, "bird");
  assert.equal(birdPath.score, -2);
  assert.deepEqual(birdPath.steps.map(step => step.action), ["left"]);
});

test("Q-value ties follow column order and produce the same path on every render", () => {
  const params = setup();
  const path = traceGreedyPath(params);
  assert.equal(path.hasTies, true);
  assert.equal(path.steps[0].action, "left");
  assert.equal(path.ending, "bird");
  assert.deepEqual(traceGreedyPath(params), path);
});

test("greedy paths track apple memory, allow meaningful revisits and stop on repeated states", () => {
  const params = setup({ ...Q_ENVIRONMENTS[2], enabledActions: ALL_ACTIONS });
  params.qTable["0,0|0"].down = 5;
  params.qTable["0,1|1"].up = 5;
  params.qTable["0,0|1"].right = 5;
  params.qTable["1,0|1"].down = 5;
  const path = traceGreedyPath(params);
  assert.equal(path.ending, "goal");
  assert.equal(path.score, 3);
  assert.deepEqual(path.steps.map(step => step.stateKey), ["0,0|0", "0,1|1", "0,0|1", "1,0|1"]);

  params.qTable["0,0|1"].down = 6;
  const loop = traceGreedyPath(params);
  assert.equal(loop.ending, "loop");
  assert.equal(loop.steps.length, 3);
  assert.equal(loop.score, 1);
});

test("a policy without a legal continuation reports a blocked path", () => {
  const params = setup(Q_ENVIRONMENTS[1]);
  params.validActionsByState[params.startKey] = [];
  assert.deepEqual(traceGreedyPath(params), { steps: [], score: 0, ending: "blocked", hasTies: false });
  params.validActionsByState[params.startKey] = ["up"];
  assert.equal(traceGreedyPath(params).steps.length, 0);
});

test("the displayed six-step route identifies exact table cells before and after the apple", () => {
  const params = setup(Q_ENVIRONMENTS[5]);
  let agentKey = params.startKey;
  let agentAppleMask = 0;
  for (const action of ["right", "right", "down", "right", "down", "down"]) {
    params.qTable[encodeStateKey(agentKey, agentAppleMask, true)][action] = 10;
    const preview = previewStep({ ...params, agentKey, agentAppleMask, action });
    agentKey = preview.nextKey;
    agentAppleMask = preview.nextMask;
  }
  const path = traceGreedyPath(params);
  assert.equal(path.ending, "goal");
  assert.equal(path.score, 2.4);
  assert.equal(path.steps.length, 6);
  assert.deepEqual(path.steps[2], { stateKey: "2,0|0", positionKey: "2,0", nextPositionKey: "2,1", action: "down" });
  assert.deepEqual(path.steps[3], { stateKey: "2,1|1", positionKey: "2,1", nextPositionKey: "3,1", action: "right" });
});

test("seeded automatic training with epsilon decay learns a safe six-step apple route", (t) => {
  let seed = 37;
  t.mock.method(Math, "random", () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  });
  const params = setup(Q_ENVIRONMENTS[5]);
  let qTable = params.qTable;
  let epsilon = 0.9;
  for (let episode = 0; episode < 1000; episode += 1) {
    let current = { ...params, qTable, epsilon };
    for (let step = 0; step < 220; step += 1) {
      const outcome = performStep(current);
      assert.ok(outcome);
      current = { ...continueFrom(params, outcome), epsilon };
      if (outcome.done) break;
    }
    qTable = current.qTable;
    epsilon = nextEpsilon(epsilon, "multiply", 0.98, 0.05);
  }
  const path = traceGreedyPath({ ...params, qTable });
  assert.equal(path.ending, "goal");
  assert.equal(path.steps.at(-1).nextPositionKey, params.goalKey);
  assert.ok(path.steps.at(-1).stateKey.endsWith("|1"));
  assert.equal(path.steps.length, 6);
  assert.equal(path.score, 2.4);
});

test("terminal goal and bird rewards ignore continuation values and leave the original table intact", () => {
  for (const [action, reward] of [["right", 2], ["left", -2]]) {
    const params = setup();
    const nextKey = action === "right" ? "2,0" : "0,0";
    params.qTable[nextKey] = { right: 999, down: 999, left: 999, up: 999 };
    const original = structuredClone(params.qTable);
    const outcome = performStep({ ...params, action });
    assert.equal(outcome.reward, reward);
    assert.equal(outcome.done, true);
    assert.equal(outcome.agentKey, nextKey);
    assert.equal(outcome.episodeScore, reward);
    assert.equal(outcome.stepCount, 1);
    assert.deepEqual(outcome.calculation, {
      stateKey: "1,0",
      nextStateKey: nextKey,
      stateLabel: "A",
      nextStateLabel: action === "right" ? "S" : "B",
      terminal: true,
      action,
      reward,
      oldQ: 0,
      maxNextQ: 0,
      target: reward,
      rawNewQ: reward * 0.5,
      newQ: reward * 0.5,
      alpha: 0.5,
      gamma: 0.9
    });
    assert.equal(outcome.qTable["1,0"][action], reward * 0.5);
    assert.deepEqual(params.qTable, original);
    assert.notEqual(outcome.qTable["1,0"], params.qTable["1,0"]);
  }
});

test("a forced action never draws randomness, rejects illegal actions and updates only that cell", (t) => {
  t.mock.method(Math, "random", () => assert.fail("Locked actions must never be resampled"));
  const params = setup();
  params.qTable["1,0"].left = 8;
  Object.freeze(params.qTable["1,0"]);
  Object.freeze(params.qTable);
  const outcome = performStep({ ...params, action: "right", epsilon: 1 });
  assert.equal(outcome.action, "right");
  assert.equal(outcome.qTable["1,0"].right, 1);
  assert.equal(outcome.qTable["1,0"].left, 8);
  assert.equal(params.qTable["1,0"].right, 0);
  assert.equal(performStep({ ...params, action: "down" }), null);
  assert.equal(performStep({
    ...params,
    action: "down",
    validActionsByState: { "1,0": ["down"] }
  }), null);
});

test("epsilon-greedy chooses among legal moves including tied greedy moves", (t) => {
  const randomDraws = [0.1, 0.99, 0.9, 0, 0.9, 0.99];
  t.mock.method(Math, "random", () => {
    assert.ok(randomDraws.length > 0);
    return randomDraws.shift();
  });
  const row = { right: 0.5, down: 0.5, left: 999, up: 999 };
  assert.equal(pickAction(row, 1, ["right", "down"]), "down");
  assert.equal(pickAction(row, 0, ["right", "down"]), "right");
  assert.equal(pickAction(row, 0, ["right", "down"]), "down");
  assert.equal(pickAction(row, 1, []), null);
  assert.equal(randomDraws.length, 0);
});

test("the maximum uses only valid next actions and every update is stored rounded", () => {
  const params = setup(Q_ENVIRONMENTS[3]);
  params.qTable["1,0"] = { right: 0.46, down: -0.5, left: 999, up: 999 };
  const outcome = performStep({ ...params, action: "right" });
  assert.equal(outcome.done, false);
  assert.equal(outcome.calculation.maxNextQ, 0.46);
  assert.ok(Math.abs(outcome.calculation.rawNewQ - 0.207) < 1e-12);
  assert.equal(outcome.calculation.newQ, 0.21);
  assert.equal(outcome.qTable["0,0"].right, 0.21);
  const unchanged = performStep({ ...params, action: "right", alpha: 0 });
  assert.equal(unchanged.calculation.oldQ, 0);
  assert.equal(unchanged.calculation.newQ, 0);
  assert.equal(unchanged.calculation.stateKey, "0,0");
});

test("the fourth preset forces entry to a choice and propagates the better learned continuation", () => {
  const params = setup(Q_ENVIRONMENTS[3]);
  assert.deepEqual(params.validActionsByState["0,0"], ["right"]);
  assert.deepEqual(params.validActionsByState["1,0"], ["right", "down"]);
  assert.equal(performStep({ ...params, action: "down" }), null);

  const enterChoice = performStep({ ...params, action: "right" });
  const bird = performStep({ ...continueFrom(params, enterChoice), action: "down" });
  assert.equal(bird.reward, -2);
  assert.equal(bird.done, true);
  assert.equal(bird.qTable["1,0"].down, -1);

  const runGoalPath = (qTable) => {
    let current = { ...params, qTable };
    for (const action of ["right", "right", "down"]) {
      const outcome = performStep({ ...current, action });
      current = continueFrom(params, outcome);
    }
    return current.qTable;
  };
  let learned = runGoalPath(bird.qTable);
  learned = runGoalPath(learned);
  assert.equal(learned["1,0"].right, 0.45);
  assert.equal(learned["1,0"].down, -1);
  const nextEntry = performStep({ ...params, qTable: learned, action: "right" });
  assert.equal(nextEntry.calculation.maxNextQ, 0.45);
  assert.equal(nextEntry.qTable["0,0"].right, 0.2);
});

test("the fifth preset contrasts a tempting apple trap with a safe route to the goal", () => {
  const params = setup(Q_ENVIRONMENTS[4]);
  assert.deepEqual(params.validActionsByState["0,0"], ["right", "down"]);
  assert.deepEqual(params.validActionsByState["0,1"], ["down"]);
  assert.deepEqual(params.validActionsByState["1,0"], ["right"]);

  const apple = performStep({ ...params, action: "down" });
  assert.equal(apple.reward, 1);
  assert.equal(apple.done, false);
  assert.equal(apple.agentKey, "0,1");
  assert.equal(apple.qTable["0,0"].down, 0.5);
  assert.equal(performStep({ ...continueFrom(params, apple), action: "right" }), null);
  const bird = performStep({ ...continueFrom(params, apple), action: "down" });
  assert.equal(bird.reward, -2);
  assert.equal(bird.done, true);
  assert.equal(bird.agentKey, "0,2");
  assert.equal(bird.episodeScore, -1);

  const neutral = performStep({ ...params, action: "right" });
  assert.equal(neutral.reward, 0);
  assert.equal(neutral.done, false);
  const goal = performStep({ ...continueFrom(params, neutral), action: "right" });
  assert.equal(goal.reward, 2);
  assert.equal(goal.done, true);
  assert.equal(goal.agentKey, "2,0");
  assert.equal(goal.episodeScore, 2);
});

test("learning the apple trap reverses its initially positive value and gamma changes the best route", () => {
  const trainBothRoutes = (gamma) => {
    const params = { ...setup(Q_ENVIRONMENTS[4]), gamma };
    let qTable = params.qTable;
    const appleValues = [];
    let appleTarget;
    for (let episode = 0; episode < 40; episode += 1) {
      for (const actions of [["down", "down"], ["right", "right"]]) {
        let current = { ...params, qTable };
        actions.forEach((action) => {
          const outcome = performStep({ ...current, action });
          if (current.agentKey === "0,0" && action === "down") {
            appleTarget = outcome.calculation.target;
          }
          current = continueFrom(params, outcome);
        });
        qTable = current.qTable;
      }
      appleValues.push(qTable["0,0"].down);
    }
    return { qTable, appleValues, appleTarget };
  };

  const farsighted = trainBothRoutes(0.9);
  assert.equal(farsighted.appleValues[0], 0.5);
  assert.ok(farsighted.appleValues.some((value) => value < 0));
  assert.equal(farsighted.qTable["0,0"].down, -0.8);
  assert.equal(farsighted.qTable["0,0"].right, 1.8);
  assert.equal(pickAction(farsighted.qTable["0,0"], 0, ["right", "down"]), "right");

  const shortsighted = trainBothRoutes(0.1);
  assert.equal(shortsighted.appleTarget, 0.8);
  // Repeated rounding can retain one hundredth above a target approached from above.
  assert.equal(shortsighted.qTable["0,0"].down, 0.81);
  assert.equal(shortsighted.qTable["0,0"].right, 0.2);
  assert.equal(pickAction(shortsighted.qTable["0,0"], 0, ["right", "down"]), "down");
});

test("future value propagates backwards across episodes using the stored rounded table", () => {
  const params = setup(Q_ENVIRONMENTS[1]);
  const runEpisode = (qTable) => {
    let current = { ...params, qTable };
    const steps = [];
    for (let index = 0; index < 3; index += 1) {
      assert.deepEqual(current.validActionsByState[current.agentKey], ["right"]);
      assert.equal(performStep({ ...current, action: "left" }), null);
      const outcome = performStep({ ...current, action: "right" });
      assert.equal(outcome.reward, index === 2 ? 2 : 0);
      assert.equal(outcome.done, index === 2);
      steps.push(outcome);
      current = continueFrom(params, outcome);
    }
    assert.equal(current.agentKey, "3,0");
    assert.equal(current.episodeScore, 2);
    return steps;
  };
  const rightValues = (qTable) => ["0,0", "1,0", "2,0"].map((key) => qTable[key].right);

  const firstEpisode = runEpisode(params.qTable);
  assert.deepEqual(rightValues(firstEpisode[2].qTable), [0, 0, 1]);
  const secondEpisode = runEpisode(firstEpisode[2].qTable);
  assert.equal(secondEpisode[0].calculation.maxNextQ, 0);
  assert.equal(secondEpisode[1].calculation.maxNextQ, 1);
  assert.deepEqual(rightValues(secondEpisode[2].qTable), [0, 0.45, 1.5]);
  const thirdEpisode = runEpisode(secondEpisode[2].qTable);
  assert.equal(thirdEpisode[0].calculation.maxNextQ, 0.45);
  assert.ok(Math.abs(thirdEpisode[0].calculation.rawNewQ - 0.2025) < 1e-12);
  assert.deepEqual(rightValues(thirdEpisode[2].qTable), [0.2, 0.9, 1.75]);
  const fourthEpisode = runEpisode(thirdEpisode[2].qTable);
  assert.equal(fourthEpisode[0].calculation.oldQ, 0.2);
  assert.deepEqual(rightValues(fourthEpisode[2].qTable), [0.51, 1.24, 1.88]);
});

test("the corner preset offers a nonterminal apple followed by a terminal goal without expanded states", () => {
  const params = setup(Q_ENVIRONMENTS[2]);
  assert.equal(params.expandedStateSpace, false);
  assert.deepEqual(Object.keys(params.qTable), ["0,0", "1,0", "0,1"]);
  const appleStep = performStep({ ...params, action: "down" });
  assert.equal(appleStep.agentKey, "0,1");
  assert.equal(appleStep.agentAppleMask, 1);
  assert.equal(appleStep.reward, 1);
  assert.equal(appleStep.done, false);
  assert.equal(appleStep.calculation.nextStateKey, "0,1");
  assert.equal(appleStep.qTable["0,0"].down, 0.5);
  const goalStep = performStep({ ...continueFrom(params, appleStep), action: "right" });
  assert.equal(goalStep.agentKey, "1,1");
  assert.equal(goalStep.reward, 2);
  assert.equal(goalStep.done, true);
  assert.equal(goalStep.episodeScore, 3);
  assert.equal(goalStep.calculation.maxNextQ, 0);
  assert.equal(goalStep.qTable["0,1"].right, 1);
});

test("school rounding is symmetric at decimal ties and never stores negative zero", () => {
  for (const [value, expected] of [[0.005, 0.01], [1.005, 1.01], [1.255, 1.26], [2.675, 2.68], [0.0049, 0], [0.207, 0.21], [1.234, 1.23]]) {
    assert.equal(roundQValue(value), expected);
    assert.equal(roundQValue(-value), expected === 0 ? 0 : -expected);
  }
  assert.equal(Object.is(roundQValue(-0), -0), false);
  assert.equal(Object.is(roundQValue(-0.001), -0), false);
  for (const [action, expected] of [["right", 0.01], ["left", -0.01]]) {
    const outcome = performStep({ ...setup(), action, alpha: 0.0025 });
    assert.equal(outcome.calculation.newQ, expected);
    assert.equal(outcome.qTable["1,0"][action], expected);
  }
});

test("an apple preview does not mutate state and agrees with the subsequent update", () => {
  const params = setup(Q_ENVIRONMENTS[2]);
  params.qTable["0,1"].right = 0.8;
  const original = structuredClone(params);
  const preview = previewStep({ ...params, action: "down" });
  assert.deepEqual(preview, {
    nextKey: "0,1",
    nextMask: 1,
    nextStateKey: "0,1",
    nextStateLabel: "C",
    reward: 1,
    terminal: false
  });
  assert.deepEqual(params, original);
  const outcome = performStep({ ...params, action: "down" });
  assert.equal(outcome.agentKey, preview.nextKey);
  assert.equal(outcome.agentAppleMask, preview.nextMask);
  assert.equal(outcome.calculation.nextStateKey, preview.nextStateKey);
  assert.equal(outcome.calculation.nextStateLabel, preview.nextStateLabel);
  assert.equal(outcome.calculation.reward, preview.reward);
  assert.equal(outcome.calculation.maxNextQ, 0.8);
  assert.equal(outcome.calculation.newQ, 0.86);
  assert.equal(outcome.done, false);
});

test("revisitable apples expand the state and pay only once per episode", () => {
  const environment = { ...Q_ENVIRONMENTS[2], enabledActions: ["right", "down", "left", "up"] };
  const params = setup(environment);
  assert.equal(params.expandedStateSpace, true);
  assert.ok(params.qTable["0,0|0"]);
  assert.ok(params.qTable["0,0|1"]);
  assert.ok(params.qTable["0,1|1"]);
  assert.equal(params.qTable["0,1|0"], undefined);
  const firstApple = performStep({ ...params, action: "down" });
  assert.equal(firstApple.reward, 1);
  assert.equal(firstApple.calculation.stateKey, "0,0|0");
  assert.equal(firstApple.calculation.nextStateKey, "0,1|1");
  const backToStart = performStep({ ...continueFrom(params, firstApple), action: "up" });
  assert.equal(backToStart.reward, 0);
  assert.equal(backToStart.calculation.nextStateKey, "0,0|1");
  const secondApple = performStep({ ...continueFrom(params, backToStart), action: "down" });
  assert.equal(secondApple.reward, 0);
  assert.equal(secondApple.calculation.stateKey, "0,0|1");
  assert.equal(secondApple.calculation.nextStateKey, "0,1|1");
  assert.equal(secondApple.qTable["0,0|0"].down, 0.5);
  assert.equal(secondApple.qTable["0,0|1"].down, 0);
  assert.equal(previewStep({ ...params, qTable: secondApple.qTable, action: "down" }).reward, 1);
});
