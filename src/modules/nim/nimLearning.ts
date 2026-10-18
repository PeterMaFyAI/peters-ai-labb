/** The state is the number of sticks left when it is the AI's turn. */
export const INITIAL_STICKS = 12;

export type Action = 1 | 2 | 3;
export type QTable = number[][];
export type VisitTable = number[][];
export type LearningParameters = { alpha: number; gamma: number; epsilon: number };

export type TrainingEpisode = {
  state: number;
  done: boolean;
  winner: "ai" | "opponent" | null;
  starts: "ai" | "opponent";
  openingOpponentAction: Action | null;
};

export type TrainingCalculation = {
  state: number;
  action: Action;
  opponentAction: Action | null;
  nextState: number;
  terminal: boolean;
  reward: number;
  oldQ: number;
  maxNextQ: number;
  newQ: number;
  alpha: number;
  gamma: number;
  epsilon: number;
  explored: boolean;
};

export type TrainingStepResult = {
  table: QTable;
  visits: VisitTable;
  episode: TrainingEpisode;
  calculation: TrainingCalculation | null;
};

export function createQTable(): QTable {
  return Array.from({ length: INITIAL_STICKS + 1 }, () => [0, 0, 0]);
}

export function createVisitTable(): VisitTable {
  return createQTable();
}

function assertState(state: number): void {
  if (!Number.isInteger(state) || state < 0 || state > INITIAL_STICKS) {
    throw new RangeError(`State must be an integer between 0 and ${INITIAL_STICKS}.`);
  }
}

export function validActions(state: number): Action[] {
  assertState(state);
  return ([1, 2, 3] as Action[]).filter((action) => action <= state);
}

/** All equally good legal actions are returned, including ties in an untrained row. */
export function greedyActions(table: QTable, state: number): Action[] {
  const actions = validActions(state);
  const maximum = Math.max(...actions.map((action) => table[state][action - 1]));
  return actions.filter((action) => table[state][action - 1] === maximum);
}

function randomAction(actions: Action[], random: () => number): Action {
  if (actions.length === 0) throw new RangeError("There is no legal action in a terminal state.");
  if (actions.length === 1) return actions[0];
  return actions[Math.floor(random() * actions.length)];
}

function assertUnitInterval(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError(`${name} must be between 0 and 1.`);
  }
}

function selectAction(
  table: QTable,
  state: number,
  epsilon: number,
  random: () => number,
): { action: Action; explored: boolean } {
  assertUnitInterval(epsilon, "Epsilon");
  const explored = epsilon > 0 && random() < epsilon;
  const actions = explored ? validActions(state) : greedyActions(table, state);
  return { action: randomAction(actions, random), explored };
}

export function chooseAction(
  table: QTable,
  state: number,
  epsilon = 0,
  random: () => number = Math.random,
): Action {
  return selectAction(table, state, epsilon, random).action;
}

/** A winning move leaves 1, 5 or 9 sticks; those states have no forced-win move. */
export function optimalActions(state: number): Action[] {
  assertState(state);
  if (state === 0 || state % 4 === 1) return [];
  return [((state - 1) % 4) as Action];
}

export function startTrainingEpisode(
  first: "ai" | "opponent",
  random: () => number = Math.random,
): TrainingEpisode {
  const openingOpponentAction = first === "opponent"
    ? randomAction(validActions(INITIAL_STICKS), random)
    : null;
  return {
    state: INITIAL_STICKS - (openingOpponentAction ?? 0),
    done: false,
    winner: null,
    starts: first,
    openingOpponentAction,
  };
}

/**
 * A transition includes the AI's action and the random opponent's reply. This
 * makes the next nonterminal state another AI decision, so the Bellman target
 * can use + gamma * max Q(nextState, action) without changing player perspective.
 */
export function performTrainingStep(
  table: QTable,
  visits: VisitTable,
  episode: TrainingEpisode,
  params: LearningParameters,
  random: () => number = Math.random,
): TrainingStepResult {
  if (episode.done) return { table, visits, episode, calculation: null };
  assertUnitInterval(params.alpha, "Alpha");
  assertUnitInterval(params.gamma, "Gamma");

  const state = episode.state;
  const { action, explored } = selectAction(table, state, params.epsilon, random);
  let nextState = state - action;
  let opponentAction: Action | null = null;
  let winner: TrainingEpisode["winner"] = null;
  let reward = 0;

  if (nextState === 0) {
    winner = "opponent";
    reward = -1;
  } else {
    opponentAction = randomAction(validActions(nextState), random);
    nextState -= opponentAction;
    if (nextState === 0) {
      winner = "ai";
      reward = 1;
    }
  }

  const terminal = winner !== null;
  const oldQ = table[state][action - 1];
  const maxNextQ = terminal
    ? 0
    : Math.max(...validActions(nextState).map((nextAction) => table[nextState][nextAction - 1]));
  const newQ = oldQ + params.alpha * (reward + params.gamma * maxNextQ - oldQ);

  const nextTable = table.map((row, index) => index === state ? [...row] : row);
  nextTable[state][action - 1] = newQ;
  const nextVisits = visits.map((row, index) => index === state ? [...row] : row);
  nextVisits[state][action - 1] += 1;

  return {
    table: nextTable,
    visits: nextVisits,
    episode: { ...episode, state: nextState, done: terminal, winner },
    calculation: {
      state,
      action,
      opponentAction,
      nextState,
      terminal,
      reward,
      oldQ,
      maxNextQ,
      newQ,
      ...params,
      explored,
    },
  };
}
