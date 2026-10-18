export type Action = "right" | "down" | "left" | "up";
export type TileItem = "none" | "apple" | "bird";
export type QRow = Record<Action, number>;
export type QTable = Record<string, QRow>;

export interface QStateDefinition {
  key: string;
  positionKey: string;
  appleMask: number;
  label: string;
}

export interface QCalculation {
  stateKey: string;
  nextStateKey: string;
  stateLabel: string;
  nextStateLabel: string;
  terminal: boolean;
  action: Action;
  reward: number;
  oldQ: number;
  maxNextQ: number;
  target: number;
  rawNewQ: number;
  newQ: number;
  alpha: number;
  gamma: number;
}

export interface StepOutcome {
  qTable: QTable;
  agentKey: string;
  agentAppleMask: number;
  reward: number;
  done: boolean;
  episodeScore: number;
  stepCount: number;
  calculation: QCalculation;
  action: Action;
}

export interface Coordinate {
  x: number;
  y: number;
}

export const ALL_ACTIONS: Action[] = ["right", "down", "left", "up"];
export const ACTION_LABELS: Record<Action, string> = {
  right: "Höger",
  down: "Ner",
  left: "Vänster",
  up: "Upp"
};
export const ACTION_SYMBOLS: Record<Action, string> = {
  right: "H",
  down: "N",
  left: "V",
  up: "U"
};
export const ACTION_DELTAS: Record<Action, Coordinate> = {
  right: { x: 1, y: 0 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
  up: { x: 0, y: -1 }
};

export const EDITOR_SIDE = 4;
export const MAX_WALKABLE_CELLS = 16;
export const MAX_APPLES = 2;
export const MAX_STEPS_PER_EPISODE = 220;
export const OTHER_STATE_LABELS = [
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O"
];
export interface QEnvironment {
  id: string;
  name: string;
  description: string;
  walkableKeys: string[];
  startKey: string;
  goalKey: string;
  cellItems: Record<string, TileItem>;
  enabledActions: Action[];
}

export const Q_ENVIRONMENTS: QEnvironment[] = [
  {
    id: "direct-reward",
    name: "1. Direkt belöning",
    description: "Träna på positiv och negativ belöning med en enda rad i Q-tabellen.",
    walkableKeys: ["0,0", "1,0", "2,0"],
    startKey: "1,0",
    goalKey: "2,0",
    cellItems: { "0,0": "bird" },
    enabledActions: ["left", "right"]
  },
  {
    id: "one-step-further",
    name: "2. Ett steg längre",
    description: "Träna på framtidsvärde och se hur värden sprids bakåt mellan episoder.",
    walkableKeys: ["0,0", "1,0", "2,0", "3,0"],
    startKey: "0,0",
    goalKey: "3,0",
    cellItems: {},
    enabledActions: ["right"]
  },
  {
    id: "around-the-corner",
    name: "3. Runt hörnet",
    description: "Koppla tillstånd till rätt rad och riktning till rätt kolumn. Prova vägen via äpplet.",
    walkableKeys: ["0,0", "1,0", "0,1", "1,1"],
    startKey: "0,0",
    goalKey: "1,1",
    cellItems: { "0,1": "apple" },
    enabledActions: ["right", "down"]
  },
  {
    id: "several-continuations",
    name: "4. Vilket framtidsvärde?",
    description: "Öva på att jämföra flera möjliga framtidsvärden. Vilket Q-värde från nästa tillstånd ska du använda?",
    walkableKeys: ["0,0", "1,0", "2,0", "1,1", "2,1"],
    startKey: "0,0",
    goalKey: "2,1",
    cellItems: { "1,1": "bird" },
    enabledActions: ["right", "down"]
  },
  {
    id: "apple-on-the-way",
    name: "5. Det lockande äpplet",
    description: "Jämför äpplets snabba belöning med vägen till målet. Prova gamma 0,1 och 0,9 i varsin träning från noll och följ hur det bästa vägvalet ändras.",
    walkableKeys: ["0,0", "1,0", "2,0", "0,1", "0,2"],
    startKey: "0,0",
    goalKey: "2,0",
    cellItems: { "0,1": "apple", "0,2": "bird" },
    enabledActions: ["right", "down"]
  }
];

export const DEFAULT_START = Q_ENVIRONMENTS[0].startKey;
export const DEFAULT_GOAL = Q_ENVIRONMENTS[0].goalKey;
export const DEFAULT_WALKABLE = Q_ENVIRONMENTS[0].walkableKeys;
export const DEFAULT_ITEMS = Q_ENVIRONMENTS[0].cellItems;

export function keyFromXY(x: number, y: number): string {
  return `${x},${y}`;
}

export function parseKey(key: string): Coordinate {
  const [xRaw, yRaw] = key.split(",");
  return {
    x: Number.parseInt(xRaw, 10),
    y: Number.parseInt(yRaw, 10)
  };
}

export function sortKeys(keys: string[]): string[] {
  return [...keys].sort((left, right) => {
    const leftCoord = parseKey(left);
    const rightCoord = parseKey(right);
    if (leftCoord.y === rightCoord.y) return leftCoord.x - rightCoord.x;
    return leftCoord.y - rightCoord.y;
  });
}

export function makeZeroRow(): QRow {
  return {
    right: 0,
    down: 0,
    left: 0,
    up: 0
  };
}

export function cloneQTable(qTable: QTable): QTable {
  const next: QTable = {};
  Object.entries(qTable).forEach(([stateKey, row]) => {
    next[stateKey] = { ...row };
  });
  return next;
}

export function createZeroQTable(stateKeys: string[]): QTable {
  const table: QTable = {};
  stateKeys.forEach((key) => {
    table[key] = makeZeroRow();
  });
  return table;
}

export function alignQTable(previous: QTable, stateKeys: string[]): QTable {
  const next: QTable = {};
  stateKeys.forEach((key) => {
    const existing = previous[key];
    next[key] = existing
      ? {
          right: existing.right ?? 0,
          down: existing.down ?? 0,
          left: existing.left ?? 0,
          up: existing.up ?? 0
        }
      : makeZeroRow();
  });
  return next;
}

export function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function roundQValue(value: number): number {
  const scaledMagnitude = Math.abs(value) * 100;
  // Compensate only for floating point representation at decimal half ties.
  const roundedMagnitude = Math.round(
    scaledMagnitude + Number.EPSILON * Math.max(1, scaledMagnitude)
  ) / 100;
  if (roundedMagnitude === 0) return 0;
  return value < 0 ? -roundedMagnitude : roundedMagnitude;
}

export function formatSigned(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

export function nextStateFromAction(stateKey: string, action: Action): string {
  const currentCoord = parseKey(stateKey);
  const delta = ACTION_DELTAS[action];
  return keyFromXY(currentCoord.x + delta.x, currentCoord.y + delta.y);
}

export function getValidActions(
  stateKey: string,
  walkableSet: Set<string>,
  enabledActions: Action[]
): Action[] {
  return enabledActions.filter((action) =>
    walkableSet.has(nextStateFromAction(stateKey, action))
  );
}

export function maxQ(row: QRow | undefined, validActions: Action[]): number {
  if (!row || validActions.length === 0) return 0;
  return Math.max(...validActions.map((action) => row[action]));
}

export function pickAction(row: QRow, epsilon: number, validActions: Action[]): Action | null {
  if (validActions.length === 0) return null;

  if (Math.random() < epsilon) {
    return validActions[Math.floor(Math.random() * validActions.length)];
  }

  const bestValue = maxQ(row, validActions);
  const bestActions = validActions.filter((action) => row[action] === bestValue);
  return bestActions[Math.floor(Math.random() * bestActions.length)];
}

export function buildLabelMap(
  walkableKeys: string[],
  startKey: string,
  goalKey: string
): Record<string, string> {
  const map: Record<string, string> = {};
  map[startKey] = "A";
  map[goalKey] = "S";

  const otherStates = sortKeys(
    walkableKeys.filter((key) => key !== startKey && key !== goalKey)
  );

  otherStates.forEach((key, index) => {
    map[key] = OTHER_STATE_LABELS[index] ?? `X${index + 1}`;
  });

  return map;
}

export function normalizeItems(
  items: Record<string, TileItem>,
  walkableSet: Set<string>,
  startKey: string,
  goalKey: string,
  labelMap: Record<string, string>
): Record<string, TileItem> {
  const next: Record<string, TileItem> = {};
  const appleKeys: string[] = [];

  Object.entries(items).forEach(([key, item]) => {
    if (!walkableSet.has(key)) return;
    if (key === startKey || key === goalKey) return;
    if (item === "bird") {
      next[key] = item;
      return;
    }
    if (item === "apple") {
      appleKeys.push(key);
    }
  });

  const sortedAppleKeys = [...appleKeys]
    .sort((left, right) => {
      const leftLabel = labelMap[left] ?? left;
      const rightLabel = labelMap[right] ?? right;
      return leftLabel.localeCompare(rightLabel);
    })
    .slice(0, MAX_APPLES);

  sortedAppleKeys.forEach((key) => {
    next[key] = "apple";
  });

  return next;
}

export function deriveAppleKeys(
  cellItems: Record<string, TileItem>,
  labelMap: Record<string, string>
): string[] {
  return Object.entries(cellItems)
    .filter(([, item]) => item === "apple")
    .map(([key]) => key)
    .sort((left, right) => {
      const leftLabel = labelMap[left] ?? left;
      const rightLabel = labelMap[right] ?? right;
      return leftLabel.localeCompare(rightLabel);
    })
    .slice(0, MAX_APPLES);
}

export function createAppleIndexByKey(appleKeys: string[]): Record<string, number> {
  const map: Record<string, number> = {};
  appleKeys.forEach((key, index) => {
    map[key] = index;
  });
  return map;
}

export function encodeStateKey(positionKey: string, appleMask: number, expanded: boolean): string {
  if (!expanded) return positionKey;
  return `${positionKey}|${appleMask}`;
}

export function normalizeAppleMask(
  mask: number,
  positionKey: string,
  appleIndexByKey: Record<string, number>,
  appleCount: number
): number {
  if (appleCount === 0) return 0;
  const safeMask = mask & ((1 << appleCount) - 1);
  const appleIndex = appleIndexByKey[positionKey];
  if (appleIndex === undefined) return safeMask;
  return safeMask | (1 << appleIndex);
}

export function collectAppleInMask(
  mask: number,
  positionKey: string,
  appleIndexByKey: Record<string, number>
): number {
  const appleIndex = appleIndexByKey[positionKey];
  if (appleIndex === undefined) return mask;
  return mask | (1 << appleIndex);
}

export function buildCollectedApplesMap(
  appleKeys: string[],
  appleMask: number
): Record<string, boolean> {
  const collected: Record<string, boolean> = {};
  appleKeys.forEach((key, index) => {
    if ((appleMask & (1 << index)) !== 0) {
      collected[key] = true;
    }
  });
  return collected;
}

export function formatStateLabel(
  positionKey: string,
  appleMask: number,
  expanded: boolean,
  appleKeys: string[],
  labelMap: Record<string, string>
): string {
  const baseLabel = labelMap[positionKey] ?? "?";
  if (!expanded || appleKeys.length === 0) return baseLabel;

  const applesText = appleKeys
    .map((_, index) => ((appleMask & (1 << index)) !== 0 ? "❌" : "🍎"))
    .join(", ");
  return `(${baseLabel},${applesText})`;
}

export function buildGraphTransitions(
  walkableKeys: string[],
  walkableSet: Set<string>,
  enabledActions: Action[],
  terminalSet: Set<string>
): Record<string, string[]> {
  const edges: Record<string, string[]> = {};
  walkableKeys.forEach((key) => {
    if (terminalSet.has(key)) {
      edges[key] = [];
      return;
    }
    edges[key] = getValidActions(key, walkableSet, enabledActions).map((action) =>
      nextStateFromAction(key, action)
    );
  });
  return edges;
}

export function getReachableNodes(startKey: string, edges: Record<string, string[]>): Set<string> {
  const reachable = new Set<string>();
  const queue: string[] = [startKey];

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || reachable.has(current)) continue;
    reachable.add(current);
    (edges[current] ?? []).forEach((next) => {
      if (!reachable.has(next)) {
        queue.push(next);
      }
    });
  }

  return reachable;
}

export function computeSccSizeByNode(
  nodes: string[],
  edges: Record<string, string[]>
): Record<string, number> {
  const visited = new Set<string>();
  const order: string[] = [];

  const dfsForward = (node: string) => {
    if (visited.has(node)) return;
    visited.add(node);
    (edges[node] ?? []).forEach((next) => {
      if (!visited.has(next)) {
        dfsForward(next);
      }
    });
    order.push(node);
  };

  nodes.forEach((node) => {
    if (!visited.has(node)) {
      dfsForward(node);
    }
  });

  const reverseEdges: Record<string, string[]> = {};
  nodes.forEach((node) => {
    reverseEdges[node] = [];
  });
  nodes.forEach((node) => {
    (edges[node] ?? []).forEach((next) => {
      if (reverseEdges[next]) {
        reverseEdges[next].push(node);
      }
    });
  });

  const assigned = new Set<string>();
  const sizeByNode: Record<string, number> = {};

  const dfsReverse = (node: string, component: string[]) => {
    if (assigned.has(node)) return;
    assigned.add(node);
    component.push(node);
    (reverseEdges[node] ?? []).forEach((previous) => {
      if (!assigned.has(previous)) {
        dfsReverse(previous, component);
      }
    });
  };

  for (let index = order.length - 1; index >= 0; index -= 1) {
    const node = order[index];
    if (assigned.has(node)) continue;
    const component: string[] = [];
    dfsReverse(node, component);
    const componentSize = component.length;
    component.forEach((member) => {
      sizeByNode[member] = componentSize;
    });
  }

  return sizeByNode;
}

export function shouldExpandAppleStates(params: {
  walkableKeys: string[];
  walkableSet: Set<string>;
  enabledActions: Action[];
  startKey: string;
  goalKey: string;
  appleKeys: string[];
  cellItems: Record<string, TileItem>;
}): boolean {
  const {
    walkableKeys,
    walkableSet,
    enabledActions,
    startKey,
    goalKey,
    appleKeys,
    cellItems
  } = params;

  if (appleKeys.length === 0 || enabledActions.length === 0) return false;
  if (!walkableSet.has(startKey)) return false;

  const birdKeys = Object.entries(cellItems)
    .filter(([, item]) => item === "bird")
    .map(([key]) => key);
  const terminalSet = new Set<string>([goalKey, ...birdKeys]);
  const edges = buildGraphTransitions(
    walkableKeys,
    walkableSet,
    enabledActions,
    terminalSet
  );
  const reachable = getReachableNodes(startKey, edges);
  const reachableNodes = walkableKeys.filter((key) => reachable.has(key));
  if (reachableNodes.length === 0) return false;

  const reachableEdges: Record<string, string[]> = {};
  reachableNodes.forEach((key) => {
    reachableEdges[key] = (edges[key] ?? []).filter((next) => reachable.has(next));
  });

  const sccSizeByNode = computeSccSizeByNode(reachableNodes, reachableEdges);
  return appleKeys.some((appleKey) => (sccSizeByNode[appleKey] ?? 1) > 1);
}

export interface StepPreviewParameters {
  agentKey: string;
  agentAppleMask: number;
  action: Action;
  goalKey: string;
  cellItems: Record<string, TileItem>;
  labelMap: Record<string, string>;
  appleKeys: string[];
  appleIndexByKey: Record<string, number>;
  expandedStateSpace: boolean;
}

export interface StepPreview {
  nextKey: string;
  nextMask: number;
  nextStateKey: string;
  nextStateLabel: string;
  reward: number;
  terminal: boolean;
}

export function previewStep(params: StepPreviewParameters): StepPreview {
  const {
    agentKey,
    agentAppleMask,
    action,
    goalKey,
    cellItems,
    labelMap,
    appleKeys,
    appleIndexByKey,
    expandedStateSpace
  } = params;
  const nextKey = nextStateFromAction(agentKey, action);
  let nextMask = normalizeAppleMask(
    agentAppleMask,
    agentKey,
    appleIndexByKey,
    appleKeys.length
  );
  let reward = 0;
  const tileItem = cellItems[nextKey] ?? "none";
  const reachedGoal = nextKey === goalKey;
  const hitBird = tileItem === "bird";

  if (hitBird) {
    reward = -2;
  } else if (reachedGoal) {
    reward = 2;
  } else if (tileItem === "apple") {
    const collectedMask = collectAppleInMask(nextMask, nextKey, appleIndexByKey);
    if (collectedMask !== nextMask) reward = 1;
    nextMask = collectedMask;
  }
  nextMask = normalizeAppleMask(
    nextMask,
    nextKey,
    appleIndexByKey,
    appleKeys.length
  );

  return {
    nextKey,
    nextMask,
    nextStateKey: encodeStateKey(nextKey, nextMask, expandedStateSpace),
    nextStateLabel: formatStateLabel(
      nextKey,
      nextMask,
      expandedStateSpace,
      appleKeys,
      labelMap
    ),
    reward,
    terminal: reachedGoal || hitBird
  };
}

export function performStep(params: {
  qTable: QTable;
  agentKey: string;
  agentAppleMask: number;
  episodeScore: number;
  stepCount: number;
  walkableSet: Set<string>;
  goalKey: string;
  cellItems: Record<string, TileItem>;
  labelMap: Record<string, string>;
  appleKeys: string[];
  appleIndexByKey: Record<string, number>;
  expandedStateSpace: boolean;
  validActionsByState: Record<string, Action[]>;
  alpha: number;
  gamma: number;
  epsilon: number;
  action?: Action;
}): StepOutcome | null {
  const {
    qTable,
    agentKey,
    agentAppleMask,
    episodeScore,
    stepCount,
    walkableSet,
    goalKey,
    cellItems,
    labelMap,
    appleKeys,
    appleIndexByKey,
    expandedStateSpace,
    validActionsByState,
    alpha,
    gamma,
    epsilon
  } = params;

  const appleCount = appleKeys.length;
  const normalizedMask = normalizeAppleMask(
    agentAppleMask,
    agentKey,
    appleIndexByKey,
    appleCount
  );
  const currentStateKey = encodeStateKey(agentKey, normalizedMask, expandedStateSpace);
  const row = qTable[currentStateKey] ?? makeZeroRow();
  const validActions = validActionsByState[agentKey] ?? [];
  const action = params.action ?? pickAction(row, epsilon, validActions);
  if (!action || !validActions.includes(action)) {
    return null;
  }

  const nextKey = nextStateFromAction(agentKey, action);
  if (!walkableSet.has(nextKey)) {
    return null;
  }

  const { nextMask, nextStateKey, nextStateLabel, reward, terminal } = previewStep({
    agentKey,
    agentAppleMask,
    action,
    goalKey,
    cellItems,
    labelMap,
    appleKeys,
    appleIndexByKey,
    expandedStateSpace
  });

  const oldQ = row[action];
  const nextRow = terminal ? undefined : qTable[nextStateKey];
  const nextValidActions = terminal ? [] : validActionsByState[nextKey] ?? [];
  const maxNextQ = terminal ? 0 : maxQ(nextRow, nextValidActions);
  const target = reward + gamma * maxNextQ;
  const rawNewQ = oldQ + alpha * (target - oldQ);
  const newQ = roundQValue(rawNewQ);

  const nextQTable = cloneQTable(qTable);
  if (!nextQTable[currentStateKey]) {
    nextQTable[currentStateKey] = makeZeroRow();
  }
  nextQTable[currentStateKey][action] = newQ;

  return {
    qTable: nextQTable,
    agentKey: nextKey,
    agentAppleMask: nextMask,
    reward,
    done: terminal,
    episodeScore: episodeScore + reward,
    stepCount: stepCount + 1,
    calculation: {
      stateKey: currentStateKey,
      nextStateKey,
      stateLabel: formatStateLabel(
        agentKey,
        normalizedMask,
        expandedStateSpace,
        appleKeys,
        labelMap
      ),
      nextStateLabel,
      terminal,
      action,
      reward,
      oldQ,
      maxNextQ,
      target,
      rawNewQ,
      newQ,
      alpha,
      gamma
    },
    action
  };
}

export function hasPathToGoal(
  walkableSet: Set<string>,
  startKey: string,
  goalKey: string,
  enabledActions: Action[]
): boolean {
  if (!walkableSet.has(startKey) || !walkableSet.has(goalKey)) return false;
  if (startKey === goalKey) return false;
  if (enabledActions.length === 0) return false;

  const queue: string[] = [startKey];
  const visited = new Set<string>([startKey]);

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current) continue;
    if (current === goalKey) return true;

    const currentCoord = parseKey(current);
    enabledActions.forEach((action) => {
      const delta = ACTION_DELTAS[action];
      const next = keyFromXY(currentCoord.x + delta.x, currentCoord.y + delta.y);
      if (!walkableSet.has(next) || visited.has(next)) return;
      visited.add(next);
      queue.push(next);
    });
  }

  return false;
}
