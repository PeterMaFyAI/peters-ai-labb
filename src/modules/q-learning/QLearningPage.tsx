import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent
} from "react";
import "./qLearning.css";

type Action = "right" | "down";
type TileItem = "none" | "apple" | "bird";
type QRow = Record<Action, number>;
type QTable = Record<string, QRow>;

interface QCalculation {
  stateLabel: string;
  nextStateLabel: string;
  action: Action;
  reward: number;
  oldQ: number;
  maxNextQ: number;
  target: number;
  newQ: number;
  alpha: number;
  gamma: number;
}

interface StepOutcome {
  qTable: QTable;
  agentKey: string;
  collectedApples: Record<string, boolean>;
  reward: number;
  done: boolean;
  episodeScore: number;
  stepCount: number;
  calculation: QCalculation;
  action: Action;
}

interface ScoreFlash {
  id: number;
  text: string;
  positive: boolean;
}

interface Coordinate {
  x: number;
  y: number;
}

const ACTIONS: Action[] = ["right", "down"];
const ACTION_LABELS: Record<Action, string> = {
  right: "Hoger",
  down: "Ner"
};
const ACTION_SYMBOLS: Record<Action, string> = {
  right: "->",
  down: "v"
};
const ACTION_DELTAS: Record<Action, Coordinate> = {
  right: { x: 1, y: 0 },
  down: { x: 0, y: 1 }
};

const EDITOR_SIDE = 10;
const MAX_WALKABLE_CELLS = 10;
const MAX_STEPS_PER_EPISODE = 220;
const OTHER_STATE_LABELS = ["B", "C", "D", "E", "F", "G", "H", "I"];
const DEFAULT_START = "0,0";
const DEFAULT_GOAL = "2,1";
const DEFAULT_WALKABLE = ["0,0", "1,0", "2,0", "0,1", "1,1", "2,1"];
const DEFAULT_ITEMS: Record<string, TileItem> = {
  "1,0": "apple",
  "1,1": "bird"
};

const wait = (ms: number): Promise<void> =>
  new Promise((resolve) => window.setTimeout(resolve, ms));

function keyFromXY(x: number, y: number): string {
  return `${x},${y}`;
}

function parseKey(key: string): Coordinate {
  const [xRaw, yRaw] = key.split(",");
  return {
    x: Number.parseInt(xRaw, 10),
    y: Number.parseInt(yRaw, 10)
  };
}

function sortKeys(keys: string[]): string[] {
  return [...keys].sort((left, right) => {
    const leftCoord = parseKey(left);
    const rightCoord = parseKey(right);
    if (leftCoord.y === rightCoord.y) return leftCoord.x - rightCoord.x;
    return leftCoord.y - rightCoord.y;
  });
}

function makeZeroRow(): QRow {
  return {
    right: 0,
    down: 0
  };
}

function cloneQTable(qTable: QTable): QTable {
  const next: QTable = {};
  Object.entries(qTable).forEach(([stateKey, row]) => {
    next[stateKey] = { ...row };
  });
  return next;
}

function createZeroQTable(stateKeys: string[]): QTable {
  const table: QTable = {};
  stateKeys.forEach((key) => {
    table[key] = makeZeroRow();
  });
  return table;
}

function alignQTable(previous: QTable, stateKeys: string[]): QTable {
  const next: QTable = {};
  stateKeys.forEach((key) => {
    const existing = previous[key];
    next[key] = existing
      ? {
          right: existing.right ?? 0,
          down: existing.down ?? 0
        }
      : makeZeroRow();
  });
  return next;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function formatSigned(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function nextStateFromAction(stateKey: string, action: Action): string {
  const currentCoord = parseKey(stateKey);
  const delta = ACTION_DELTAS[action];
  return keyFromXY(currentCoord.x + delta.x, currentCoord.y + delta.y);
}

function getValidActions(stateKey: string, walkableSet: Set<string>): Action[] {
  return ACTIONS.filter((action) => walkableSet.has(nextStateFromAction(stateKey, action)));
}

function maxQ(row: QRow | undefined, validActions: Action[]): number {
  if (!row || validActions.length === 0) return 0;
  return Math.max(...validActions.map((action) => row[action]));
}

function pickAction(row: QRow, epsilon: number, validActions: Action[]): Action | null {
  if (validActions.length === 0) return null;

  if (Math.random() < epsilon) {
    return validActions[Math.floor(Math.random() * validActions.length)];
  }

  const bestValue = maxQ(row, validActions);
  const bestActions = validActions.filter((action) => row[action] === bestValue);
  return bestActions[Math.floor(Math.random() * bestActions.length)];
}

function buildLabelMap(
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

function normalizeItems(
  items: Record<string, TileItem>,
  walkableSet: Set<string>,
  startKey: string,
  goalKey: string
): Record<string, TileItem> {
  const next: Record<string, TileItem> = {};
  Object.entries(items).forEach(([key, item]) => {
    if (!walkableSet.has(key)) return;
    if (key === startKey || key === goalKey) return;
    if (item === "apple" || item === "bird") {
      next[key] = item;
    }
  });
  return next;
}

function performStep(params: {
  qTable: QTable;
  agentKey: string;
  collectedApples: Record<string, boolean>;
  episodeScore: number;
  stepCount: number;
  walkableSet: Set<string>;
  goalKey: string;
  cellItems: Record<string, TileItem>;
  labelMap: Record<string, string>;
  validActionsByState: Record<string, Action[]>;
  alpha: number;
  gamma: number;
  epsilon: number;
}): StepOutcome | null {
  const {
    qTable,
    agentKey,
    collectedApples,
    episodeScore,
    stepCount,
    walkableSet,
    goalKey,
    cellItems,
    labelMap,
    validActionsByState,
    alpha,
    gamma,
    epsilon
  } = params;

  const row = qTable[agentKey] ?? makeZeroRow();
  const validActions = validActionsByState[agentKey] ?? [];
  const action = pickAction(row, epsilon, validActions);
  if (!action) {
    return null;
  }

  const nextKey = nextStateFromAction(agentKey, action);
  if (!walkableSet.has(nextKey)) {
    return null;
  }

  const nextCollected = { ...collectedApples };
  let reward = 0;
  const tileItem = cellItems[nextKey] ?? "none";
  const reachedGoal = nextKey === goalKey;
  const hitBird = tileItem === "bird";

  if (hitBird) {
    reward = -2;
  } else if (reachedGoal) {
    reward = 2;
  } else if (tileItem === "apple" && !nextCollected[nextKey]) {
    reward += 1;
    nextCollected[nextKey] = true;
  }

  const terminal = reachedGoal || hitBird;

  const oldQ = row[action];
  const nextRow = terminal ? undefined : qTable[nextKey];
  const nextValidActions = terminal ? [] : validActionsByState[nextKey] ?? [];
  const maxNextQ = terminal ? 0 : maxQ(nextRow, nextValidActions);
  const target = reward + gamma * maxNextQ;
  const newQ = oldQ + alpha * (target - oldQ);

  const nextQTable = cloneQTable(qTable);
  if (!nextQTable[agentKey]) {
    nextQTable[agentKey] = makeZeroRow();
  }
  nextQTable[agentKey][action] = newQ;

  return {
    qTable: nextQTable,
    agentKey: nextKey,
    collectedApples: nextCollected,
    reward,
    done: terminal,
    episodeScore: episodeScore + reward,
    stepCount: stepCount + 1,
    calculation: {
      stateLabel: labelMap[agentKey] ?? "?",
      nextStateLabel: labelMap[nextKey] ?? "?",
      action,
      reward,
      oldQ,
      maxNextQ,
      target,
      newQ,
      alpha,
      gamma
    },
    action
  };
}

function hasPathToGoal(
  walkableSet: Set<string>,
  startKey: string,
  goalKey: string
): boolean {
  if (!walkableSet.has(startKey) || !walkableSet.has(goalKey)) return false;
  if (startKey === goalKey) return false;

  const queue: string[] = [startKey];
  const visited = new Set<string>([startKey]);

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current) continue;
    if (current === goalKey) return true;

    const currentCoord = parseKey(current);
    ACTIONS.forEach((action) => {
      const delta = ACTION_DELTAS[action];
      const next = keyFromXY(currentCoord.x + delta.x, currentCoord.y + delta.y);
      if (!walkableSet.has(next) || visited.has(next)) return;
      visited.add(next);
      queue.push(next);
    });
  }

  return false;
}

function cellVisualItem(
  key: string,
  startKey: string,
  goalKey: string,
  cellItems: Record<string, TileItem>,
  collectedApples: Record<string, boolean>
): string {
  if (key === startKey || key === goalKey) return "";
  const item = cellItems[key] ?? "none";
  if (item === "apple" && collectedApples[key]) return "";
  if (item === "apple") return "🍎";
  if (item === "bird") return "🐦";
  return "";
}

function QLearningPage(): JSX.Element {
  const [walkableKeys, setWalkableKeys] = useState<string[]>(() =>
    sortKeys(DEFAULT_WALKABLE)
  );
  const [startKey, setStartKey] = useState(DEFAULT_START);
  const [goalKey, setGoalKey] = useState(DEFAULT_GOAL);
  const [cellItems, setCellItems] = useState<Record<string, TileItem>>(
    DEFAULT_ITEMS
  );
  const labelMap = useMemo(
    () => buildLabelMap(walkableKeys, startKey, goalKey),
    [walkableKeys, startKey, goalKey]
  );
  const qStateKeys = useMemo(() => {
    return walkableKeys
      .filter((key) => key !== goalKey)
      .sort((left, right) => {
        const leftLabel = labelMap[left] ?? "";
        const rightLabel = labelMap[right] ?? "";
        return leftLabel.localeCompare(rightLabel);
      });
  }, [goalKey, labelMap, walkableKeys]);

  const [qTable, setQTable] = useState<QTable>(() =>
    createZeroQTable(DEFAULT_WALKABLE.filter((key) => key !== DEFAULT_GOAL))
  );
  const [agentKey, setAgentKey] = useState(startKey);
  const [collectedApples, setCollectedApples] = useState<Record<string, boolean>>(
    {}
  );
  const [episodeDone, setEpisodeDone] = useState(false);
  const [episodeScore, setEpisodeScore] = useState(0);
  const [stepCount, setStepCount] = useState(0);
  const [episodeIndex, setEpisodeIndex] = useState(1);
  const [lastReward, setLastReward] = useState(0);
  const [statusText, setStatusText] = useState(
    "Stega for att se hur Q-tabellen uppdateras."
  );
  const [calculation, setCalculation] = useState<QCalculation | null>(null);

  const [alpha, setAlpha] = useState(0.5);
  const [gamma, setGamma] = useState(0.9);
  const [epsilon, setEpsilon] = useState(0.9);

  const [isAutoRunning, setIsAutoRunning] = useState(false);
  const [autoMode, setAutoMode] = useState<"animated" | "fast" | null>(null);
  const [animationDelay, setAnimationDelay] = useState(420);
  const [animatedEpisodes, setAnimatedEpisodes] = useState(8);
  const [fastEpisodes, setFastEpisodes] = useState(150);
  const [stopRequested, setStopRequested] = useState(false);
  const stopRequestedRef = useRef(false);

  const [scoreFlashes, setScoreFlashes] = useState<ScoreFlash[]>([]);
  const [editMode, setEditMode] = useState(false);
  const [dragToken, setDragToken] = useState<"start" | "goal" | null>(null);
  const [editorInfo, setEditorInfo] = useState("");

  const flashIdRef = useRef(0);
  const flashTimeoutsRef = useRef<number[]>([]);

  const walkableSet = useMemo(() => new Set(walkableKeys), [walkableKeys]);
  const validActionsByState = useMemo(() => {
    const map: Record<string, Action[]> = {};
    walkableKeys.forEach((key) => {
      map[key] = getValidActions(key, walkableSet);
    });
    return map;
  }, [walkableKeys, walkableSet]);
  const normalizedItems = useMemo(
    () => normalizeItems(cellItems, walkableSet, startKey, goalKey),
    [cellItems, goalKey, startKey, walkableSet]
  );

  const gridBounds = useMemo(() => {
    const coords = walkableKeys.map(parseKey);
    const xValues = coords.map((coord) => coord.x);
    const yValues = coords.map((coord) => coord.y);
    return {
      minX: Math.min(...xValues),
      maxX: Math.max(...xValues),
      minY: Math.min(...yValues),
      maxY: Math.max(...yValues)
    };
  }, [walkableKeys]);

  const trainingCells = useMemo(() => {
    const cells: string[] = [];
    for (let y = gridBounds.minY; y <= gridBounds.maxY; y += 1) {
      for (let x = gridBounds.minX; x <= gridBounds.maxX; x += 1) {
        cells.push(keyFromXY(x, y));
      }
    }
    return cells;
  }, [gridBounds.maxX, gridBounds.maxY, gridBounds.minX, gridBounds.minY]);

  const trainingColumns = gridBounds.maxX - gridBounds.minX + 1;
  const hasGoalPath = useMemo(
    () => hasPathToGoal(walkableSet, startKey, goalKey),
    [goalKey, startKey, walkableSet]
  );

  useEffect(() => {
    setQTable((previous) => alignQTable(previous, qStateKeys));
  }, [qStateKeys]);

  useEffect(() => {
    stopRequestedRef.current = stopRequested;
  }, [stopRequested]);

  useEffect(() => {
    return () => {
      flashTimeoutsRef.current.forEach((timeoutId) => {
        window.clearTimeout(timeoutId);
      });
      flashTimeoutsRef.current = [];
    };
  }, []);

  useEffect(() => {
    const handleMouseUp = () => {
      setDragToken(null);
    };
    window.addEventListener("mouseup", handleMouseUp);
    return () => window.removeEventListener("mouseup", handleMouseUp);
  }, []);

  const pushRewardFlash = useCallback((reward: number) => {
    const nextId = flashIdRef.current + 1;
    flashIdRef.current = nextId;
    const positive = reward >= 0;
    const text = `${reward >= 0 ? "+" : ""}${reward.toFixed(1)} p`;

    setScoreFlashes((previous) => [...previous, { id: nextId, text, positive }]);

    const timeoutId = window.setTimeout(() => {
      setScoreFlashes((previous) => previous.filter((flash) => flash.id !== nextId));
    }, 920);
    flashTimeoutsRef.current.push(timeoutId);
  }, []);

  const syncOutcomeToUi = useCallback(
    (outcome: StepOutcome, currentEpisode: number, withFlash: boolean) => {
      setQTable(outcome.qTable);
      setAgentKey(outcome.agentKey);
      setCollectedApples(outcome.collectedApples);
      setEpisodeDone(outcome.done);
      setEpisodeScore(outcome.episodeScore);
      setStepCount(outcome.stepCount);
      setLastReward(outcome.reward);
      setCalculation(outcome.calculation);
      setEpisodeIndex(currentEpisode);
      setStatusText(
        `${outcome.calculation.stateLabel} --${
          ACTION_LABELS[outcome.action]
        }--> ${outcome.calculation.nextStateLabel}. Beloning ${formatSigned(
          outcome.reward
        )}.${
          outcome.done
            ? " Episoden ar klar. Klicka pa \"Ny episod\" for att ga tillbaka till A."
            : ""
        }`
      );
      if (withFlash) {
        pushRewardFlash(outcome.reward);
      }
    },
    [pushRewardFlash]
  );

  const resetEpisode = useCallback(
    (nextEpisodeIndex?: number, message?: string) => {
      setAgentKey(startKey);
      setCollectedApples({});
      setEpisodeDone(false);
      setEpisodeScore(0);
      setStepCount(0);
      setLastReward(0);
      setCalculation(null);
      if (typeof nextEpisodeIndex === "number") {
        setEpisodeIndex(nextEpisodeIndex);
      }
      if (message) {
        setStatusText(message);
      }
    },
    [startKey]
  );

  const resetQForCurrentGrid = useCallback(
    (message: string) => {
      setQTable(createZeroQTable(qStateKeys));
      setScoreFlashes([]);
      setIsAutoRunning(false);
      setAutoMode(null);
      setStopRequested(false);
      setEpisodeIndex(1);
      setAgentKey(startKey);
      setCollectedApples({});
      setEpisodeDone(false);
      setEpisodeScore(0);
      setStepCount(0);
      setLastReward(0);
      setCalculation(null);
      setStatusText(message);
    },
    [qStateKeys, startKey]
  );

  const handleManualStep = () => {
    if (isAutoRunning || editMode) return;

    if (episodeDone) {
      const nextIndex = episodeIndex + 1;
      resetEpisode(nextIndex, `Ny episod ${nextIndex} startad.`);
      return;
    }

    const outcome = performStep({
      qTable,
      agentKey,
      collectedApples,
      episodeScore,
      stepCount,
      walkableSet,
      goalKey,
      cellItems: normalizedItems,
      labelMap,
      validActionsByState,
      alpha,
      gamma,
      epsilon
    });

    if (!outcome) {
      setStatusText(
        `Inga giltiga drag finns fran ${labelMap[agentKey] ?? "aktuellt tillstand"}.`
      );
      return;
    }

    syncOutcomeToUi(outcome, episodeIndex, true);
  };

  const handleResetQ = () => {
    if (isAutoRunning) return;
    resetQForCurrentGrid("Q-tabellen nollstalldes och episod 1 ar redo.");
  };

  const runAnimatedTraining = useCallback(async () => {
    if (isAutoRunning || editMode) return;

    const episodeCount = Math.max(1, Math.floor(animatedEpisodes || 1));
    setIsAutoRunning(true);
    setAutoMode("animated");
    setStopRequested(false);
    stopRequestedRef.current = false;
    setStatusText(`Animerad traning startad (${episodeCount} episoder).`);

    let localQ = cloneQTable(qTable);
    const firstEpisode = episodeIndex;
    let stopped = false;

    for (let episodeNumber = 0; episodeNumber < episodeCount; episodeNumber += 1) {
      const activeEpisode = firstEpisode + episodeNumber;
      let localAgent = startKey;
      let localCollected: Record<string, boolean> = {};
      let localScore = 0;
      let localStepCount = 0;
      let done = false;
      let blocked = false;

      setEpisodeIndex(activeEpisode);
      setAgentKey(startKey);
      setCollectedApples({});
      setEpisodeDone(false);
      setEpisodeScore(0);
      setStepCount(0);
      setLastReward(0);
      setCalculation(null);

      for (
        let stepIndex = 0;
        stepIndex < MAX_STEPS_PER_EPISODE;
        stepIndex += 1
      ) {
        if (stopRequestedRef.current) {
          stopped = true;
          break;
        }

        const outcome = performStep({
          qTable: localQ,
          agentKey: localAgent,
          collectedApples: localCollected,
          episodeScore: localScore,
          stepCount: localStepCount,
          walkableSet,
          goalKey,
          cellItems: normalizedItems,
          labelMap,
          validActionsByState,
          alpha,
          gamma,
          epsilon
        });

        if (!outcome) {
          blocked = true;
          setStatusText(
            `Episod ${activeEpisode} avbruten: inga giltiga drag fran ${
              labelMap[localAgent] ?? "aktuellt tillstand"
            }.`
          );
          break;
        }

        localQ = outcome.qTable;
        localAgent = outcome.agentKey;
        localCollected = outcome.collectedApples;
        localScore = outcome.episodeScore;
        localStepCount = outcome.stepCount;
        done = outcome.done;

        syncOutcomeToUi(outcome, activeEpisode, true);
        await wait(animationDelay);

        if (done) {
          break;
        }
      }

      if (stopped) break;

      if (!done && !blocked) {
        setStatusText(
          `Episod ${activeEpisode} stoppades efter ${MAX_STEPS_PER_EPISODE} steg (ingen maltraff).`
        );
      }
    }

    const nextEpisode = firstEpisode + episodeCount;
    setQTable(localQ);

    if (stopped) {
      setStatusText("Animerad traning stoppad.");
    } else {
      resetEpisode(nextEpisode, `Animerad traning klar (${episodeCount} episoder).`);
    }

    setIsAutoRunning(false);
    setAutoMode(null);
    setStopRequested(false);
    stopRequestedRef.current = false;
  }, [
    alpha,
    animatedEpisodes,
    animationDelay,
    editMode,
    episodeIndex,
    epsilon,
    gamma,
    goalKey,
    isAutoRunning,
    labelMap,
    normalizedItems,
    qTable,
    resetEpisode,
    startKey,
    syncOutcomeToUi,
    validActionsByState,
    walkableSet
  ]);

  const handleAnimatedButton = () => {
    if (!isAutoRunning) {
      void runAnimatedTraining();
      return;
    }

    if (autoMode === "animated") {
      setStopRequested(true);
      setStatusText("Stoppsignal skickad. Traning avslutas...");
    }
  };

  const handleFastTraining = () => {
    if (isAutoRunning || editMode) return;

    const episodeCount = Math.max(1, Math.floor(fastEpisodes || 1));
    setIsAutoRunning(true);
    setAutoMode("fast");
    setStatusText(`Snabbtraning kor ${episodeCount} episoder...`);

    let localQ = cloneQTable(qTable);
    let nextEpisode = episodeIndex;
    let lastCalculation: QCalculation | null = calculation;

    for (let episodeNumber = 0; episodeNumber < episodeCount; episodeNumber += 1) {
      let localAgent = startKey;
      let localCollected: Record<string, boolean> = {};
      let localScore = 0;
      let localStepCount = 0;

      for (
        let stepIndex = 0;
        stepIndex < MAX_STEPS_PER_EPISODE;
        stepIndex += 1
      ) {
        const outcome = performStep({
          qTable: localQ,
          agentKey: localAgent,
          collectedApples: localCollected,
          episodeScore: localScore,
          stepCount: localStepCount,
          walkableSet,
          goalKey,
          cellItems: normalizedItems,
          labelMap,
          validActionsByState,
          alpha,
          gamma,
          epsilon
        });

        if (!outcome) {
          break;
        }

        localQ = outcome.qTable;
        localAgent = outcome.agentKey;
        localCollected = outcome.collectedApples;
        localScore = outcome.episodeScore;
        localStepCount = outcome.stepCount;
        lastCalculation = outcome.calculation;

        if (outcome.done) {
          break;
        }
      }

      nextEpisode += 1;
    }

    setQTable(localQ);
    setCalculation(lastCalculation);
    resetEpisode(
      nextEpisode,
      `Snabbtraning klar: ${episodeCount} episoder genomforda utan animation.`
    );
    setIsAutoRunning(false);
    setAutoMode(null);
  };

  const toggleEditMode = () => {
    if (isAutoRunning) return;
    if (!editMode) {
      setEditMode(true);
      setEditorInfo(
        "Redigeringslage aktivt. Vansterklick: lagg till/ta bort ruta. Hogerklick: neutral -> apple -> fagel."
      );
      resetQForCurrentGrid("Q-tabellen nollstalldes nar redigeringslaget startade.");
      return;
    }

    setEditMode(false);
    setEditorInfo("");
    resetQForCurrentGrid("Ny Q-tabell initierad nar redigeringslaget avslutades.");
  };

  const handleEditorMouseDown = (
    event: MouseEvent<HTMLDivElement>,
    key: string
  ) => {
    if (!editMode || event.button !== 0) return;
    event.preventDefault();

    if (key === startKey) {
      setDragToken("start");
      return;
    }

    if (key === goalKey) {
      setDragToken("goal");
      return;
    }

    if (walkableSet.has(key)) {
      setWalkableKeys((previous) => sortKeys(previous.filter((cell) => cell !== key)));
      setCellItems((previous) => {
        const next = { ...previous };
        delete next[key];
        return next;
      });
      setEditorInfo("Ruta borttagen.");
      return;
    }

    if (walkableKeys.length >= MAX_WALKABLE_CELLS) {
      setEditorInfo("Max tio rutor tillatet (inklusive A och S).");
      return;
    }

    setWalkableKeys((previous) => sortKeys([...previous, key]));
    setEditorInfo("Ny ruta placerad.");
  };

  const moveMarker = useCallback(
    (marker: "start" | "goal", targetKey: string) => {
      if (marker === "start" && targetKey === goalKey) {
        setEditorInfo("A och S kan inte dela ruta.");
        return;
      }
      if (marker === "goal" && targetKey === startKey) {
        setEditorInfo("A och S kan inte dela ruta.");
        return;
      }

      const sourceKey = marker === "start" ? startKey : goalKey;
      const targetExists = walkableSet.has(targetKey);

      if (!targetExists && walkableKeys.length >= MAX_WALKABLE_CELLS) {
        setEditorInfo("Max tio rutor tillatet.");
        return;
      }

      setWalkableKeys((previous) => {
        if (targetExists) return previous;
        return sortKeys(
          previous.map((key) => (key === sourceKey ? targetKey : key))
        );
      });

      setCellItems((previous) => {
        const next = { ...previous };
        delete next[sourceKey];
        delete next[targetKey];
        return next;
      });

      if (marker === "start") {
        setStartKey(targetKey);
      } else {
        setGoalKey(targetKey);
      }
      setEditorInfo(`${marker === "start" ? "Start" : "Mal"} flyttad.`);
    },
    [goalKey, startKey, walkableKeys.length, walkableSet]
  );

  const handleEditorMouseUp = (
    event: MouseEvent<HTMLDivElement>,
    key: string
  ) => {
    if (!editMode || event.button !== 0) return;
    if (!dragToken) return;
    event.preventDefault();
    moveMarker(dragToken, key);
    setDragToken(null);
  };

  const handleEditorRightClick = (
    event: MouseEvent<HTMLDivElement>,
    key: string
  ) => {
    if (!editMode) return;
    event.preventDefault();

    if (!walkableSet.has(key) || key === startKey || key === goalKey) return;

    const current = normalizedItems[key] ?? "none";
    const nextItem: TileItem =
      current === "none" ? "apple" : current === "apple" ? "bird" : "none";

    setCellItems((previous) => {
      const next = { ...previous };
      if (nextItem === "none") {
        delete next[key];
      } else {
        next[key] = nextItem;
      }
      return next;
    });
  };

  const editorCells = useMemo(() => {
    const cells: string[] = [];
    for (let y = 0; y < EDITOR_SIDE; y += 1) {
      for (let x = 0; x < EDITOR_SIDE; x += 1) {
        cells.push(keyFromXY(x, y));
      }
    }
    return cells;
  }, []);

  const stateCount = walkableKeys.length;
  const canTrain = hasGoalPath && stateCount >= 2;

  return (
    <div className="container page-flow q-learning-page">
      <section className="section">
        <h1 className="q-learning-title">Q-learning: masken och appeljakten</h1>
        <p className="q-learning-lead">
          Utforska forstarkningsinlarning steg for steg. Agenten (🐛) lar sig
          hitta till malet S med hjalp av beloningar, Q-tabell och epsilon-greedy.
        </p>

        <div className="q-learning-toolbar">
          <button
            className={`q-btn ${editMode ? "q-btn-secondary" : "q-btn-primary"}`}
            onClick={toggleEditMode}
            disabled={isAutoRunning}
          >
            {editMode ? "Lamna redigeringslage" : "Redigeringslage (10x10)"}
          </button>
          <button
            className="q-btn"
            onClick={handleResetQ}
            disabled={isAutoRunning}
          >
            Nollstall Q-tabell
          </button>
        </div>
      </section>

      <section className="section q-learning-main-section">
        <div className="q-learning-main-grid">
          <article className="q-card">
            <div className="q-card-head">
              <h2>Miljo</h2>
              <p>
                Episod {episodeIndex} - steg {stepCount} - poang{" "}
                <strong>{episodeScore.toFixed(1)}</strong>
              </p>
            </div>

            <div className="q-training-wrap">
              <div
                className="q-training-grid"
                style={{
                  gridTemplateColumns: `repeat(${trainingColumns}, minmax(74px, 1fr))`
                }}
              >
                {trainingCells.map((key) => {
                  const isWalkable = walkableSet.has(key);
                  const isStart = key === startKey;
                  const isGoal = key === goalKey;
                  const isAgent = key === agentKey;
                  const label = labelMap[key] ?? "";
                  const visualItem = cellVisualItem(
                    key,
                    startKey,
                    goalKey,
                    normalizedItems,
                    collectedApples
                  );

                  return (
                    <div
                      key={`training-${key}`}
                      className={`q-cell ${
                        isWalkable ? "q-cell-walkable" : "q-cell-blocked"
                      } ${isStart ? "q-cell-start" : ""} ${
                        isGoal ? "q-cell-goal" : ""
                      } ${isAgent ? "q-cell-agent" : ""}`}
                    >
                      {isWalkable ? (
                        <>
                          <span className="q-cell-label">{label}</span>
                          {visualItem ? (
                            <span className="q-cell-item" aria-hidden="true">
                              {visualItem}
                            </span>
                          ) : null}
                          {isAgent ? (
                            <span className="q-cell-agent-icon" aria-hidden="true">
                              🐛
                            </span>
                          ) : null}
                        </>
                      ) : null}
                    </div>
                  );
                })}
              </div>

              <div className="q-score-flashes" aria-live="polite">
                {scoreFlashes.map((flash) => (
                  <span
                    key={flash.id}
                    className={`q-score-flash ${
                      flash.positive ? "positive" : "negative"
                    }`}
                  >
                    {flash.text}
                  </span>
                ))}
              </div>
            </div>

            <div className="q-status-line">
              <span
                className={`q-reward-chip ${
                  lastReward >= 0 ? "positive" : "negative"
                }`}
              >
                Senaste beloning: {formatSigned(lastReward)}
              </span>
              <span
                className={`q-episode-chip ${episodeDone ? "done" : "running"}`}
              >
                {episodeDone ? "Episod avslutad" : "Episod aktiv"}
              </span>
            </div>

            <p className="q-status-text">{statusText}</p>

            {!canTrain ? (
              <p className="q-warning">
                Start och mal ar inte sammankopplade med gangbara rutor. Traning
                blir da mycket svag.
              </p>
            ) : null}
          </article>

          <article className="q-card">
            <div className="q-card-head">
              <h2>Q-tabell</h2>
              <p>En rad per tillstand utom S. Kolumner: hoger och ner.</p>
            </div>
            <div className="q-table-wrap">
              <table className="q-table">
                <thead>
                  <tr>
                    <th>Tillstand</th>
                    {ACTIONS.map((action) => (
                      <th key={`head-${action}`}>{ACTION_LABELS[action]}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {qStateKeys.map((stateKey) => (
                    <tr key={`row-${stateKey}`}>
                      <td>{labelMap[stateKey]}</td>
                      {ACTIONS.map((action) => {
                        const isValidAction =
                          validActionsByState[stateKey]?.includes(action) ?? false;
                        return (
                          <td
                            key={`${stateKey}-${action}`}
                            className={isValidAction ? "" : "q-table-invalid"}
                          >
                            {isValidAction
                              ? (qTable[stateKey]?.[action] ?? 0).toFixed(2)
                              : "X"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="q-calc-box">
              <h3>Aktuell berakning</h3>
              {calculation ? (
                <>
                  <p>
                    Q({calculation.stateLabel}, {ACTION_SYMBOLS[calculation.action]}) ={" "}
                    {calculation.oldQ.toFixed(3)} + {calculation.alpha.toFixed(2)} * (
                    {calculation.reward.toFixed(2)} + {calculation.gamma.toFixed(2)} *{" "}
                    {calculation.maxNextQ.toFixed(3)} - {calculation.oldQ.toFixed(3)})
                  </p>
                  <p>
                    Target = {calculation.target.toFixed(3)} och nytt Q-varde ={" "}
                    <strong>{calculation.newQ.toFixed(3)}</strong>
                  </p>
                </>
              ) : (
                <p>Ingen uppdatering an. Klicka pa "Stega ett drag".</p>
              )}
            </div>
          </article>
        </div>
      </section>

      <section className="section">
        <div className="q-controls-grid">
          <article className="q-controls-card">
            <h2>Stegvis traning</h2>
            <div className="q-btn-row">
              <button
                className="q-btn q-btn-primary"
                onClick={handleManualStep}
                disabled={isAutoRunning || !canTrain || editMode}
              >
                {episodeDone ? "Ny episod" : "Stega ett drag"}
              </button>
            </div>
          </article>

          <article className="q-controls-card">
            <h2>Parametrar</h2>
            <div className="q-param-grid">
              <label className="q-field">
                <span>alpha</span>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={alpha}
                  onChange={(event) =>
                    setAlpha(clamp01(Number.parseFloat(event.target.value) || 0))
                  }
                  disabled={isAutoRunning}
                />
              </label>
              <label className="q-field">
                <span>gamma</span>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={gamma}
                  onChange={(event) =>
                    setGamma(clamp01(Number.parseFloat(event.target.value) || 0))
                  }
                  disabled={isAutoRunning}
                />
              </label>
              <label className="q-field">
                <span>epsilon</span>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={epsilon}
                  onChange={(event) =>
                    setEpsilon(clamp01(Number.parseFloat(event.target.value) || 0))
                  }
                  disabled={isAutoRunning}
                />
              </label>
            </div>
          </article>

          <article className="q-controls-card">
            <h2>Automatisk traning (med animation)</h2>
            <div className="q-param-grid">
              <label className="q-field">
                <span>Hastighet</span>
                <select
                  value={animationDelay}
                  onChange={(event) =>
                    setAnimationDelay(Number.parseInt(event.target.value, 10))
                  }
                  disabled={isAutoRunning}
                >
                  <option value={700}>Lugn</option>
                  <option value={420}>Normal</option>
                  <option value={220}>Snabb</option>
                  <option value={130}>Turbo</option>
                </select>
              </label>
              <label className="q-field">
                <span>Episoder</span>
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={animatedEpisodes}
                  onChange={(event) =>
                    setAnimatedEpisodes(
                      Math.max(1, Number.parseInt(event.target.value, 10) || 1)
                    )
                  }
                  disabled={isAutoRunning}
                />
              </label>
            </div>
            <div className="q-btn-row">
              <button
                className="q-btn q-btn-primary"
                onClick={handleAnimatedButton}
                disabled={!canTrain || (isAutoRunning && autoMode !== "animated")}
              >
                {isAutoRunning && autoMode === "animated"
                  ? stopRequested
                    ? "Stoppar..."
                    : "Stoppa animation"
                  : "Starta animerad traning"}
              </button>
            </div>
          </article>

          <article className="q-controls-card">
            <h2>Snabbtraning (utan animation)</h2>
            <div className="q-param-grid">
              <label className="q-field">
                <span>Episoder</span>
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={fastEpisodes}
                  onChange={(event) =>
                    setFastEpisodes(
                      Math.max(1, Number.parseInt(event.target.value, 10) || 1)
                    )
                  }
                  disabled={isAutoRunning}
                />
              </label>
            </div>
            <div className="q-btn-row">
              <button
                className="q-btn q-btn-primary"
                onClick={handleFastTraining}
                disabled={isAutoRunning || !canTrain || editMode}
              >
                Kor snabbtraning
              </button>
            </div>
          </article>
        </div>
      </section>

      {editMode ? (
        <section className="section q-editor-section">
          <div className="q-card-head">
            <h2>Editor (10x10)</h2>
            <p>
              Svart bakgrund. Placera vita gangbara rutor med vansterklick.
              Hogerklick pa vit ruta vaxlar neutral till apple (+1) till fagel (-2).
            </p>
          </div>

          <div className="q-editor-board-wrap">
            <div className="q-editor-board">
              {editorCells.map((key) => {
                const isWalkable = walkableSet.has(key);
                const isStart = key === startKey;
                const isGoal = key === goalKey;
                const item = normalizedItems[key] ?? "none";

                return (
                  <div
                    key={`editor-${key}`}
                    className={`q-editor-cell ${
                      isWalkable ? "walkable" : "blocked"
                    } ${isStart ? "start" : ""} ${isGoal ? "goal" : ""} ${
                      dragToken ? "drag-active" : ""
                    }`}
                    onMouseDown={(event) => handleEditorMouseDown(event, key)}
                    onMouseUp={(event) => handleEditorMouseUp(event, key)}
                    onContextMenu={(event) => handleEditorRightClick(event, key)}
                  >
                    {isWalkable ? (
                      <>
                        <span className="q-editor-label">{labelMap[key] ?? ""}</span>
                        {!isStart && !isGoal && item === "apple" ? (
                          <span className="q-editor-item" aria-hidden="true">
                            🍎
                          </span>
                        ) : null}
                        {!isStart && !isGoal && item === "bird" ? (
                          <span className="q-editor-item" aria-hidden="true">
                            🐦
                          </span>
                        ) : null}
                      </>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </div>

          <p className="q-editor-info">{editorInfo}</p>
          <p className="q-editor-meta">
            Gangbara rutor: {stateCount}/{MAX_WALKABLE_CELLS}. Start A och mal S
            raknas alltid med.
          </p>
        </section>
      ) : null}
    </div>
  );
}

export default QLearningPage;
