import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent
} from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import "./qLearning.css";

import {
  ALL_ACTIONS, ACTION_LABELS, ACTION_SYMBOLS,
  EDITOR_SIDE, MAX_WALKABLE_CELLS, MAX_APPLES, MAX_STEPS_PER_EPISODE,
  DEFAULT_START, DEFAULT_GOAL, DEFAULT_WALKABLE, DEFAULT_ITEMS, Q_ENVIRONMENTS,
  sortKeys, createZeroQTable, buildLabelMap, normalizeItems, deriveAppleKeys,
  createAppleIndexByKey, normalizeAppleMask, buildCollectedApplesMap, keyFromXY,
  getValidActions, shouldExpandAppleStates, encodeStateKey, formatStateLabel,
  parseKey, hasPathToGoal, alignQTable, cloneQTable, performStep, clamp01,
  pickAction, makeZeroRow, previewStep, roundQValue, nextEpsilon, traceGreedyPath,
  type Action, type TileItem, type QTable, type QStateDefinition,
  type QCalculation, type StepOutcome, type EpsilonSchedule
} from "./qLearning";

interface PendingStep {
  stateKey: string;
  action: Action;
}

const wait = (ms: number): Promise<void> =>
  new Promise((resolve) => window.setTimeout(resolve, ms));

function renderMath(expression: string, displayMode = true): { __html: string } {
  return { __html: katex.renderToString(expression, { throwOnError: false, displayMode }) };
}

function mathNumber(value: number): string {
  // Remove binary arithmetic tails without hiding small nonzero values.
  return Number(value.toPrecision(15)).toString()
    .replace(/e([+-]?\d+)$/, String.raw` \cdot 10^{$1}`);
}

function calculationFormula(calculation: QCalculation | null): string {
  // A shared array gives each number the same column as its symbolic term.
  const theory = [
    "Q(s,a)", "=", "Q(s,a)", "+", String.raw`\alpha`, String.raw`\cdot`,
    String.raw`\big(`, "r", "+", String.raw`\gamma`, String.raw`\cdot`,
    String.raw`\max_{a'} Q(s',a')`, "-", "Q(s,a)", String.raw`\big)`, ""
  ];
  const rows = [theory.join(" & ")];
  if (calculation) {
    const oldQ = calculation.oldQ.toFixed(2);
    const oldTerm = calculation.oldQ < 0 ? `(${oldQ})` : oldQ;
    const nextQ = calculation.maxNextQ.toFixed(2);
    const nextTerm = calculation.maxNextQ < 0 ? `(${nextQ})` : nextQ;
    const rounded = Number(calculation.rawNewQ.toPrecision(15)) !== calculation.newQ;
    const result = `${rounded ? String.raw`\approx` : "="} ${calculation.newQ.toFixed(2)}`;
    const state = /^[A-Z]\d*$/.test(calculation.stateLabel)
      ? calculation.stateLabel
      : String.raw`\text{${calculation.stateLabel}}`;
    rows.push([
      String.raw`Q(${state},\mathrm{${ACTION_SYMBOLS[calculation.action]}})`,
      "=", oldTerm, "+", mathNumber(calculation.alpha), String.raw`\cdot`,
      String.raw`\big(`, mathNumber(calculation.reward), "+", mathNumber(calculation.gamma),
      String.raw`\cdot`, nextTerm, "-", oldTerm, String.raw`\big)`, result
    ].join(" & "));
  }
  return String.raw`\def\arraystretch{1.65}\begin{darray}{${"c".repeat(theory.length)}}${rows.join(String.raw` \\ `)}\end{darray}`;
}

function cellVisualItem(
  key: string,
  startKey: string,
  goalKey: string,
  cellItems: Record<string, TileItem>,
  collectedApples: Record<string, boolean>
): string {
  if (key === goalKey) return "🎯";
  if (key === startKey) return "";
  const item = cellItems[key] ?? "none";
  if (item === "apple" && collectedApples[key]) return "";
  if (item === "apple") return "\u{1F34E}";
  if (item === "bird") return "\u{1F426}";
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

  const [qTable, setQTable] = useState<QTable>(() =>
    createZeroQTable(
      DEFAULT_WALKABLE.filter(
        (key) => key !== DEFAULT_GOAL && DEFAULT_ITEMS[key] !== "bird"
      )
    )
  );
  const [agentKey, setAgentKey] = useState(startKey);
  const [agentAppleMask, setAgentAppleMask] = useState(0);
  const [episodeDone, setEpisodeDone] = useState(false);
  const [episodeScore, setEpisodeScore] = useState(0);
  const [stepCount, setStepCount] = useState(0);
  const [episodeIndex, setEpisodeIndex] = useState(1);
  const [statusText, setStatusText] = useState(
    "Miljön är redo.\nKlicka på Stega ett drag och räkna själv."
  );
  const [calculation, setCalculation] = useState<QCalculation | null>(null);
  const [pendingStep, setPendingStep] = useState<PendingStep | null>(null);
  const [hintShown, setHintShown] = useState(false);
  const [selectedEnvironmentId, setSelectedEnvironmentId] = useState(Q_ENVIRONMENTS[0].id);

  const [alpha, setAlpha] = useState(0.5);
  const [gamma, setGamma] = useState(0.9);
  const [epsilon, setEpsilon] = useState(0.9);
  const [epsilonStart, setEpsilonStart] = useState(0.9);
  const [stepCost, setStepCost] = useState(Q_ENVIRONMENTS[0].stepCost);
  const [epsilonSchedule, setEpsilonSchedule] = useState<EpsilonSchedule>("constant");
  const [epsilonChange, setEpsilonChange] = useState(0.98);
  const [epsilonMinimum, setEpsilonMinimum] = useState(0.05);
  const [showBestPath, setShowBestPath] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const [isAutoRunning, setIsAutoRunning] = useState(false);
  const [animationDelay, setAnimationDelay] = useState(420);
  const [trainingEpisodes, setTrainingEpisodes] = useState(8);
  const [showAnimations, setShowAnimations] = useState(true);
  const [stopRequested, setStopRequested] = useState(false);
  const stopRequestedRef = useRef(false);

  const [editMode, setEditMode] = useState(false);
  const [enabledActions, setEnabledActions] = useState<Action[]>(Q_ENVIRONMENTS[0].enabledActions);
  const [dragToken, setDragToken] = useState<"start" | "goal" | null>(null);
  const [editorInfo, setEditorInfo] = useState("");

  const walkableSet = useMemo(() => new Set(walkableKeys), [walkableKeys]);
  const normalizedItems = useMemo(
    () => normalizeItems(cellItems, walkableSet, startKey, goalKey, labelMap),
    [cellItems, goalKey, labelMap, startKey, walkableSet]
  );
  const appleKeys = useMemo(
    () => deriveAppleKeys(normalizedItems, labelMap),
    [labelMap, normalizedItems]
  );
  const appleIndexByKey = useMemo(
    () => createAppleIndexByKey(appleKeys),
    [appleKeys]
  );

  const normalizedAgentAppleMask = useMemo(
    () =>
      normalizeAppleMask(
        agentAppleMask,
        agentKey,
        appleIndexByKey,
        appleKeys.length
      ),
    [agentAppleMask, agentKey, appleIndexByKey, appleKeys.length]
  );
  const collectedApples = useMemo(
    () => buildCollectedApplesMap(appleKeys, normalizedAgentAppleMask),
    [appleKeys, normalizedAgentAppleMask]
  );

  const editorCells = useMemo(() => {
    const cells: string[] = [];
    for (let y = 0; y < EDITOR_SIDE; y += 1) {
      for (let x = 0; x < EDITOR_SIDE; x += 1) {
        cells.push(keyFromXY(x, y));
      }
    }
    return cells;
  }, []);

  const validActionsByState = useMemo(() => {
    const map: Record<string, Action[]> = {};
    walkableKeys.forEach((key) => {
      map[key] = getValidActions(key, walkableSet, enabledActions);
    });
    return map;
  }, [enabledActions, walkableKeys, walkableSet]);
  const expandedStateSpace = useMemo(
    () =>
      shouldExpandAppleStates({
        walkableKeys,
        walkableSet,
        enabledActions,
        startKey,
        goalKey,
        appleKeys,
        cellItems: normalizedItems
      }),
    [
      appleKeys,
      enabledActions,
      goalKey,
      normalizedItems,
      startKey,
      walkableKeys,
      walkableSet
    ]
  );

  const qStates = useMemo(() => {
    const positions = walkableKeys
      .filter((key) => key !== goalKey && normalizedItems[key] !== "bird")
      .sort((left, right) => {
        const leftLabel = labelMap[left] ?? "";
        const rightLabel = labelMap[right] ?? "";
        return leftLabel.localeCompare(rightLabel);
      });

    const states: QStateDefinition[] = [];
    const combinations = expandedStateSpace ? 1 << appleKeys.length : 1;
    positions.forEach((positionKey) => {
      const forcedAppleIndex = appleIndexByKey[positionKey];

      for (let mask = 0; mask < combinations; mask += 1) {
        if (
          expandedStateSpace &&
          forcedAppleIndex !== undefined &&
          (mask & (1 << forcedAppleIndex)) === 0
        ) {
          continue;
        }

        const normalizedMask = normalizeAppleMask(
          mask,
          positionKey,
          appleIndexByKey,
          appleKeys.length
        );
        states.push({
          key: encodeStateKey(positionKey, normalizedMask, expandedStateSpace),
          positionKey,
          appleMask: normalizedMask,
          label: formatStateLabel(
            positionKey,
            normalizedMask,
            expandedStateSpace,
            appleKeys,
            labelMap
          )
        });
      }
    });

    return states;
  }, [
    appleIndexByKey,
    appleKeys,
    expandedStateSpace,
    goalKey,
    labelMap,
    normalizedItems,
    walkableKeys
  ]);

  const qStateKeys = useMemo(() => qStates.map((state) => state.key), [qStates]);
  const qStateLabelsByKey = useMemo(() => {
    const map: Record<string, string> = {};
    qStates.forEach((state) => {
      map[state.key] = state.label;
    });
    return map;
  }, [qStates]);
  const validActionsByQState = useMemo(() => {
    const map: Record<string, Action[]> = {};
    qStates.forEach((state) => {
      map[state.key] = validActionsByState[state.positionKey] ?? [];
    });
    return map;
  }, [qStates, validActionsByState]);

  const currentQStateKey = useMemo(
    () => encodeStateKey(agentKey, normalizedAgentAppleMask, expandedStateSpace),
    [agentKey, expandedStateSpace, normalizedAgentAppleMask]
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
    if (editMode) return editorCells;

    const cells: string[] = [];
    for (let y = gridBounds.minY; y <= gridBounds.maxY; y += 1) {
      for (let x = gridBounds.minX; x <= gridBounds.maxX; x += 1) {
        cells.push(keyFromXY(x, y));
      }
    }
    return cells;
  }, [
    editMode,
    editorCells,
    gridBounds.maxX,
    gridBounds.maxY,
    gridBounds.minX,
    gridBounds.minY
  ]);

  const trainingColumns = editMode
    ? EDITOR_SIDE
    : gridBounds.maxX - gridBounds.minX + 1;
  const tableActions = enabledActions;
  const hasGoalPath = useMemo(
    () => hasPathToGoal(walkableSet, startKey, goalKey, enabledActions),
    [enabledActions, goalKey, startKey, walkableSet]
  );

  useEffect(() => {
    setQTable((previous) => alignQTable(previous, qStateKeys));
  }, [qStateKeys]);

  useEffect(() => {
    if (normalizedAgentAppleMask === agentAppleMask) return;
    setAgentAppleMask(normalizedAgentAppleMask);
  }, [agentAppleMask, normalizedAgentAppleMask]);

  useEffect(() => {
    if (!editMode) return;
    setAgentKey(startKey);
    setAgentAppleMask(0);
  }, [editMode, startKey]);

  useEffect(() => {
    const handleMouseUp = () => {
      setDragToken(null);
    };
    window.addEventListener("mouseup", handleMouseUp);
    return () => window.removeEventListener("mouseup", handleMouseUp);
  }, []);

  const syncOutcomeToUi = useCallback(
    (outcome: StepOutcome, currentEpisode: number) => {
      setQTable(outcome.qTable);
      setAgentKey(outcome.agentKey);
      setAgentAppleMask(outcome.agentAppleMask);
      setEpisodeDone(outcome.done);
      setEpisodeScore(outcome.episodeScore);
      setStepCount(outcome.stepCount);
      setCalculation(outcome.calculation);
      setEpisodeIndex(currentEpisode);
      setStatusText(
        `${outcome.calculation.stateLabel} → ${ACTION_LABELS[outcome.action]} → ${outcome.calculation.nextStateLabel}.`
      );
    },
    []
  );

  const resetEpisode = useCallback(
    (nextEpisodeIndex?: number, message?: string) => {
      setAgentKey(startKey);
      setAgentAppleMask(0);
      setEpisodeDone(false);
      setEpisodeScore(0);
      setStepCount(0);
      setCalculation(null);
      setPendingStep(null);
      setHintShown(false);
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
      setEpsilon(epsilonStart);
      setQTable(createZeroQTable(qStateKeys));
      setIsAutoRunning(false);
      setStopRequested(false);
      setEpisodeIndex(1);
      setAgentKey(startKey);
      setAgentAppleMask(0);
      setEpisodeDone(false);
      setEpisodeScore(0);
      setStepCount(0);
      setCalculation(null);
      setPendingStep(null);
      setHintShown(false);
      setStatusText(message);
    },
    [epsilonStart, qStateKeys, startKey]
  );

  const handleManualStep = () => {
    if (isAutoRunning || editMode || !canTrain) return;
    if (episodeDone) {
      resetEpisode(episodeIndex + 1, `Ny episod ${episodeIndex + 1} startad.`);
      return;
    }
    if (!pendingStep) {
      const action = pickAction(qTable[currentQStateKey] ?? makeZeroRow(), epsilon,
        validActionsByState[agentKey] ?? []);
      if (!action) {
        setStatusText("Inga giltiga drag finns från aktuellt tillstånd.");
        return;
      }
      setPendingStep({ stateKey: currentQStateKey, action });
      setHintShown(false);
      setCalculation(null);
      setStatusText(`Masken kommer att gå ${ACTION_LABELS[action].toLowerCase()}.\nBeräkna det nya Q-värdet.`);
      return;
    }
    // Complete the announced action without sampling another random move.
    if (pendingStep.stateKey !== currentQStateKey) return;
    const outcome = performStep({
      qTable, agentKey, agentAppleMask: normalizedAgentAppleMask,
      episodeScore, stepCount, walkableSet, goalKey,
      cellItems: normalizedItems, labelMap, appleKeys, appleIndexByKey,
      expandedStateSpace, validActionsByState, alpha, gamma, epsilon, stepCost,
      action: pendingStep.action
    });
    setPendingStep(null);
    setHintShown(false);
    if (outcome) {
      syncOutcomeToUi(outcome, episodeIndex);
    }
  };

  const handleEnvironmentChange = (id: string) => {
    if (isAutoRunning) return;
    const environment = Q_ENVIRONMENTS.find((item) => item.id === id);
    if (!environment) return;
    setSelectedEnvironmentId(id);
    setWalkableKeys(sortKeys(environment.walkableKeys));
    setStartKey(environment.startKey);
    setGoalKey(environment.goalKey);
    setCellItems({ ...environment.cellItems });
    setEnabledActions([...environment.enabledActions]);
    setStepCost(environment.stepCost);
    setEpsilon(epsilonStart);
    setEpsilonSchedule(environment.automaticOnly ? "multiply" : "constant");
    if (environment.automaticOnly) {
      setAdvancedOpen(true);
      setTrainingEpisodes(200);
      setAnimationDelay(130);
      setEpsilon(0.9);
      setEpsilonStart(0.9);
      setEpsilonChange(0.98);
      setEpsilonMinimum(0.05);
    }
    setQTable(createZeroQTable(environment.walkableKeys.filter(
      (key) => key !== environment.goalKey && environment.cellItems[key] !== "bird"
    )));

    setAgentKey(environment.startKey);
    setAgentAppleMask(0);
    setEpisodeIndex(1);
    setEpisodeDone(false);
    setEpisodeScore(0);
    setStepCount(0);
    setCalculation(null);
    setPendingStep(null);
    setHintShown(false);
    setEditMode(false);
    setDragToken(null);
    setEditorInfo("");
    setStatusText(environment.automaticOnly
      ? "Miljön är redo. Starta automatisk träning i panelen till höger."
      : "Miljön är redo.\nKlicka på Stega ett drag och räkna själv.");
  };

  const handleResetQ = () => {
    if (isAutoRunning) return;
    resetQForCurrentGrid("Q-tabellen nollställdes och episod 1 är redo.");
  };

  const runTraining = useCallback(async () => {
    if (isAutoRunning || editMode || pendingStep) return;

    const episodeCount = Math.max(1, Math.floor(trainingEpisodes || 1));
    setIsAutoRunning(true);
    setStopRequested(false);
    stopRequestedRef.current = false;
    setStatusText(`Träning startad (${episodeCount} episoder${showAnimations ? " med" : " utan"} animationer).`);

    let localQ = cloneQTable(qTable);
    let localEpsilon = epsilonStart;
    setEpsilon(localEpsilon);
    const firstEpisode = episodeIndex;
    let completedEpisodes = 0;
    let stopped = false;

    for (let episodeNumber = 0; episodeNumber < episodeCount; episodeNumber += 1) {
      if (stopRequestedRef.current) {
        stopped = true;
        break;
      }

      const activeEpisode = firstEpisode + episodeNumber;
      let localAgent = startKey;
      let localMask = normalizeAppleMask(
        0,
        startKey,
        appleIndexByKey,
        appleKeys.length
      );
      let localScore = 0;
      let localStepCount = 0;
      let done = false;
      let blocked = false;

      if (showAnimations) {
        setEpisodeIndex(activeEpisode);
        setAgentKey(startKey);
        setAgentAppleMask(localMask);
        setEpisodeDone(false);
        setEpisodeScore(0);
        setStepCount(0);
        setCalculation(null);
        setPendingStep(null);
        setHintShown(false);
      }

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
          agentAppleMask: localMask,
          episodeScore: localScore,
          stepCount: localStepCount,
          walkableSet,
          goalKey,
          cellItems: normalizedItems,
          labelMap,
          appleKeys,
          appleIndexByKey,
          expandedStateSpace,
          validActionsByState,
          alpha,
          gamma,
          epsilon: localEpsilon,
          stepCost
        });

        if (!outcome) {
          blocked = true;
          setStatusText(
            `Episod ${activeEpisode} avbruten: inga giltiga drag från ${formatStateLabel(
              localAgent,
              localMask,
              expandedStateSpace,
              appleKeys,
              labelMap
            )}.`
          );
          break;
        }

        localQ = outcome.qTable;
        localAgent = outcome.agentKey;
        localMask = outcome.agentAppleMask;
        localScore = outcome.episodeScore;
        localStepCount = outcome.stepCount;
        done = outcome.done;

        if (showAnimations) {
          syncOutcomeToUi(outcome, activeEpisode);
          await wait(animationDelay);
        }

        if (done) {
          break;
        }
      }

      if (stopped) break;
      completedEpisodes += 1;
      localEpsilon = nextEpsilon(localEpsilon, epsilonSchedule, epsilonChange, epsilonMinimum);
      if (showAnimations) setEpsilon(localEpsilon);

      if (!done && !blocked && showAnimations) {
        setStatusText(
          `Episod ${activeEpisode} stoppades efter ${MAX_STEPS_PER_EPISODE} steg (ingen målträff).`
        );
      }

      // Yield between batches so training without animations can also be stopped.
      if (!showAnimations && completedEpisodes % 25 === 0) {
        setQTable(localQ);
        setEpisodeIndex(firstEpisode + completedEpisodes);
        setEpsilon(localEpsilon);
        await wait(0);
      }
    }

    const nextEpisode = firstEpisode + completedEpisodes;
    setQTable(localQ);
    setEpsilon(localEpsilon);

    if (stopped) {
      const message = `Träningen stoppades efter ${completedEpisodes} hela episoder.`;
      if (showAnimations) {
        setStatusText(message);
      } else {
        resetEpisode(nextEpisode, message);
      }
    } else {
      resetEpisode(nextEpisode, `Träning klar: ${episodeCount} episoder genomförda ${showAnimations ? "med" : "utan"} animationer.`);
    }

    setIsAutoRunning(false);
    setStopRequested(false);
    stopRequestedRef.current = false;
  }, [
    alpha,
    appleIndexByKey,
    appleKeys,
    animationDelay,
    editMode,
    episodeIndex,
    epsilonStart,
    epsilonChange,
    epsilonMinimum,
    epsilonSchedule,
    expandedStateSpace,
    gamma,
    goalKey,
    isAutoRunning,
    pendingStep,
    labelMap,
    normalizedItems,
    qTable,
    resetEpisode,
    showAnimations,
    startKey,
    stepCost,
    syncOutcomeToUi,
    trainingEpisodes,
    validActionsByState,
    walkableSet
  ]);

  const handleTrainingButton = () => {
    if (!isAutoRunning) {
      void runTraining();
      return;
    }

    stopRequestedRef.current = true;
    setStopRequested(true);
    setStatusText("Stoppsignal skickad. Träningen avslutas...");
  };

  const toggleEditMode = () => {
    if (isAutoRunning || pendingStep) return;
    if (!editMode) {
      setEditMode(true);
      setSelectedEnvironmentId("custom");
      setEditorInfo(
        "Redigeringsläge aktivt i miljön (4x4). Vänsterklick: lägg till/ta bort ruta. Högerklick: neutral -> äpple -> fågel. Dra A/S för att flytta start/mål."
      );
      resetQForCurrentGrid("Q-tabellen nollställdes när redigeringsläget startade.");
      return;
    }

    setEditMode(false);
    setEditorInfo("");
    resetQForCurrentGrid("Ny Q-tabell initierad när redigeringsläget avslutades.");
  };

  const handleActionToggle = (action: Action) => {
    if (!editMode || isAutoRunning) return;

    setEnabledActions((previous) => {
      if (previous.includes(action)) {
        return previous.filter((item) => item !== action);
      }

      return ALL_ACTIONS.filter(
        (candidate) => candidate === action || previous.includes(candidate)
      );
    });
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
      setEditorInfo("Max 16 rutor tillåtet (inklusive A och S).");
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
        setEditorInfo("Max 16 rutor tillåtet.");
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
        setAgentKey(targetKey);
        setAgentAppleMask(0);
      } else {
        setGoalKey(targetKey);
      }
      setEditorInfo(`${marker === "start" ? "Start" : "Mål"} flyttad.`);
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

    if (current === "none" && nextItem === "apple" && appleKeys.length >= MAX_APPLES) {
      setEditorInfo("Max 2 äpplen tillåts i Q-learning-läget.");
      return;
    }

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

  const stateCount = walkableKeys.length;
  const canTrain = hasGoalPath && stateCount >= 2 && tableActions.length > 0;
  const selectedActionsText =
    tableActions.length > 0
      ? tableActions.map((action) => ACTION_LABELS[action]).join(", ")
      : "inga actions valda";
  const selectedEnvironment = Q_ENVIRONMENTS.find((item) => item.id === selectedEnvironmentId);
  const bestPath = useMemo(() => showBestPath && !editMode ? traceGreedyPath({
    qTable, startKey, goalKey, walkableSet, validActionsByState,
    cellItems: normalizedItems, labelMap, appleKeys, appleIndexByKey,
    expandedStateSpace, stepCost
  }) : null, [showBestPath, editMode, qTable, startKey, goalKey, walkableSet,
    validActionsByState, normalizedItems, labelMap, appleKeys, appleIndexByKey,
    expandedStateSpace, stepCost]);
  const highlightedPath = bestPath?.steps.length ? bestPath : null;
  const pathSummary = bestPath
    ? bestPath.ending === "goal"
      ? `Q-tabellens väg når målet på ${bestPath.steps.length} steg med ${bestPath.score.toFixed(2)} poäng.`
      : bestPath.ending === "bird"
        ? `Q-tabellens väg leder just nu till en fågel efter ${bestPath.steps.length} steg.`
        : bestPath.ending === "loop"
          ? `Q-tabellens väg fastnar just nu i en slinga efter ${bestPath.steps.length} steg.`
          : "Q-tabellens väg saknar just nu en giltig fortsättning."
    : "";
  const bestPathCells = new Set(highlightedPath
    ? [startKey, ...highlightedPath.steps.map((step) => step.nextPositionKey)] : []);
  const bestPathActions = new Set(highlightedPath?.steps.map((step) => `${step.stateKey}|${step.action}`));
  const pathPoints = highlightedPath
    ? [startKey, ...highlightedPath.steps.map((step) => step.nextPositionKey)].map((key) => {
      const { x, y } = parseKey(key);
      return `${x - gridBounds.minX + 0.5},${y - gridBounds.minY + 0.5}`;
    }).join(" ") : "";
  const pendingPreview = pendingStep ? previewStep({
    agentKey, agentAppleMask: normalizedAgentAppleMask, action: pendingStep.action,
    goalKey, cellItems: normalizedItems, labelMap, appleKeys,
    appleIndexByKey, expandedStateSpace, stepCost
  }) : null;

  return (
    <div className="container page-flow q-learning-page">
      <section className="section">
        <h1 className="q-learning-title">Q-learning: masken och äppeljakten</h1>
        <p className="q-learning-lead">
          Hjälp masken ({"\u{1F41B}"}) att lära sig hitta till målet S.
          Räkna själv på papper och jämför med facit, eller följ automatisk träning.
        </p>

      </section>

      <section className="section q-learning-main-section">
        <div className="q-learning-main-grid">
          <article className="q-card">
            <div className="q-card-head">
              <h2>Miljö</h2>
              <p>
                Episod {episodeIndex} - steg {stepCount} - poäng{" "}
                <strong>{episodeScore.toFixed(1)}</strong>
              </p>
            </div>

            <label className="q-environment-picker">
              <span>Välj miljö</span>
              <select value={selectedEnvironmentId} disabled={isAutoRunning}
                onChange={(event) => handleEnvironmentChange(event.target.value)}>
                {Q_ENVIRONMENTS.map((environment) => (
                  <option key={environment.id} value={environment.id}>{environment.name}</option>
                ))}
                {selectedEnvironmentId === "custom" && <option value="custom">Egen miljö</option>}
              </select>
            </label>
            <p className="q-environment-description">
              {selectedEnvironment?.description ?? "Din egen miljö. Giltiga riktningar väljs i redigeringsläget."}
              {" "}Vid miljöbyte nollställs träningen.
            </p>

            <div className="q-training-wrap">
              <div
                className="q-training-grid"
                style={{
                  gridTemplateColumns: `repeat(${trainingColumns}, var(--q-cell-size))`
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
                  const itemReward = isGoal ? 2 : visualItem === "🐦" ? -2 : visualItem === "🍎" ? 1 : null;
                  const cellReward = itemReward === null ? null : roundQValue(itemReward + stepCost);

                  return (
                    <div
                      key={`training-${key}`}
                      aria-label={isWalkable ? `Ruta ${label}${isStart ? ", start" : ""}${isGoal ? ", mål" : ""}${normalizedItems[key] === "bird" ? ", fågel" : ""}${visualItem === "🍎" ? ", äpple" : ""}${cellReward !== null ? `, belöning ${cellReward > 0 ? "+" : ""}${cellReward}` : ""}${isAgent ? ", masken" : ""}${bestPathCells.has(key) ? ", väg enligt Q-tabellen" : ""}` : "Blockerad ruta"}
                      className={`q-cell ${
                        isWalkable ? "q-cell-walkable" : "q-cell-blocked"
                      } ${isStart ? "q-cell-start" : ""} ${
                        isGoal ? "q-cell-goal" : ""
                      } ${isAgent ? "q-cell-agent" : ""} ${
                        cellReward !== null ? "q-cell-has-reward" : ""
                      } ${
                        editMode ? "q-cell-editable" : ""
                      } ${dragToken ? "q-cell-drag-active" : ""} ${bestPathCells.has(key) ? "q-cell-best-path" : ""}`}
                      onMouseDown={(event) => handleEditorMouseDown(event, key)}
                      onMouseUp={(event) => handleEditorMouseUp(event, key)}
                      onContextMenu={(event) => handleEditorRightClick(event, key)}
                    >
                      {isWalkable ? (
                        <>
                          <span className="q-cell-label">{label}</span>
                          {visualItem ? (
                            <span className="q-cell-item" aria-hidden="true">
                              {visualItem}
                            </span>
                          ) : null}
                          {cellReward !== null && (
                            <span className={`q-cell-reward ${cellReward > 0 ? "positive" : cellReward < 0 ? "negative" : ""}`} aria-hidden="true">
                              {cellReward > 0 ? "+" : cellReward < 0 ? "−" : ""}{Math.abs(cellReward)}
                            </span>
                          )}
                          {isAgent ? (
                            <span className="q-cell-agent-icon" aria-hidden="true">
                              {"\u{1F41B}"}
                            </span>
                          ) : null}
                        </>
                      ) : null}
                    </div>
                  );
                })}
                {highlightedPath && (
                  <svg className="q-best-path-overlay" aria-hidden="true"
                    viewBox={`0 0 ${trainingColumns} ${Math.ceil(trainingCells.length / trainingColumns)}`}
                    preserveAspectRatio="none">
                    <polyline className="q-best-path-halo" points={pathPoints} />
                    <polyline className="q-best-path-line" points={pathPoints} />
                  </svg>
                )}
              </div>

            </div>

            <p className="q-environment-description">Mål och fågel avslutar episoden.</p>
            {stepCost !== 0 && <p className="q-environment-description">
              Varje drag kostar {Math.abs(stepCost).toFixed(2)} poäng, även till äpple, fågel och mål.
              Belöningarna i rutorna inkluderar stegkostnaden.
            </p>}
            {highlightedPath && <p className="q-path-note">
              {pathSummary}
            </p>}

            {editMode ? (
              <div className="q-edit-panel">
                <p className="q-edit-help">
                  Redigera direkt i miljön (4x4): vänsterklick lägger till/tar bort
                  gångbar ruta, högerklick växlar neutral till äpple till fågel, och
                  dra A/S för att flytta start/mål.
                </p>
                <div className="q-action-picker" role="group" aria-label="Tillåtna actions">
                  {ALL_ACTIONS.map((action) => (
                    <label key={`allow-${action}`} className="q-action-option">
                      <input
                        type="checkbox"
                        checked={enabledActions.includes(action)}
                        onChange={() => handleActionToggle(action)}
                        disabled={isAutoRunning}
                      />
                      <span>{ACTION_LABELS[action]}</span>
                    </label>
                  ))}
                </div>
                <p className="q-editor-info">{editorInfo}</p>
                <p className="q-editor-meta">
                  Gångbara rutor: {stateCount}/{MAX_WALKABLE_CELLS}. Start A och mål S
                  räknas alltid med.
                </p>
              </div>
            ) : null}

            <div className="q-step-panel">
              <h3 className="q-step-heading">{selectedEnvironment?.automaticOnly ? "Träningsstatus" : "Räkna ett steg i taget"}</h3>
              <p className="q-step-prompt" aria-live="polite">{statusText}</p>
              {!selectedEnvironment?.automaticOnly && <div className="q-step-actions">
                <button className="q-btn q-btn-primary" onClick={handleManualStep}
                  disabled={isAutoRunning || !canTrain || editMode}>
                  {episodeDone ? "Ny episod" : pendingStep ? "Fullfölj steget" : "Stega ett drag"}
                </button>
                {pendingStep && <button className="q-btn" onClick={() => setHintShown(!hintShown)}
                  aria-expanded={hintShown} aria-controls="q-step-hint">
                  {hintShown ? "Dölj ledtråd" : "Visa ledtråd"}
                </button>}
              </div>}
              {hintShown && pendingStep && pendingPreview && (
                <p className="q-hint" id="q-step-hint">
                  Läs det gamla värdet på rad <strong>{qStateLabelsByKey[pendingStep.stateKey]}</strong>,
                  kolumn <strong>{ACTION_LABELS[pendingStep.action]}</strong>.
                  {pendingPreview.terminal
                    ? " Draget går till mål eller fågel och avslutar episoden. Framtidsvärdet är därför 0; någon nästa tabellrad behövs inte."
                    : <> Leta sedan efter det största giltiga Q-värdet på rad <strong>{pendingPreview.nextStateLabel}</strong>. Rutor med X ingår inte i maxvärdet.</>}
                </p>
              )}
            </div>

            {!canTrain ? (
              <p className="q-warning">
                {tableActions.length === 0
                  ? "Välj minst en tillåten action i redigeringsläget för att kunna träna."
                  : "Start och mål är inte sammankopplade med gångbara rutor utifrån valda actions."}
              </p>
            ) : null}
            <div className="q-parameters">
              <h3>Parametrar</h3>
              <div className="q-param-grid q-training-fields">
                <label className="q-field">
                  <span>α (alfa)</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={alpha}
                    onChange={(event) =>
                      setAlpha(clamp01(Number.parseFloat(event.target.value) || 0))
                    }
                    disabled={isAutoRunning || !!pendingStep}
                  />
                </label>
                <label className="q-field">
                  <span>γ (gamma)</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={gamma}
                    onChange={(event) =>
                      setGamma(clamp01(Number.parseFloat(event.target.value) || 0))
                    }
                    disabled={isAutoRunning || !!pendingStep}
                  />
                </label>
              </div>
              <label className="q-field q-step-cost-field">
                <span>Stegkostnad</span>
                <input type="number" max={0} step={0.1} value={stepCost}
                  disabled={isAutoRunning || !!pendingStep}
                  onChange={(event) => {
                    const cost = roundQValue(Math.min(0, Number.parseFloat(event.target.value) || 0));
                    if (cost === stepCost) return;
                    setStepCost(cost);
                    resetQForCurrentGrid("Stegkostnaden ändrades. Q-tabellen nollställdes.");
                  }} />
              </label>
              <p className="q-environment-description">0 ger ingen stegkostnad. När kostnaden ändras nollställs träningen.</p>
            </div>
          </article>

          <div className="q-learning-side-panels">
            <article className="q-card">
              <div className="q-card-head q-table-head">
                <h2>Q-tabell</h2>
                <button className="q-btn q-reset-button" onClick={handleResetQ} disabled={isAutoRunning}>
                  Nollställ Q-tabell
                </button>
                <p>
                  En rad per tillstånd utom S och fågelrutor. Kolumner:{" "}
                  {selectedActionsText}.
                </p>
              </div>
              <div className="q-table-wrap">
                <table className={`q-table ${showBestPath ? "q-table-highlighting-path" : ""}`}>
                  <thead>
                    <tr>
                      <th>Tillstånd</th>
                      {tableActions.map((action) => (
                        <th key={`head-${action}`}>{ACTION_LABELS[action]}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {qStateKeys.map((stateKey) => (
                      <tr key={`row-${stateKey}`}>
                        <td>{qStateLabelsByKey[stateKey]}</td>
                        {tableActions.map((action) => {
                          const isValidAction =
                            validActionsByQState[stateKey]?.includes(action) ?? false;
                          const isUpdated = isValidAction && calculation?.stateKey === stateKey && calculation.action === action;
                          const isBestPath = isValidAction && bestPathActions.has(`${stateKey}|${action}`);
                          const cellDescription = [
                            isBestPath ? `Väg enligt Q-tabellen: ${(qTable[stateKey]?.[action] ?? 0).toFixed(2)}` : "",
                            isUpdated ? `Senast uppdaterat: ${calculation.newQ.toFixed(2)}, tidigare ${calculation.oldQ.toFixed(2)}` : ""
                          ].filter(Boolean).join(". ");
                          return (
                            <td
                              key={`${stateKey}-${action}`}
                              className={`${!isValidAction ? "q-table-invalid" : isUpdated ? "q-table-updated" : ""} ${isBestPath ? "q-table-best-path" : ""}`}
                              aria-label={cellDescription || undefined}
                            >
                              {isValidAction ? <span className="q-cell-values">
                                <span>{(qTable[stateKey]?.[action] ?? 0).toFixed(2)}</span>
                                {isUpdated && <s className="q-old-value" aria-hidden="true">{calculation.oldQ.toFixed(2)}</s>}
                              </span> : "X"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="q-rounding-note">Varje nytt Q-värde avrundas till två decimaler. Använd det avrundade värdet i nästa beräkning. X betyder att draget inte är giltigt.</p>
              {expandedStateSpace && <p className="q-rounding-note">
                Samma ruta har olika tillstånd beroende på om äpplet finns kvar (🍎) eller är uppätet (❌).
              </p>}
              {highlightedPath && <p className="q-path-note">Röda ramar visar dragen när man följer Q-tabellens högsta giltiga värden.</p>}

              {!selectedEnvironment?.automaticOnly && <div className="q-calc-box">
                <h3>Q-formeln och facit</h3>
                <div
                  className="q-calc-formula"
                  dangerouslySetInnerHTML={renderMath(calculationFormula(calculation))}
                />
              </div>}
            </article>

            <details className="q-advanced" open={advancedOpen}
              onToggle={(event) => setAdvancedOpen(event.currentTarget.open)}>
              <summary>Fler möjligheter</summary>
              <div className="q-controls-grid">
                <article className="q-controls-card">
                  <h2>Automatisk träning</h2>
                  <div className="q-param-grid q-training-fields">
                    <label className="q-field">
                      <span>Hastighet</span>
                      <select
                        value={animationDelay}
                        onChange={(event) =>
                          setAnimationDelay(Number.parseInt(event.target.value, 10))
                        }
                        disabled={isAutoRunning || !!pendingStep || !showAnimations}
                        aria-describedby={!showAnimations ? "q-training-speed-note" : undefined}
                      >
                        <option value={700}>Lugn</option>
                        <option value={420}>Normal</option>
                        <option value={220}>Snabb</option>
                        <option value={130}>Turbo</option>
                      </select>
                    </label>
                    <label className="q-field">
                      <span>Antal episoder</span>
                      <input
                        type="number"
                        min={1}
                        step={1}
                        value={trainingEpisodes}
                        onChange={(event) =>
                          setTrainingEpisodes(
                            Math.max(1, Number.parseInt(event.target.value, 10) || 1)
                          )
                        }
                        disabled={isAutoRunning || !!pendingStep}
                      />
                    </label>
                  </div>
                  <fieldset className="q-epsilon-settings" disabled={isAutoRunning || !!pendingStep}>
                    <legend>Epsilon under träningen</legend>
                    <label className="q-field">
                      <span>Förändring av epsilon</span>
                      <select value={epsilonSchedule} onChange={(event) => {
                        const schedule = event.target.value as EpsilonSchedule;
                        setEpsilonSchedule(schedule);
                        setEpsilonChange(schedule === "subtract" ? 0.01 : 0.98);
                      }}>
                        <option value="constant">Oförändrat</option>
                        <option value="multiply">Minska med faktor per episod</option>
                        <option value="subtract">Minska med fast belopp per episod</option>
                      </select>
                    </label>
                    <div className={`q-param-grid ${epsilonSchedule === "constant" ? "q-epsilon-constant" : ""}`}>
                      <label className="q-field">
                        <span>Startvärde för epsilon</span>
                        <input type="number" min={0} max={1} step={0.05} value={epsilonStart}
                          onChange={(event) => {
                            const value = clamp01(Number.parseFloat(event.target.value) || 0);
                            setEpsilonStart(value);
                            setEpsilon(value);
                          }} />
                      </label>
                      {epsilonSchedule !== "constant" && <>
                        <label className="q-field">
                          <span>{epsilonSchedule === "multiply" ? "Faktor per episod" : "Minskning per episod"}</span>
                          <input type="number" min={0} max={1} step={0.01} value={epsilonChange}
                            onChange={(event) => setEpsilonChange(clamp01(Number.parseFloat(event.target.value) || 0))} />
                        </label>
                        <label className="q-field">
                          <span>Lägsta epsilon</span>
                          <input type="number" min={0} max={1} step={0.01} value={epsilonMinimum}
                            onChange={(event) => setEpsilonMinimum(clamp01(Number.parseFloat(event.target.value) || 0))} />
                        </label>
                      </>}
                    </div>
                    <p className="q-environment-description">
                      Aktuellt ε: <strong>{epsilon.toFixed(3)}</strong>.
                      {epsilonSchedule === "constant"
                        ? " Samma värde används i varje episod."
                        : ` Efter varje hel episod ${epsilonSchedule === "multiply" ? `multipliceras ε med ${epsilonChange}` : `minskas ε med ${epsilonChange}`}, ner till ${epsilonMinimum}.`}
                      {" "}Varje träningsomgång börjar med startvärdet ovan.
                    </p>
                  </fieldset>
                  <label className="q-action-option">
                    <input
                      type="checkbox"
                      checked={showAnimations}
                      onChange={(event) => setShowAnimations(event.target.checked)}
                      disabled={isAutoRunning || !!pendingStep}
                    />
                    <span>Visa animationer</span>
                  </label>
                  {!showAnimations && (
                    <p className="q-environment-description" id="q-training-speed-note">
                      Utan animationer körs träningen så snabbt som möjligt.
                    </p>
                  )}
                  <label className="q-action-option">
                    <input type="checkbox" checked={showBestPath}
                      onChange={(event) => setShowBestPath(event.target.checked)} disabled={editMode} />
                    <span>Visa bästa väg enligt Q-tabellen</span>
                  </label>
                  {showBestPath && <p className="q-environment-description">
                    {pathSummary} Vägen börjar i A och följer det högsta giltiga Q-värdet i varje tillstånd.
                    {bestPath?.hasTies && " Vid lika värden används riktningarnas ordning i tabellen."}
                  </p>}
                  <div className="q-btn-row">
                    <button
                      className="q-btn q-btn-primary"
                      onClick={handleTrainingButton}
                      disabled={!canTrain || editMode || !!pendingStep || stopRequested}
                    >
                      {isAutoRunning
                        ? stopRequested
                          ? "Stoppar..."
                          : "Stoppa träning"
                        : "Starta träning"}
                    </button>
                  </div>
                </article>
                <article className="q-controls-card">
                  <h2>Egen miljö</h2>
                  <p>Ändra rutorna, flytta start och mål eller välj andra riktningar.</p>
                  <button className={`q-btn ${editMode ? "q-btn-secondary" : ""}`}
                    onClick={toggleEditMode} disabled={isAutoRunning || !!pendingStep}>
                    {editMode ? "Lämna redigeringsläge" : "Redigeringsläge (4×4)"}
                  </button>
                </article>
              </div>
            </details>
          </div>
        </div>
      </section>

    </div>
  );
}

export default QLearningPage;
