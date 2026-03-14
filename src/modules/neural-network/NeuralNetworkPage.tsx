
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  BATCH_SIZE,
  INITIAL_PARAMS,
  LABEL_TO_TARGET,
  LEARNING_RATE_AUTO,
  LEARNING_RATE_MANUAL,
  type IrrigationSample,
  type Label,
  labelFromOutput,
  testData,
  trainingData
} from "./neuralNetworkData";
import {
  applyGradients,
  cloneParams,
  computeBatchGradients,
  computeGradients,
  evaluateDataset,
  forwardPassCore,
  forwardPassModel,
  normalizeInputs,
  shuffle,
  type ForwardPassModel
} from "./neuralNetworkMath";
import "./neuralNetwork.css";

type FlashKind = "forward" | "update";

interface HiddenDisplay {
  pre: number | null;
  post: number | null;
}

interface ManualForwardCache {
  example: IrrigationSample;
  forward: ForwardPassModel;
}

interface ConnectionLine {
  id: string;
  from: number;
  to: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  labelX: number;
  labelY: number;
  angle: number;
}

const EMPTY_HIDDEN: HiddenDisplay[] = [
  { pre: null, post: null },
  { pre: null, post: null },
  { pre: null, post: null }
];

const MANUAL_FACTOR = 2;
const wait = (ms: number): Promise<void> =>
  new Promise((resolve) => window.setTimeout(resolve, ms));
const round2 = (value: number): number => Math.round(value * 100) / 100;
const format2 = (value: number): string => round2(value).toFixed(2);
const formatBias = (value: number): string =>
  value >= 0 ? `+${format2(value)}` : format2(value);
const formatMathFactor = (value: number): string => {
  const formatted = format2(value);
  return value < 0 ? `(${formatted})` : formatted;
};
const formatMathResult = (value: number): string => format2(value);

function getEdgeMid(
  element: HTMLElement,
  containerRect: DOMRect,
  side: "left" | "right"
): { x: number; y: number } {
  const rect = element.getBoundingClientRect();
  return {
    x: (side === "right" ? rect.right : rect.left) - containerRect.left,
    y: rect.top + rect.height / 2 - containerRect.top
  };
}

function makeConnection(
  id: string,
  from: number,
  to: number,
  start: { x: number; y: number },
  end: { x: number; y: number }
): ConnectionLine {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy) || 1;
  const labelRatio = 0.33;
  const normalOffset = -7;
  const labelX = start.x + dx * labelRatio + (-dy / length) * normalOffset;
  const labelY = start.y + dy * labelRatio + (dx / length) * normalOffset;

  return {
    id,
    from,
    to,
    x1: start.x,
    y1: start.y,
    x2: end.x,
    y2: end.y,
    labelX,
    labelY,
    angle: (Math.atan2(dy, dx) * 180) / Math.PI
  };
}

function tableMarkup(rows: IrrigationSample[]): JSX.Element {
  return (
    <div className="neural-net-table-scroll">
      <table className="neural-net-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Jordfukt (%)</th>
            <th>Temp (°C)</th>
            <th>Bevattna?</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.id}-${row.moisture}-${row.temperature}`}>
              <td>{row.id}</td>
              <td>{row.moisture}</td>
              <td>{row.temperature}</td>
              <td>{row.label}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function NeuralNetworkPage(): JSX.Element {
  const [params, setParams] = useState(() => cloneParams(INITIAL_PARAMS));
  const paramsRef = useRef(params);

  const [moisture, setMoisture] = useState(40);
  const [temperature, setTemperature] = useState(20);
  const modelInputs = useMemo<[number, number]>(
    () => normalizeInputs([moisture, temperature]),
    [moisture, temperature]
  );

  const [hidden, setHidden] = useState<HiddenDisplay[]>(EMPTY_HIDDEN);
  const [output, setOutput] = useState<number | null>(null);
  const [prediction, setPrediction] = useState<Label | null>(null);

  const [visibleHiddenNodes, setVisibleHiddenNodes] = useState<[boolean, boolean, boolean]>([
    true,
    true,
    true
  ]);
  const [showHiddenCalculationPanel, setShowHiddenCalculationPanel] = useState(false);
  const [showOutputCalculationPanel, setShowOutputCalculationPanel] = useState(false);

  const [manualMode, setManualMode] = useState(false);
  const [manualAnimating, setManualAnimating] = useState(false);
  const [manualPointer, setManualPointer] = useState(0);
  const [manualExampleInfo, setManualExampleInfo] = useState("");
  const [manualStatus, setManualStatus] = useState("");
  const [manualFeedback, setManualFeedback] = useState("");
  const [manualFeedbackKind, setManualFeedbackKind] = useState<
    "correct" | "wrong" | ""
  >("");
  const [manualForwardCache, setManualForwardCache] =
    useState<ManualForwardCache | null>(null);

  const [autoRunning, setAutoRunning] = useState(false);
  const [autoAnimationEnabled, setAutoAnimationEnabled] = useState(true);
  const [autoRunWithAnimations, setAutoRunWithAnimations] = useState(false);
  const [autoStopRequested, setAutoStopRequested] = useState(false);
  const [epochs, setEpochs] = useState(12);

  const [trainingEvaluation, setTrainingEvaluation] = useState("");
  const [testEvaluation, setTestEvaluation] = useState("");

  const [flashState, setFlashState] = useState<Record<string, FlashKind>>({});
  const [ihLines, setIhLines] = useState<ConnectionLine[]>([]);
  const [hoLines, setHoLines] = useState<ConnectionLine[]>([]);

  const timeoutsRef = useRef<number[]>([]);
  const autoStopRef = useRef(autoStopRequested);

  const stageRef = useRef<HTMLDivElement | null>(null);
  const inputNodeRefs = useRef<Array<HTMLDivElement | null>>([]);
  const hiddenNodeRefs = useRef<Array<HTMLDivElement | null>>([]);
  const outputNodeRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    paramsRef.current = params;
  }, [params]);

  useEffect(() => {
    autoStopRef.current = autoStopRequested;
  }, [autoStopRequested]);

  useEffect(() => {
    return () => {
      timeoutsRef.current.forEach((timeoutId) => window.clearTimeout(timeoutId));
      timeoutsRef.current = [];
    };
  }, []);

  const flash = useCallback((id: string, kind: FlashKind, duration = 800) => {
    setFlashState((previous) => ({ ...previous, [id]: kind }));
    const timeoutId = window.setTimeout(() => {
      setFlashState((previous) => {
        const next = { ...previous };
        delete next[id];
        return next;
      });
    }, duration);
    timeoutsRef.current.push(timeoutId);
  }, []);

  const resetOutputs = useCallback(() => {
    setHidden(EMPTY_HIDDEN);
    setOutput(null);
    setPrediction(null);
  }, []);

  const hiddenCalcs = useMemo(() => {
    return params.B_H.map((bias, index) => {
      const w0 = params.W_IH[0][index];
      const w1 = params.W_IH[1][index];
      const sum = modelInputs[0] * w0 + modelInputs[1] * w1 + bias;
      return { w0, w1, bias, sum };
    });
  }, [modelInputs, params]);

  const outputCalc = useMemo(() => {
    const forward = forwardPassCore(modelInputs, params);
    const sum = forward.actHidden.reduce((accumulator, value, index) => {
      return accumulator + value * params.W_HO[index];
    }, params.B_O);
    return {
      hiddenValues: forward.actHidden,
      weights: params.W_HO,
      bias: params.B_O,
      sum
    };
  }, [modelInputs, params]);

  const visibleHiddenBias = useMemo(() => {
    return visibleHiddenNodes;
  }, [visibleHiddenNodes]);

  const isIHLineVisible = useCallback(
    (line: ConnectionLine): boolean => {
      return visibleHiddenNodes[line.to];
    },
    [visibleHiddenNodes]
  );

  const recomputeConnections = useCallback(() => {
    const stage = stageRef.current;
    const outputNode = outputNodeRef.current;

    if (!stage || !outputNode) return;

    if (
      inputNodeRefs.current.some((node) => node === null) ||
      hiddenNodeRefs.current.some((node) => node === null)
    ) {
      return;
    }

    const stageRect = stage.getBoundingClientRect();
    const inputPoints = inputNodeRefs.current.map((node) =>
      getEdgeMid(node as HTMLElement, stageRect, "right")
    );
    const hiddenLeftPoints = hiddenNodeRefs.current.map((node) =>
      getEdgeMid(node as HTMLElement, stageRect, "left")
    );
    const hiddenRightPoints = hiddenNodeRefs.current.map((node) =>
      getEdgeMid(node as HTMLElement, stageRect, "right")
    );
    const outputLeftPoint = getEdgeMid(outputNode, stageRect, "left");

    const nextIH: ConnectionLine[] = [];
    const nextHO: ConnectionLine[] = [];

    for (let j = 0; j < 3; j += 1) {
      for (let i = 0; i < 2; i += 1) {
        nextIH.push(
          makeConnection(
            `ih-line-${i}-${j}`,
            i,
            j,
            inputPoints[i],
            hiddenLeftPoints[j]
          )
        );
      }
    }

    for (let j = 0; j < 3; j += 1) {
      nextHO.push(
        makeConnection(`ho-line-${j}`, j, j, hiddenRightPoints[j], outputLeftPoint)
      );
    }

    setIhLines(nextIH);
    setHoLines(nextHO);
  }, []);

  useEffect(() => {
    const onResize = () => {
      window.requestAnimationFrame(recomputeConnections);
    };

    onResize();
    window.addEventListener("resize", onResize);

    const observer = new ResizeObserver(onResize);
    if (stageRef.current) {
      observer.observe(stageRef.current);
    }

    return () => {
      window.removeEventListener("resize", onResize);
      observer.disconnect();
    };
  }, [recomputeConnections]);

  useEffect(() => {
    const frameId = window.requestAnimationFrame(recomputeConnections);
    return () => window.cancelAnimationFrame(frameId);
  }, [recomputeConnections, visibleHiddenNodes, hidden, output]);

  const toggleHiddenNodeVisibility = (nodeIndex: 0 | 1 | 2) => {
    setVisibleHiddenNodes((previous) => {
      const next: [boolean, boolean, boolean] = [...previous];
      next[nodeIndex] = !previous[nodeIndex];
      return next;
    });
  };

  const calculateHidden = () => {
    const forward = forwardPassCore(modelInputs, paramsRef.current);
    setHidden(
      forward.preHidden.map((pre, index) => ({
        pre,
        post: forward.actHidden[index]
      }))
    );
    setShowHiddenCalculationPanel(true);
  };

  const calculateOutput = () => {
    const forward = forwardPassCore(modelInputs, paramsRef.current);
    setHidden(
      forward.preHidden.map((pre, index) => ({
        pre,
        post: forward.actHidden[index]
      }))
    );
    setOutput(forward.output);
    setPrediction(labelFromOutput(forward.output));
    setShowOutputCalculationPanel(true);
  };

  const toggleManualMode = () => {
    if (autoRunning) return;

    if (manualMode) {
      setManualMode(false);
      setManualForwardCache(null);
      setManualExampleInfo("");
      setManualStatus("");
      setManualFeedback("");
      setManualFeedbackKind("");
      return;
    }

    setManualMode(true);
    setManualForwardCache(null);
    setManualFeedback("");
    setManualFeedbackKind("");
    setManualExampleInfo('Klicka på "Ladda nästa exempel" för att börja manuell träning.');
    setManualStatus(`Nästa exempel i kön: ${trainingData[manualPointer].id}`);
  };

  const runManualForward = useCallback(
    async (example: IrrigationSample) => {
      setManualAnimating(true);
      setManualFeedback("");
      setManualFeedbackKind("");
      setManualForwardCache(null);
      setManualExampleInfo(
        `Exempel ${example.id}: jordfuktighet ${example.moisture} %, temperatur ${example.temperature} °C, mål: ${example.label}.`
      );
      setManualStatus("Laddar indata i nätverket...");

      setMoisture(example.moisture);
      setTemperature(example.temperature);
      flash("input-node-0", "forward");
      flash("input-node-1", "forward");
      await wait(500 * MANUAL_FACTOR);

      const forward = forwardPassModel([example.moisture, example.temperature], paramsRef.current);
      resetOutputs();

      for (let j = 0; j < 3; j += 1) {
        flash(`hidden-node-${j}`, "forward");
        flash(`hidden-bias-${j}`, "forward");
        flash(`ih-line-0-${j}`, "forward");
        flash(`ih-line-1-${j}`, "forward");
        setHidden((previous) => {
          const next = previous.slice();
          next[j] = { pre: forward.preHidden[j], post: forward.actHidden[j] };
          return next;
        });
        await wait(450 * MANUAL_FACTOR);
      }

      setManualStatus("Beräknar utmatningslagret...");
      flash("output-node", "forward");
      flash("output-bias", "forward");
      for (let j = 0; j < 3; j += 1) {
        flash(`ho-line-${j}`, "forward");
      }
      setOutput(forward.output);
      setPrediction(labelFromOutput(forward.output));
      await wait(350 * MANUAL_FACTOR);

      const predicted = labelFromOutput(forward.output);
      if (predicted === example.label) {
        setManualFeedback(`Rätt! Nätverket förutsåg ${predicted}.`);
        setManualFeedbackKind("correct");
      } else {
        setManualFeedback(`Fel! Förväntat ${example.label}, men nätverket gav ${predicted}.`);
        setManualFeedbackKind("wrong");
      }

      setManualForwardCache({ example, forward });
      setManualStatus('Klicka på "Bakåtpropagering" för att uppdatera vikterna.');
      setManualAnimating(false);
    },
    [flash, resetOutputs]
  );

  const loadNextExample = async () => {
    if (!manualMode || manualAnimating || autoRunning || manualForwardCache !== null) return;
    const example = trainingData[manualPointer];
    setManualPointer((previous) => (previous + 1) % trainingData.length);
    await runManualForward(example);
  };

  const runBackprop = async () => {
    if (!manualForwardCache || manualAnimating || autoRunning) return;

    setManualAnimating(true);
    setManualStatus("Uppdaterar parametrar med bakåtpropagering...");

    const snapshot = cloneParams(paramsRef.current);
    const gradients = computeGradients(
      manualForwardCache.forward,
      manualForwardCache.forward.modelInputs,
      LABEL_TO_TARGET[manualForwardCache.example.label],
      snapshot
    );

    let next = cloneParams(snapshot);

    flash("output-bias", "update", 900);
    next.B_O -= LEARNING_RATE_MANUAL * gradients.B_O;
    setParams(cloneParams(next));
    await wait(500);

    for (let j = 0; j < 3; j += 1) {
      next.W_HO[j] -= LEARNING_RATE_MANUAL * gradients.W_HO[j];
      flash(`ho-line-${j}`, "update");
      setParams(cloneParams(next));
      await wait(400);
    }

    for (let j = 0; j < 3; j += 1) {
      next.B_H[j] -= LEARNING_RATE_MANUAL * gradients.B_H[j];
      flash(`hidden-bias-${j}`, "update", 900);
      setParams(cloneParams(next));
      await wait(400);
    }

    for (let j = 0; j < 3; j += 1) {
      for (let i = 0; i < 2; i += 1) {
        next.W_IH[i][j] -= LEARNING_RATE_MANUAL * gradients.W_IH[i][j];
        flash(`ih-line-${i}-${j}`, "update");
        setParams(cloneParams(next));
        await wait(320);
      }
    }

    setManualForwardCache(null);
    setManualStatus(`Nästa exempel i kön: ${trainingData[manualPointer].id}`);
    setManualAnimating(false);
  };

  const runAutoTraining = useCallback(async () => {
    if (autoRunning) return;
    if (manualMode) {
      setManualStatus("Avsluta manuell träning innan automatisk träning startar.");
      return;
    }

    const totalEpochs = Math.max(1, Number.isFinite(epochs) ? Math.floor(epochs) : 1);
    const animate = autoAnimationEnabled;

    setAutoRunning(true);
    setAutoStopRequested(false);
    setAutoRunWithAnimations(animate);

    let nextParams = cloneParams(paramsRef.current);
    let lastStatus = "Startar automatisk träning...";
    let stoppedEarly = false;
    setManualStatus(lastStatus);

    for (let epoch = 1; epoch <= totalEpochs; epoch += 1) {
      const shuffledData = shuffle(trainingData);
      let lossSum = 0;
      let count = 0;

      for (let index = 0; index < shuffledData.length; index += BATCH_SIZE) {
        const batch = shuffledData.slice(index, index + BATCH_SIZE);
        const gradients = computeBatchGradients(batch, nextParams);
        nextParams = applyGradients(nextParams, gradients, LEARNING_RATE_AUTO);
        lossSum += gradients.loss * gradients.batchSize;
        count += gradients.batchSize;

        if (animate) {
          setParams(cloneParams(nextParams));
          await wait(30);
        }
      }

      if (!animate) {
        setParams(cloneParams(nextParams));
      }

      const avgLoss = count === 0 ? 0 : lossSum / count;
      const epochStatus = `Epok ${epoch}/${totalEpochs} klar - medelfel: ${format2(avgLoss)}`;
      lastStatus = epochStatus;
      setManualStatus(epochStatus);

      if (animate && autoStopRef.current && epoch < totalEpochs) {
        stoppedEarly = true;
        setManualStatus(`${epochStatus} - automatisk träning avslutas.`);
        break;
      }
    }

    setParams(cloneParams(nextParams));

    if (animate && autoStopRef.current) {
      if (!stoppedEarly) {
        setManualStatus(`${lastStatus} - automatisk träning avslutas.`);
      }
    } else {
      setManualStatus(`${lastStatus} ✓`);
    }

    setAutoRunning(false);
    setAutoRunWithAnimations(false);
    setAutoStopRequested(false);
  }, [autoAnimationEnabled, autoRunning, epochs, manualMode]);

  const handleAutoTrain = () => {
    if (autoRunning && autoRunWithAnimations && !autoStopRequested) {
      setAutoStopRequested(true);
      setManualStatus((previous) => {
        const text = previous.trim();
        if (text.length === 0) return "Automatisk träning avslutas efter pågående epok.";
        return `${text} - automatisk träning avslutas efter pågående epok.`;
      });
      return;
    }

    void runAutoTraining();
  };

  const evaluateTraining = () => {
    const result = evaluateDataset(trainingData, paramsRef.current);
    setTrainingEvaluation(
      `Medelkvadratiskt fel: ${format2(result.mse)} - Träffsäkerhet: ${format2(result.accuracy)} %`
    );
  };

  const evaluateTest = () => {
    const result = evaluateDataset(testData, paramsRef.current);
    setTestEvaluation(
      `Medelkvadratiskt fel: ${format2(result.mse)} - Träffsäkerhet: ${format2(result.accuracy)} %`
    );
  };

  const resetNetwork = () => {
    timeoutsRef.current.forEach((timeoutId) => window.clearTimeout(timeoutId));
    timeoutsRef.current = [];
    setFlashState({});

    setParams(cloneParams(INITIAL_PARAMS));
    setMoisture(40);
    setTemperature(20);

    setVisibleHiddenNodes([true, true, true]);
    setShowHiddenCalculationPanel(false);
    setShowOutputCalculationPanel(false);

    setManualMode(false);
    setManualPointer(0);
    setManualAnimating(false);
    setManualForwardCache(null);
    setManualExampleInfo("");
    setManualStatus("");
    setManualFeedback("");
    setManualFeedbackKind("");

    setAutoAnimationEnabled(true);
    setAutoRunWithAnimations(false);
    setAutoStopRequested(false);
    setEpochs(12);

    setTrainingEvaluation("");
    setTestEvaluation("");
    resetOutputs();
  };

  const autoTrainButtonText = autoRunning
    ? autoRunWithAnimations
      ? autoStopRequested
        ? "Avslutar..."
        : "Avsluta träning"
      : "Träning pågår..."
    : "Automatisk träning";

  const autoTrainDisabled =
    manualMode || (autoRunning && (!autoRunWithAnimations || autoStopRequested));
  const loadNextDisabled =
    autoRunning || manualAnimating || !manualMode || manualForwardCache !== null;
  const backpropDisabled =
    autoRunning || manualAnimating || !manualMode || manualForwardCache === null;

  return (
    <div className="container page-flow neural-net-page">
      <section className="section">
        <h1 className="neural-net-title">Neuralt nätverk för bevattning</h1>
        <p className="neural-net-lead">
          Den här modulen visar hur ett neuralt nätverk fattar beslut, gör sina beräkningar steg för
          steg och tränas med nya exempel.
        </p>
      </section>

      <section className="section neural-net-network-section">
        <div className="neural-net-network-head">
          <h2>Nätverksvy</h2>
          <button
            className="neural-net-btn neural-net-btn-danger"
            onClick={resetNetwork}
            disabled={autoRunning || manualAnimating}
          >
            Återställ nätverket
          </button>
        </div>

        <div className="neural-net-stage-scroll">
          <div className="neural-net-stage-wrap">
            <div className="neural-net-stage" ref={stageRef}>
              <svg className="neural-net-connection-svg" aria-hidden="true">
                {ihLines.map((line) => {
                  const isFlashing = flashState[line.id] !== undefined;
                  const visible = isIHLineVisible(line) || isFlashing;

                  const lineClass =
                    flashState[line.id] === "forward"
                      ? "neural-net-conn-line neural-net-conn-forward"
                      : flashState[line.id] === "update"
                        ? "neural-net-conn-line neural-net-conn-update"
                        : "neural-net-conn-line";
                  const textClass =
                    flashState[line.id] === "forward"
                      ? "neural-net-conn-text neural-net-conn-text-forward"
                      : flashState[line.id] === "update"
                        ? "neural-net-conn-text neural-net-conn-text-update"
                        : "neural-net-conn-text";

                  return (
                    <g key={line.id} className={visible ? "" : "neural-net-conn-hidden"}>
                      <line className={lineClass} x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} />
                      <text
                        className={textClass}
                        x={line.labelX}
                        y={line.labelY}
                        transform={`rotate(${line.angle} ${line.labelX} ${line.labelY})`}
                      >
                        {params.W_IH[line.from][line.to].toFixed(2)}
                      </text>
                    </g>
                  );
                })}

                {hoLines.map((line) => {
                  const visible = true;

                  const lineClass =
                    flashState[line.id] === "forward"
                      ? "neural-net-conn-line neural-net-conn-forward"
                      : flashState[line.id] === "update"
                        ? "neural-net-conn-line neural-net-conn-update"
                        : "neural-net-conn-line";
                  const textClass =
                    flashState[line.id] === "forward"
                      ? "neural-net-conn-text neural-net-conn-text-forward"
                      : flashState[line.id] === "update"
                        ? "neural-net-conn-text neural-net-conn-text-update"
                        : "neural-net-conn-text";

                  return (
                    <g key={line.id} className={visible ? "" : "neural-net-conn-hidden"}>
                      <line className={lineClass} x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} />
                      <text
                        className={textClass}
                        x={line.labelX}
                        y={line.labelY}
                        transform={`rotate(${line.angle} ${line.labelX} ${line.labelY})`}
                      >
                        {params.W_HO[line.from].toFixed(2)}
                      </text>
                    </g>
                  );
                })}
              </svg>
              <div className="neural-net-layer-column neural-net-layer-input">
                <div className="neural-net-layer-head">
                  <h3>Inmatningslager</h3>
                </div>
                <div className="neural-net-layer-stack">
                  {[0, 1].map((inputIndex) => (
                    <div
                      key={`input-${inputIndex}`}
                      ref={(element) => {
                        inputNodeRefs.current[inputIndex] = element;
                      }}
                      className={`neural-net-input-node ${
                        flashState[`input-node-${inputIndex}`] === "forward"
                          ? "neural-net-node-forward"
                          : flashState[`input-node-${inputIndex}`] === "update"
                            ? "neural-net-node-update"
                            : ""
                      }`}
                    >
                      <div className="neural-net-input-part neural-net-input-raw">
                        <label htmlFor={inputIndex === 0 ? "moisture-input" : "temperature-input"}>
                          {inputIndex === 0 ? "Jordfuktighet (%)" : "Lufttemperatur (°C)"}
                        </label>
                        <input
                          id={inputIndex === 0 ? "moisture-input" : "temperature-input"}
                          type="number"
                          step="1"
                          value={inputIndex === 0 ? moisture : temperature}
                          onChange={(event) => {
                            const value = Number.parseFloat(event.target.value) || 0;
                            if (inputIndex === 0) {
                              setMoisture(value);
                            } else {
                              setTemperature(value);
                            }
                          }}
                        />
                      </div>
                      <div className="neural-net-input-part neural-net-input-normalized">
                        <span>Normaliserat</span>
                        <strong>{format2(modelInputs[inputIndex])}</strong>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="neural-net-layer-column neural-net-layer-hidden">
                <div className="neural-net-layer-head">
                  <h3>Dolt lager</h3>
                </div>
                <div className="neural-net-layer-stack">
                  {hidden.map((cell, index) => {
                    const showBias = visibleHiddenBias[index] || flashState[`hidden-bias-${index}`] !== undefined;

                    return (
                      <div
                        key={`hidden-${index}`}
                        ref={(element) => {
                          hiddenNodeRefs.current[index] = element;
                        }}
                        className={`neural-net-hidden-node ${
                          flashState[`hidden-node-${index}`] === "forward"
                            ? "neural-net-node-forward"
                            : flashState[`hidden-node-${index}`] === "update"
                              ? "neural-net-node-update"
                              : ""
                        }`}
                      >
                        <span
                          className={`neural-net-bias neural-net-bias-hidden ${
                            showBias ? "" : "neural-net-bias-invisible"
                          } ${
                            flashState[`hidden-bias-${index}`] === "forward"
                              ? "neural-net-bias-forward"
                              : flashState[`hidden-bias-${index}`] === "update"
                                ? "neural-net-bias-update"
                                : ""
                          }`}
                        >
                          {formatBias(params.B_H[index])}
                        </span>
                        <div className="neural-net-hidden-cell neural-net-hidden-pre">
                          {cell.pre === null ? "?" : format2(cell.pre)}
                        </div>
                        <div className="neural-net-hidden-cell neural-net-hidden-relu" aria-hidden="true">
                          <svg viewBox="0 0 40 40">
                            <path d="M5 22H20L35 7" stroke="currentColor" strokeWidth="4" fill="none" />
                          </svg>
                        </div>
                        <div className="neural-net-hidden-cell neural-net-hidden-post">
                          {cell.post === null ? "?" : format2(cell.post)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="neural-net-layer-column neural-net-layer-output">
                <div className="neural-net-layer-head">
                  <h3>Utmatningslagret</h3>
                  <p className="neural-net-output-note">Bevattningen startas om värdet är positivt (&gt;0)</p>
                </div>
                <div className="neural-net-layer-stack neural-net-output-stack">
                  <div className="neural-net-output-wrapper">
                    <div
                      ref={outputNodeRef}
                      className={`neural-net-output-node ${
                        flashState["output-node"] === "forward"
                          ? "neural-net-node-forward"
                          : flashState["output-node"] === "update"
                            ? "neural-net-node-update"
                            : ""
                      }`}
                    >
                      <span
                        className={`neural-net-bias neural-net-bias-output ${
                          flashState["output-bias"] === "forward"
                            ? "neural-net-bias-forward"
                            : flashState["output-bias"] === "update"
                              ? "neural-net-bias-update"
                              : ""
                        }`}
                      >
                        {formatBias(params.B_O)}
                      </span>
                      <span className="neural-net-output-value">{output === null ? "?" : format2(output)}</span>
                    </div>
                    <p className="neural-net-prediction">Bevattning startas: {prediction ?? "?"}</p>
                  </div>
                </div>
              </div>

              <div className="neural-net-inline-calcs" aria-live="polite">
                <div
                  className={`neural-net-inline-calc neural-net-inline-calc-hidden ${
                    showHiddenCalculationPanel ? "is-visible" : ""
                  }`}
                >
                  {visibleHiddenNodes.some((isVisible) => isVisible) ? (
                    <div className="neural-net-hidden-formula-list">
                      {hiddenCalcs.map((hiddenCalc, index) => {
                        if (!visibleHiddenNodes[index]) return null;

                        return (
                          <p key={`hidden-formula-${index}`} className="neural-net-formula neural-net-formula-compact neural-net-math">
                            <strong className="neural-net-math-neuron">N{index + 1}</strong>
                            <span className="neural-net-math-op">: </span>
                            <span>{formatMathFactor(hiddenCalc.w0)}</span>
                            <span className="neural-net-math-op"> · </span>
                            <strong className="neural-net-math-value neural-net-math-input">
                              {formatMathFactor(modelInputs[0])}
                            </strong>
                            <span className="neural-net-math-op"> + </span>
                            <span>{formatMathFactor(hiddenCalc.w1)}</span>
                            <span className="neural-net-math-op"> · </span>
                            <strong className="neural-net-math-value neural-net-math-input">
                              {formatMathFactor(modelInputs[1])}
                            </strong>
                            <span className="neural-net-math-op"> + </span>
                            <span>{formatMathFactor(hiddenCalc.bias)}</span>
                            <span className="neural-net-math-op"> = </span>
                            <strong className="neural-net-math-value neural-net-math-hidden">
                              {formatMathResult(hiddenCalc.sum)}
                            </strong>
                          </p>
                        );
                      })}
                    </div>
                  ) : (
                    <p className="neural-net-muted">Markera minst en av N1, N2 eller N3 för att visa uträkningarna.</p>
                  )}
                </div>

                <div
                  className={`neural-net-inline-calc neural-net-inline-calc-output ${
                    showOutputCalculationPanel ? "is-visible" : ""
                  }`}
                >
                  <p className="neural-net-formula neural-net-formula-compact neural-net-math neural-net-formula-nowrap">
                    <span>{formatMathFactor(outputCalc.weights[0])}</span>
                    <span className="neural-net-math-op"> · </span>
                    <strong className="neural-net-math-value neural-net-math-hidden">
                      {formatMathFactor(outputCalc.hiddenValues[0])}
                    </strong>
                    <span className="neural-net-math-op"> + </span>
                    <span>{formatMathFactor(outputCalc.weights[1])}</span>
                    <span className="neural-net-math-op"> · </span>
                    <strong className="neural-net-math-value neural-net-math-hidden">
                      {formatMathFactor(outputCalc.hiddenValues[1])}
                    </strong>
                    <span className="neural-net-math-op"> + </span>
                    <span>{formatMathFactor(outputCalc.weights[2])}</span>
                    <span className="neural-net-math-op"> · </span>
                    <strong className="neural-net-math-value neural-net-math-hidden">
                      {formatMathFactor(outputCalc.hiddenValues[2])}
                    </strong>
                    <span className="neural-net-math-op"> + </span>
                    <span>{formatMathFactor(outputCalc.bias)}</span>
                    <span className="neural-net-math-op"> = </span>
                    <strong className="neural-net-math-value neural-net-math-output">
                      {formatMathResult(outputCalc.sum)}
                    </strong>
                  </p>
                </div>
              </div>
            </div>

            <div className="neural-net-stage-actions">
              <div className="neural-net-visibility-controls neural-net-hidden-visibility" aria-label="Visa vikter och bias per nod">
                {([0, 1, 2] as const).map((nodeIndex) => (
                  <label key={`node-visibility-${nodeIndex}`} className="neural-net-check">
                    <input
                      type="checkbox"
                      checked={visibleHiddenNodes[nodeIndex]}
                      onChange={() => toggleHiddenNodeVisibility(nodeIndex)}
                    />
                    N{nodeIndex + 1}
                  </label>
                ))}
              </div>

              <button className="neural-net-btn neural-net-hidden-action-btn" onClick={calculateHidden}>
                Beräkna dolt lager
              </button>

              <button className="neural-net-btn neural-net-output-action-btn" onClick={calculateOutput}>
                Beräkna utmatningslagret
              </button>
            </div>
          </div>
        </div>
      </section>

      <section className="section neural-net-manual-panel">
        <div className="neural-net-manual-grid">
          <div>
            <button className="neural-net-btn neural-net-btn-accent" onClick={loadNextExample} disabled={loadNextDisabled}>
              Ladda nästa exempel
            </button>
            <p className="neural-net-info-text">{manualExampleInfo}</p>
          </div>

          <div>
            <p className="neural-net-status">{manualStatus}</p>
          </div>

          <div>
            <button className="neural-net-btn neural-net-btn-danger" onClick={runBackprop} disabled={backpropDisabled}>
              Bakåtpropagering
            </button>
            <p className={`neural-net-feedback ${manualFeedbackKind === "correct" ? "ok" : manualFeedbackKind === "wrong" ? "bad" : ""}`}>
              {manualFeedback}
            </p>
          </div>
        </div>
      </section>

      <section className="section">
        <h2>Träningsområde</h2>

        <div className="neural-net-action-stack">
          <div className="neural-net-action-row neural-net-action-row-main">
            <button className={`neural-net-btn ${manualMode ? "neural-net-btn-secondary" : "neural-net-btn-primary"}`} onClick={toggleManualMode} disabled={autoRunning}>
              {manualMode ? "Avsluta manuell träning" : "Manuell träning"}
            </button>
            <button className="neural-net-btn neural-net-btn-primary" onClick={handleAutoTrain} disabled={autoTrainDisabled}>
              {autoTrainButtonText}
            </button>

            <label className="neural-net-field neural-net-field-compact">
              <span>Epoker</span>
              <input
                type="number"
                min={1}
                value={epochs}
                onChange={(event) => setEpochs(Math.max(1, Number.parseInt(event.target.value, 10) || 1))}
                disabled={autoRunning}
              />
            </label>

            <label className="neural-net-check">
              <input type="checkbox" checked={autoAnimationEnabled} onChange={(event) => setAutoAnimationEnabled(event.target.checked)} disabled={autoRunning} />
              Visa animationer
            </label>
          </div>
        </div>

        <div className="neural-net-eval-grid">
          <article className="neural-net-eval-card">
            <div className="neural-net-button-row">
              <button className="neural-net-btn" onClick={evaluateTraining} disabled={autoRunning || manualAnimating}>
                Utvärdera på träningsdata
              </button>
            </div>
            <p className="neural-net-status-left">{trainingEvaluation}</p>
            {tableMarkup(trainingData)}
          </article>

          <article className="neural-net-eval-card">
            <div className="neural-net-button-row">
              <button className="neural-net-btn" onClick={evaluateTest} disabled={autoRunning || manualAnimating}>
                Utvärdera på testdata
              </button>
            </div>
            <p className="neural-net-status-left">{testEvaluation}</p>
            {tableMarkup(testData)}
          </article>
        </div>
      </section>
    </div>
  );
}

export default NeuralNetworkPage;
