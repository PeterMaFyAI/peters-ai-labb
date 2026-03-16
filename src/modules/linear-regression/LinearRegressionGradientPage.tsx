import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import katex from "katex";
import "katex/dist/katex.min.css";
import {
  calculateGradientStandardized,
  calculateMSE,
  calculateOptimalParameters,
  dataPoints,
  featureScalingStats,
  toOriginalWeightBias,
  toStandardizedWeightBias
} from "./linearRegressionData";
import "./linearRegression.css";

interface HistoryPoint {
  weightStd: number;
  biasStd: number;
  weight: number;
  bias: number;
  mse: number;
}

interface ParameterRanges {
  wMin: number;
  wMax: number;
  bMin: number;
  bMax: number;
}

interface SurfaceMesh {
  weightValues: number[];
  biasValues: number[];
  zValues: number[][];
}

const DEFAULT_INITIAL_WEIGHT = 0.1;
const DEFAULT_INITIAL_BIAS = 1;
const DEFAULT_LEARNING_RATE = 0.1;

const optimalParameters = calculateOptimalParameters(dataPoints);

function createLinearRange(start: number, end: number, count: number): number[] {
  if (count < 2) {
    return [start];
  }

  const values: number[] = [];
  const step = (end - start) / (count - 1);
  for (let index = 0; index < count; index += 1) {
    values.push(start + index * step);
  }

  return values;
}

function buildRangesAround(startWeight: number, startBias: number): ParameterRanges {
  const weightDistance = Math.abs(startWeight - optimalParameters.weight);
  const biasDistance = Math.abs(startBias - optimalParameters.bias);

  const weightPadding = Math.max(0.03, 1.5 * weightDistance);
  const biasPadding = Math.max(1, 1.5 * biasDistance);

  return {
    wMin: optimalParameters.weight - weightPadding,
    wMax: optimalParameters.weight + weightPadding,
    bMin: optimalParameters.bias - biasPadding,
    bMax: optimalParameters.bias + biasPadding
  };
}

function buildRangesFromHistory(history: HistoryPoint[]): ParameterRanges {
  const baseRanges = buildRangesAround(history[0].weight, history[0].bias);

  const weights = history.map((point) => point.weight);
  const biases = history.map((point) => point.bias);

  const minWeight = Math.min(...weights);
  const maxWeight = Math.max(...weights);
  const minBias = Math.min(...biases);
  const maxBias = Math.max(...biases);

  const currentWidth = baseRanges.wMax - baseRanges.wMin;
  const currentHeight = baseRanges.bMax - baseRanges.bMin;

  const weightMargin = Math.max(0.01, 0.12 * currentWidth);
  const biasMargin = Math.max(0.15, 0.12 * currentHeight);

  return {
    wMin: Math.min(baseRanges.wMin, minWeight - weightMargin),
    wMax: Math.max(baseRanges.wMax, maxWeight + weightMargin),
    bMin: Math.min(baseRanges.bMin, minBias - biasMargin),
    bMax: Math.max(baseRanges.bMax, maxBias + biasMargin)
  };
}

function buildSurfaceMesh(ranges: ParameterRanges, resolution = 55): SurfaceMesh {
  const weightValues = createLinearRange(ranges.wMin, ranges.wMax, resolution);
  const biasValues = createLinearRange(ranges.bMin, ranges.bMax, resolution);

  const zValues = biasValues.map((bias) =>
    weightValues.map((weight) => calculateMSE(weight, bias, dataPoints))
  );

  return {
    weightValues,
    biasValues,
    zValues
  };
}

function createHistoryPoint(weightStd: number, biasStd: number): HistoryPoint {
  const original = toOriginalWeightBias(weightStd, biasStd, featureScalingStats);
  return {
    weightStd,
    biasStd,
    weight: original.weight,
    bias: original.bias,
    mse: calculateMSE(original.weight, original.bias, dataPoints)
  };
}

function parseFiniteNumber(value: string, fallback: number): number {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parsePositiveNumber(value: string, fallback: number): number {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function renderMath(expression: string): { __html: string } {
  return {
    __html: katex.renderToString(expression, {
      throwOnError: false,
      displayMode: true
    })
  };
}

function LinearRegressionGradientPage(): JSX.Element {
  const plotElementRef = useRef<HTMLDivElement | null>(null);
  const plotlyRef = useRef<any>(null);
  const cameraRef = useRef({ eye: { x: 1.5, y: 1.4, z: 0.9 } });

  const [initialWeightInput, setInitialWeightInput] = useState(String(DEFAULT_INITIAL_WEIGHT));
  const [initialBiasInput, setInitialBiasInput] = useState(String(DEFAULT_INITIAL_BIAS));
  const [learningRateInput, setLearningRateInput] = useState(String(DEFAULT_LEARNING_RATE));

  const [history, setHistory] = useState<HistoryPoint[]>(() => {
    const standardized = toStandardizedWeightBias(
      DEFAULT_INITIAL_WEIGHT,
      DEFAULT_INITIAL_BIAS,
      featureScalingStats
    );
    return [createHistoryPoint(standardized.weightStd, standardized.biasStd)];
  });
  const [stepIndex, setStepIndex] = useState(0);

  const currentPoint = history[Math.min(stepIndex, history.length - 1)];
  const visibleHistory = useMemo(() => history.slice(0, stepIndex + 1), [history, stepIndex]);
  const ranges = useMemo(() => buildRangesFromHistory(history), [history]);
  const mesh = useMemo(() => buildSurfaceMesh(ranges), [ranges]);
  const modelFormulaHtml = useMemo(
    () => renderMath(String.raw`\hat{y}_i = w x_i + b`),
    []
  );
  const mseFormulaHtml = useMemo(
    () =>
      renderMath(
        String.raw`\mathrm{MSE}(w,b)=\frac{1}{n}\sum_{i=1}^{n}(w x_i+b-y_i)^2,\qquad n=50`
      ),
    []
  );

  const applyStartPoint = (): void => {
    const safeInitialWeight = parseFiniteNumber(initialWeightInput, DEFAULT_INITIAL_WEIGHT);
    const safeInitialBias = parseFiniteNumber(initialBiasInput, DEFAULT_INITIAL_BIAS);
    const safeLearningRate = parsePositiveNumber(learningRateInput, DEFAULT_LEARNING_RATE);

    setInitialWeightInput(String(safeInitialWeight));
    setInitialBiasInput(String(safeInitialBias));
    setLearningRateInput(String(safeLearningRate));

    const standardized = toStandardizedWeightBias(
      safeInitialWeight,
      safeInitialBias,
      featureScalingStats
    );

    setHistory([createHistoryPoint(standardized.weightStd, standardized.biasStd)]);
    setStepIndex(0);
  };

  const readLearningRate = (): number => {
    const safeLearningRate = parsePositiveNumber(learningRateInput, DEFAULT_LEARNING_RATE);
    setLearningRateInput(String(safeLearningRate));
    return safeLearningRate;
  };

  const stepForward = (): void => {
    const learningRate = readLearningRate();

    if (stepIndex < history.length - 1) {
      setStepIndex(stepIndex + 1);
      return;
    }

    const gradient = calculateGradientStandardized(
      currentPoint.weightStd,
      currentPoint.biasStd,
      featureScalingStats,
      dataPoints
    );

    const nextWeightStd = currentPoint.weightStd - learningRate * gradient.gradientWeightStd;
    const nextBiasStd = currentPoint.biasStd - learningRate * gradient.gradientBiasStd;
    const nextPoint = createHistoryPoint(nextWeightStd, nextBiasStd);

    setHistory([...history, nextPoint]);
    setStepIndex(stepIndex + 1);
  };

  const stepBackward = (): void => {
    if (stepIndex === 0) {
      return;
    }

    setStepIndex(stepIndex - 1);
  };

  useEffect(() => {
    let cancelled = false;

    const renderPlot = async (): Promise<void> => {
      const plotElement = plotElementRef.current;
      if (!plotElement) {
        return;
      }

      if (!plotlyRef.current) {
        const plotlyModule = await import("plotly.js-dist-min");
        if (cancelled) {
          return;
        }
        plotlyRef.current = plotlyModule.default;
      }

      const Plotly = plotlyRef.current;
      const layoutCamera = (plotElement as any).layout?.scene?.camera;
      if (layoutCamera) {
        cameraRef.current = layoutCamera;
      }

      const surfaceTrace = {
        type: "surface",
        name: "MSE-yta",
        x: mesh.weightValues,
        y: mesh.biasValues,
        z: mesh.zValues,
        opacity: 0.95,
        showscale: true,
        colorscale: "Viridis",
        colorbar: {
          title: "MSE"
        }
      };

      const pathTrace = {
        type: "scatter3d",
        mode: "lines+markers",
        name: "Gradientsteg",
        x: visibleHistory.map((point) => point.weight),
        y: visibleHistory.map((point) => point.bias),
        z: visibleHistory.map((point) => point.mse),
        line: {
          color: "#bf2f3a",
          width: 6
        },
        marker: {
          color: "#bf2f3a",
          size: 3
        }
      };

      const startPointTrace = {
        type: "scatter3d",
        mode: "markers",
        name: "Startpunkt",
        x: [history[0].weight],
        y: [history[0].bias],
        z: [history[0].mse],
        marker: {
          color: "#bf2f3a",
          size: 9,
          symbol: "x"
        }
      };

      const currentPointTrace = {
        type: "scatter3d",
        mode: "markers",
        name: "Aktuell punkt",
        x: [currentPoint.weight],
        y: [currentPoint.bias],
        z: [currentPoint.mse],
        marker: {
          color: "#0d6c9d",
          size: 5
        }
      };

      const layout = {
        margin: { l: 0, r: 0, t: 12, b: 0 },
        paper_bgcolor: "#ffffff",
        uirevision: "keep-gradient-camera",
        scene: {
          uirevision: "keep-gradient-camera",
          xaxis: {
            title: "w",
            range: [ranges.wMin, ranges.wMax]
          },
          yaxis: {
            title: "b",
            range: [ranges.bMin, ranges.bMax]
          },
          zaxis: {
            title: "MSE(w, b)"
          },
          camera: cameraRef.current
        },
        legend: {
          orientation: "h",
          y: 1.03
        }
      };

      await Plotly.react(
        plotElement,
        [surfaceTrace, pathTrace, startPointTrace, currentPointTrace],
        layout,
        {
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ["lasso3d", "select2d"]
        }
      );
    };

    void renderPlot();

    return () => {
      cancelled = true;
    };
  }, [currentPoint, history, mesh, ranges, visibleHistory]);

  useEffect(() => {
    return () => {
      const plotElement = plotElementRef.current;
      if (plotElement && plotlyRef.current) {
        plotlyRef.current.purge(plotElement);
      }
    };
  }, []);

  return (
    <div className="container page-flow">
      <section className="section">
        <div className="linear-regression-header">
          <div>
            <h1 className="linear-regression-page-title">Gradientnedstigning på MSE(w, b)</h1>
            <p className="linear-regression-page-lead">
              Vi använder samma 50 datapunkter som på regressionssidan, men visar nu hur
              gradientnedstigning stegvis rör sig på förlustytan.
            </p>
          </div>
          <Link className="btn btn-secondary" to="/visualiseringar/linjar-regression">
            Tillbaka till regression
          </Link>
        </div>
      </section>

      <main className="linear-regression-gradient-layout">
        <section className="section">
          <h2>Förlustfunktion</h2>
          <p className="lead">
            Linjär modell:
          </p>
          <div
            className="linear-regression-math-inline"
            dangerouslySetInnerHTML={modelFormulaHtml}
          />
          <p className="lead">
            Vi minimerar MSE för att hitta parametrar som passar
            data så bra som möjligt.
          </p>
          <div className="linear-regression-math-block" dangerouslySetInnerHTML={mseFormulaHtml} />
          <p className="lead">
            Gradientstegen beräknas i standardiserat rum för stabilitet, men alla visade värden är i
            originalparametrarna w och b.
          </p>

          <section className="linear-regression-control-card">
            <h3>Inställningar</h3>

            <div className="linear-regression-field-row">
              <label className="linear-regression-field linear-regression-field-compact">
                <span>Start w</span>
                <input
                  type="number"
                  step="0.001"
                  value={initialWeightInput}
                  onChange={(event) => setInitialWeightInput(event.target.value)}
                />
              </label>

              <label className="linear-regression-field linear-regression-field-compact">
                <span>Start b</span>
                <input
                  type="number"
                  step="0.01"
                  value={initialBiasInput}
                  onChange={(event) => setInitialBiasInput(event.target.value)}
                />
              </label>
            </div>

            <label className="linear-regression-field">
              <span>Steglängd (standardiserat rum)</span>
              <input
                type="number"
                step="0.01"
                min="0.001"
                value={learningRateInput}
                onChange={(event) => setLearningRateInput(event.target.value)}
              />
            </label>

            <div className="linear-regression-button-row">
              <button className="btn btn-secondary" type="button" onClick={applyStartPoint}>
                Sätt startpunkt
              </button>
              <button className="btn btn-secondary" type="button" onClick={stepBackward} disabled={stepIndex === 0}>
                Bakåt
              </button>
              <button className="btn btn-primary" type="button" onClick={stepForward}>
                Framåt
              </button>
            </div>

            <div className="linear-regression-status-grid">
              <p>
                <strong>Steg:</strong> {stepIndex}
              </p>
              <p>
                <strong>Nuvarande w:</strong> {currentPoint.weight.toFixed(6)}
              </p>
              <p>
                <strong>Nuvarande b:</strong> {currentPoint.bias.toFixed(6)}
              </p>
              <p>
                <strong>Nuvarande MSE(w, b):</strong> {currentPoint.mse.toFixed(6)}
              </p>
              <p>
                <strong>Optimal w:</strong> {optimalParameters.weight.toFixed(6)}
              </p>
              <p>
                <strong>Optimal b:</strong> {optimalParameters.bias.toFixed(6)}
              </p>
            </div>
          </section>
        </section>

        <section className="section">
          <h2>MSE(w, b) som funktionsyta</h2>
          <div ref={plotElementRef} className="linear-regression-surface" />
        </section>
      </main>
    </div>
  );
}

export default LinearRegressionGradientPage;
