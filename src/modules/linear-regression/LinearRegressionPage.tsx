import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  calculateMSE,
  calculateOptimalParameters,
  dataPoints
} from "./linearRegressionData";
import "./linearRegression.css";

const CANVAS_WIDTH = 760;
const CANVAS_HEIGHT = 520;
const MARGIN = { left: 70, right: 25, top: 20, bottom: 68 };

const PLOT_WIDTH = CANVAS_WIDTH - MARGIN.left - MARGIN.right;
const PLOT_HEIGHT = CANVAS_HEIGHT - MARGIN.top - MARGIN.bottom;

const X_TICK_STEP = 25;
const Y_TICK_STEP = 1;
const X_MIN_PLOT = 0;
const Y_MIN_PLOT = 0;
const X_MAX_PLOT = Math.ceil(Math.max(...dataPoints.map((point) => point.area)) / X_TICK_STEP) * X_TICK_STEP;
const Y_MAX_PLOT = Math.ceil(Math.max(...dataPoints.map((point) => point.price)) / Y_TICK_STEP) * Y_TICK_STEP;

function dataToCanvasX(xValue: number): number {
  return MARGIN.left + ((xValue - X_MIN_PLOT) / (X_MAX_PLOT - X_MIN_PLOT)) * PLOT_WIDTH;
}

function dataToCanvasY(yValue: number): number {
  return (
    CANVAS_HEIGHT - MARGIN.bottom - ((yValue - Y_MIN_PLOT) / (Y_MAX_PLOT - Y_MIN_PLOT)) * PLOT_HEIGHT
  );
}

function drawAxes(context: CanvasRenderingContext2D): void {
  context.strokeStyle = "#2f3142";
  context.lineWidth = 1;

  context.beginPath();
  context.moveTo(MARGIN.left, CANVAS_HEIGHT - MARGIN.bottom);
  context.lineTo(CANVAS_WIDTH - MARGIN.right, CANVAS_HEIGHT - MARGIN.bottom);
  context.moveTo(MARGIN.left, CANVAS_HEIGHT - MARGIN.bottom);
  context.lineTo(MARGIN.left, MARGIN.top);
  context.stroke();

  context.fillStyle = "#3b3f56";
  context.font = "12px Manrope, sans-serif";

  context.textAlign = "center";
  context.textBaseline = "top";
  for (let xTick = X_MIN_PLOT; xTick <= X_MAX_PLOT; xTick += X_TICK_STEP) {
    const xPosition = dataToCanvasX(xTick);
    context.beginPath();
    context.moveTo(xPosition, CANVAS_HEIGHT - MARGIN.bottom);
    context.lineTo(xPosition, CANVAS_HEIGHT - MARGIN.bottom + 6);
    context.stroke();
    context.fillText(String(xTick), xPosition, CANVAS_HEIGHT - MARGIN.bottom + 9);
  }

  context.textAlign = "right";
  context.textBaseline = "middle";
  for (let yTick = Y_MIN_PLOT; yTick <= Y_MAX_PLOT; yTick += Y_TICK_STEP) {
    const yPosition = dataToCanvasY(yTick);
    context.beginPath();
    context.moveTo(MARGIN.left - 6, yPosition);
    context.lineTo(MARGIN.left, yPosition);
    context.stroke();
    context.fillText(String(yTick), MARGIN.left - 10, yPosition);
  }

  context.fillStyle = "#2f3142";
  context.font = "14px Space Grotesk, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "alphabetic";
  context.fillText("Area (kvm)", MARGIN.left + PLOT_WIDTH / 2, CANVAS_HEIGHT - 20);

  context.save();
  context.translate(24, MARGIN.top + PLOT_HEIGHT / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("Pris (miljoner kr)", 0, 0);
  context.restore();
}

function drawDataPoints(context: CanvasRenderingContext2D): void {
  context.fillStyle = "#1c2a38";

  dataPoints.forEach((point) => {
    context.beginPath();
    context.arc(dataToCanvasX(point.area), dataToCanvasY(point.price), 3.8, 0, Math.PI * 2);
    context.fill();
  });
}

function drawRegressionLine(
  context: CanvasRenderingContext2D,
  weight: number,
  bias: number,
  strokeStyle: string,
  lineDash: number[] = []
): void {
  context.strokeStyle = strokeStyle;
  context.lineWidth = 2;
  context.setLineDash(lineDash);
  context.beginPath();
  context.moveTo(dataToCanvasX(X_MIN_PLOT), dataToCanvasY(weight * X_MIN_PLOT + bias));
  context.lineTo(dataToCanvasX(X_MAX_PLOT), dataToCanvasY(weight * X_MAX_PLOT + bias));
  context.stroke();
  context.setLineDash([]);
}

function drawErrorSegments(context: CanvasRenderingContext2D, weight: number, bias: number): void {
  context.strokeStyle = "#bf2f3a";
  context.lineWidth = 1;
  context.setLineDash([4, 4]);

  dataPoints.forEach((point) => {
    const prediction = weight * point.area + bias;
    context.beginPath();
    context.moveTo(dataToCanvasX(point.area), dataToCanvasY(point.price));
    context.lineTo(dataToCanvasX(point.area), dataToCanvasY(prediction));
    context.stroke();
  });

  context.setLineDash([]);
}

function LinearRegressionPage(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [showLine, setShowLine] = useState(false);
  const [weight, setWeight] = useState(0.08);
  const [bias, setBias] = useState(0.25);
  const [showError, setShowError] = useState(false);
  const [showOptimalLine, setShowOptimalLine] = useState(false);
  const [doPrediction, setDoPrediction] = useState(false);
  const [predictionInput, setPredictionInput] = useState("");

  const optimalParameters = useMemo(() => calculateOptimalParameters(dataPoints), []);

  const lineEquation = showLine ? `y = ${weight.toFixed(3)}x + ${bias.toFixed(2)}` : "";

  const lineMse =
    showLine && showError ? `MSE(w, b) = ${calculateMSE(weight, bias, dataPoints).toFixed(4)}` : "";

  const optimalEquation = showOptimalLine
    ? `Optimal linje: y = ${optimalParameters.weight.toFixed(3)}x + ${optimalParameters.bias.toFixed(2)}`
    : "";

  const optimalMse = showOptimalLine
    ? `MSE(w, b) = ${calculateMSE(optimalParameters.weight, optimalParameters.bias, dataPoints).toFixed(4)}`
    : "";

  const predictedPrice = useMemo(() => {
    if (!doPrediction) {
      return "";
    }

    const areaValue = Number.parseFloat(predictionInput);
    if (Number.isNaN(areaValue)) {
      return "";
    }

    const predicted = weight * areaValue + bias;
    return `Pris ~ ${predicted.toFixed(2)} miljoner kr`;
  }, [bias, doPrediction, predictionInput, weight]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    drawAxes(context);
    drawDataPoints(context);

    if (showLine) {
      drawRegressionLine(context, weight, bias, "#0d6c9d");
      if (showError) {
        drawErrorSegments(context, weight, bias);
      }
    }

    if (showOptimalLine) {
      drawRegressionLine(
        context,
        optimalParameters.weight,
        optimalParameters.bias,
        "#1f8f61",
        [9, 6]
      );
    }
  }, [bias, optimalParameters.bias, optimalParameters.weight, showError, showLine, showOptimalLine, weight]);

  return (
    <div className="container page-flow">
      <section className="section">
        <div className="linear-regression-header">
          <div>
            <h1 className="linear-regression-page-title">Linjär regression med MSE</h1>
            <p className="linear-regression-page-lead">
              Utforska hur vikt och bias påverkar regressionslinjen, felet (MSE) och prediktioner.
              När du vill gå vidare kan du visa samma problem på undersidan för gradientnedstigning.
            </p>
          </div>
          <Link className="btn btn-primary" to="/visualiseringar/linjar-regression/gradientnedstigning">
            Visa gradientnedstigning
          </Link>
        </div>
      </section>

      <main className="linear-regression-layout">
        <section className="section">
          <h2>Träningsdata för lägenhetspriser</h2>
          <div className="linear-regression-table-wrap">
            <table className="linear-regression-table">
              <thead>
                <tr>
                  <th>Index</th>
                  <th>Area (kvm)</th>
                  <th>Pris (miljoner kr)</th>
                </tr>
              </thead>
              <tbody>
                {dataPoints.map((point) => (
                  <tr key={point.index}>
                    <td>{point.index}</td>
                    <td>{point.area.toFixed(0)}</td>
                    <td>{point.price.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="section">
          <canvas
            ref={canvasRef}
            className="linear-regression-plot-canvas"
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
          />

          <div className="linear-regression-controls-grid">
            <section className="linear-regression-control-card">
              <h3>Linje</h3>

              <label className="linear-regression-row-check">
                <input
                  type="checkbox"
                  checked={showLine}
                  onChange={(event) => setShowLine(event.target.checked)}
                />
                Visa linje
              </label>

              <label className="linear-regression-field">
                <span>Vikt w</span>
                <input
                  type="range"
                  min="0"
                  max="0.1"
                  step="0.001"
                  value={weight}
                  onChange={(event) => setWeight(Number.parseFloat(event.target.value))}
                />
              </label>

              <label className="linear-regression-field">
                <span>Bias b</span>
                <input
                  type="range"
                  min="-1"
                  max="1"
                  step="0.01"
                  value={bias}
                  onChange={(event) => setBias(Number.parseFloat(event.target.value))}
                />
              </label>

              <p className="linear-regression-equation linear-regression-equation-blue">{lineEquation}</p>
            </section>

            <section className="linear-regression-control-card">
              <h3>Analys och prediktion</h3>

              <label className="linear-regression-row-check">
                <input
                  type="checkbox"
                  checked={showError}
                  onChange={(event) => setShowError(event.target.checked)}
                />
                Visa fel
              </label>

              <p className="linear-regression-equation linear-regression-equation-red">{lineMse}</p>

              <label className="linear-regression-row-check">
                <input
                  type="checkbox"
                  checked={showOptimalLine}
                  onChange={(event) => setShowOptimalLine(event.target.checked)}
                />
                Visa optimal linje
              </label>

              <p className="linear-regression-equation linear-regression-equation-green">
                {optimalEquation}
              </p>
              <p className="linear-regression-equation linear-regression-equation-red">{optimalMse}</p>

              <label className="linear-regression-row-check">
                <input
                  type="checkbox"
                  checked={doPrediction}
                  onChange={(event) => setDoPrediction(event.target.checked)}
                />
                Gör prediktion
              </label>

              {doPrediction ? (
                <div className="linear-regression-prediction">
                  <label htmlFor="predictionInput">Area (kvm)</label>
                  <input
                    id="predictionInput"
                    type="number"
                    min="0"
                    step="0.1"
                    value={predictionInput}
                    onChange={(event) => setPredictionInput(event.target.value)}
                  />
                  <p className="linear-regression-prediction-result linear-regression-equation linear-regression-equation-blue">
                    {predictedPrice}
                  </p>
                </div>
              ) : null}
            </section>
          </div>
        </section>
      </main>
    </div>
  );
}

export default LinearRegressionPage;

