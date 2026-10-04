import { useMemo, useState } from "react";
import KMeansPlot from "./KMeansPlot";
import {
  advanceRun,
  CENTROID_COLORS,
  classifyPoint,
  clusterPoints,
  covarianceMatrix,
  createDefaultClusters,
  createClassificationPoint,
  createExampleClusters,
  KMEANS_EXAMPLES,
  DEFAULT_POINTS_PER_CLUSTER,
  emptyRun,
  generateClusters,
  initializeRun
} from "./kMeansMath";
import type { ClassificationPoint, ClusterParameters, KMeansRun } from "./kMeansMath";
import "./kMeans.css";

interface ParameterSliderProps {
  label: string;
  name: keyof ClusterParameters;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (name: keyof ClusterParameters, value: number) => void;
}

const formatNumber = (value: number): string => value.toLocaleString("sv-SE", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2
});

function ParameterSlider({ label, name, value, min, max, step, onChange }: ParameterSliderProps): JSX.Element {
  const id = `kmeans-${name}`;
  return (
    <div className="kmeans-parameter">
      <div className="kmeans-parameter-label">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id}>{formatNumber(value)}</output>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(name, Number(event.target.value))} />
    </div>
  );
}

function runFeedback(run: KMeansRun, pointCount: number): string {
  const emptyMessage = run.emptyCentroidIds.length > 0
    ? ` ${run.emptyCentroidIds.map((id) => `C${id + 1}`).join(", ")} saknar punkter och behåller sin position.`
    : "";
  if (run.phase === "converged") {
    return `Stabil gruppering. Ingen punkt bytte grupp.${emptyMessage}`;
  }
  if (run.phase === "update") {
    return (run.changedPoints === null
      ? `${pointCount} punkter har fått en grupp. Flytta medelpunkterna.`
      : `${run.changedPoints} ${run.changedPoints === 1 ? "punkt" : "punkter"} bytte grupp. Flytta medelpunkterna.`) + emptyMessage;
  }
  return `Medelpunkterna har flyttats. Tilldela punkterna igen.${emptyMessage}`;
}

function KMeansPage(): JSX.Element {
  const [clusters, setClusters] = useState(createDefaultClusters);
  const [exampleCount, setExampleCount] = useState<number | null>(3);
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null);
  const [clusterCount, setClusterCount] = useState(3);
  const [pointCountInput, setPointCountInput] = useState(String(DEFAULT_POINTS_PER_CLUSTER));
  const [k, setK] = useState(3);
  const [run, setRun] = useState(emptyRun);
  const [classificationPoint, setClassificationPoint] = useState<ClassificationPoint | null>(null);
  const [feedback, setFeedback] = useState("Välj k och placera ut medelpunkterna för att börja.");

  const points = useMemo(() => clusters.flatMap(clusterPoints), [clusters]);
  const selectedCluster = clusters.find((cluster) => cluster.id === selectedClusterId);
  const covariance = selectedCluster ? covarianceMatrix(selectedCluster) : null;
  const pointCount = Number(pointCountInput);
  const validPointCount = pointCountInput.trim() !== "" && Number.isInteger(pointCount) && pointCount >= 10 && pointCount <= 200;
  const groupCounts = run.centroids.map((centroid) =>
    run.assignments?.filter((assignment) => assignment === centroid.id).length ?? 0
  );

  const handleGenerate = (): void => {
    if (!validPointCount) return;
    setClusters(generateClusters(clusterCount, pointCount));
    setExampleCount(null);
    setSelectedClusterId(null);
    setRun(emptyRun());
    setClassificationPoint(null);
    const generatedMessage = clusterCount === 1
      ? `Ett nytt punktmoln med ${pointCount} punkter har genererats.`
      : `${clusterCount} nya punktmoln med ${pointCount} punkter vardera har genererats.`;
    setFeedback(`${generatedMessage} Placera ut medelpunkter för att börja.`);
  };

  const handleParameterChange = (name: keyof ClusterParameters, value: number): void => {
    setClusters((previous) => previous.map((cluster) =>
      cluster.id === selectedClusterId ? { ...cluster, [name]: value } : cluster
    ));
    setExampleCount(null);
    setRun(emptyRun());
    setClassificationPoint(null);
    setFeedback("Data har ändrats. Placera ut nya medelpunkter.");
  };

  const handleInitialize = (): void => {
    if (points.length < k || (run.phase !== "uninitialized" && run.phase !== "converged")) return;
    setRun(initializeRun(points, k));
    setClassificationPoint(null);
    setFeedback("Medelpunkterna är på plats. Tilldela punkterna.");
    if (window.matchMedia("(min-width: 980px)").matches) {
      window.scrollTo({ top: 0, behavior: "auto" });
    }
  };

  const handleStep = (phase: "assign" | "update"): void => {
    if (run.phase !== phase || points.length < k) return;
    const next = advanceRun(points, run);
    setRun(next);
    setFeedback(runFeedback(next, points.length));
    if (window.matchMedia("(min-width: 980px)").matches) {
      window.scrollTo({ top: 0, behavior: "auto" });
    }
  };

  const handleNewPoint = (): void => {
    if (run.phase !== "converged") return;
    setClassificationPoint(createClassificationPoint(points));
    setFeedback("Ny datapunkt skapad. Klicka på Klassificera punkten.");
    if (window.matchMedia("(min-width: 980px)").matches) {
      window.scrollTo({ top: 0, behavior: "auto" });
    }
  };

  const handleClassify = (): void => {
    if (run.phase !== "converged" || classificationPoint === null || classificationPoint.centroidId !== null) return;
    const centroidId = classifyPoint(classificationPoint, run);
    setClassificationPoint({ ...classificationPoint, centroidId });
    setFeedback(`Den nya datapunkten klassificerades som C${centroidId + 1}.`);
    if (window.matchMedia("(min-width: 980px)").matches) {
      window.scrollTo({ top: 0, behavior: "auto" });
    }
  };

  const handleClear = (): void => {
    setClusters([]);
    setExampleCount(null);
    setSelectedClusterId(null);
    setRun(emptyRun());
    setClassificationPoint(null);
    setFeedback("Ritytan är tom. Välj ett exempel eller skapa nya data.");
  };

  const handleExample = (count: number): void => {
    setClusters(createExampleClusters(count));
    setExampleCount(count);
    setSelectedClusterId(null);
    setClusterCount(count);
    setPointCountInput(String(DEFAULT_POINTS_PER_CLUSTER));
    setK(count);
    setRun(emptyRun());
    setClassificationPoint(null);
    setFeedback(`Exemplet med ${count} kluster är klart. Placera ut medelpunkterna.`);
    window.scrollTo({ top: 0, behavior: "auto" });
  };

  const handleRestore = (): void => {
    handleExample(3);
    setFeedback("Startexemplet är återställt. Placera ut medelpunkterna.");
  };

  const canInitialize = points.length >= k && (run.phase === "uninitialized" || run.phase === "converged");
  const canAssign = points.length >= k && run.phase === "assign";
  const canUpdate = points.length >= k && run.phase === "update";
  const canClassify = run.phase === "converged" && classificationPoint !== null && classificationPoint.centroidId === null;

  return (
    <div className="container page-flow kmeans-page">
      <header className="kmeans-header">
        <p className="section-eyebrow">Klustring · steg för steg</p>
        <h1>K-means</h1>
        <p className="lead">Skapa punktmoln och följ hur k-means hittar grupper genom att tilldela punkter och flytta medelpunkter.</p>
      </header>

      <div className="kmeans-stage" aria-label="Demonstration av k-means">
        <section className="section kmeans-plot-section" aria-labelledby="kmeans-plot-heading">
          <div className="kmeans-plot-heading">
            <h2 id="kmeans-plot-heading">Datapunkter och medelpunkter</h2>
            <span className="kmeans-data-summary">{points.length} punkter · {clusters.length} {clusters.length === 1 ? "genererat kluster" : "genererade kluster"}</span>
          </div>
          <KMeansPlot points={points} run={run} selectedClusterId={selectedClusterId} classificationPoint={classificationPoint} />
          <div className="kmeans-legend" aria-label="Teckenförklaring">
            <span className="kmeans-legend-item"><i className="kmeans-dot" />{run.assignments === null ? "Otilldelade punkter" : "Datapunkter"}</span>
            {run.centroids.map((centroid, index) => (
              <span key={centroid.id} className="kmeans-legend-item">
                <b className="kmeans-cross" style={{ color: CENTROID_COLORS[centroid.id] }}>×</b>
                C{centroid.id + 1}{run.assignments !== null ? ` · ${groupCounts[index]} ${groupCounts[index] === 1 ? "punkt" : "punkter"}` : ""}
              </span>
            ))}
            {classificationPoint && <span className="kmeans-legend-item"><i className="kmeans-dot kmeans-new-point-dot" style={{ background: classificationPoint.centroidId === null ? "#17202b" : CENTROID_COLORS[classificationPoint.centroidId] }} />Ny datapunkt{classificationPoint.centroidId !== null ? ` · C${classificationPoint.centroidId + 1}` : ""}</span>}
            {selectedCluster && <span className="kmeans-legend-item"><i className="kmeans-ring" />Kontur: genererat kluster {selectedCluster.id}</span>}
          </div>
        </section>

        <section className="section kmeans-control-panel kmeans-algorithm-panel" aria-labelledby="kmeans-algorithm-heading">
          <div className="kmeans-panel-heading">
            <span className="kmeans-panel-icon kmeans-panel-icon-blue" aria-hidden="true">↻</span>
            <div><h2 id="kmeans-algorithm-heading">Kör k-means</h2><p>Välj antal grupper och ta ett steg i taget.</p></div>
          </div>
          <label className="search-field kmeans-k-field">
            <span>Antal medelpunkter (k)</span>
            <select value={k} onChange={(event) => {
              const nextK = Number(event.target.value);
              setK(nextK);
              setRun(emptyRun());
              setClassificationPoint(null);
              setFeedback(`k är nu ${nextK}. Placera ut medelpunkter för att börja på samma data.`);
            }}>
              {[1, 2, 3, 4, 5].map((value) => <option key={value} value={value}>{value}</option>)}
            </select>
          </label>
          <button className={`kmeans-step kmeans-initialize-button${canInitialize ? " is-active" : ""}`} type="button" aria-label="Steg 1: Placera ut medelpunkter" aria-current={canInitialize && run.phase === "uninitialized" ? "step" : undefined} disabled={!canInitialize} onClick={handleInitialize}>
            <span className="kmeans-step-number" aria-hidden="true">1</span>
            <span className="kmeans-step-title">Placera ut medelpunkter</span>
          </button>
          {points.length === 0 && <p className="kmeans-hint">Generera datapunkter för att kunna starta algoritmen.</p>}
          <div className="kmeans-step-loop" aria-label="Algoritmens återkommande steg">
            <ol className="kmeans-steps" start={2}>
              <li>
                <button className={`kmeans-step${canAssign ? " is-active" : ""}`} type="button" aria-label="Steg 2: Tilldela punkter" aria-current={canAssign ? "step" : undefined} disabled={!canAssign} onClick={() => handleStep("assign")}>
                  <span className="kmeans-step-number" aria-hidden="true">2</span>
                  <span><span className="kmeans-step-title">Tilldela punkter</span><span className="kmeans-step-description">Färga efter närmaste medelpunkt.</span></span>
                </button>
              </li>
              <li>
                <button className={`kmeans-step${canUpdate ? " is-active" : ""}`} type="button" aria-label="Steg 3: Flytta medelpunkter" aria-current={canUpdate ? "step" : undefined} disabled={!canUpdate} onClick={() => handleStep("update")}>
                  <span className="kmeans-step-number" aria-hidden="true">3</span>
                  <span><span className="kmeans-step-title">Flytta medelpunkter</span><span className="kmeans-step-description">Flytta till gruppens medelvärde.</span></span>
                </button>
              </li>
            </ol>
            <p className="kmeans-loop-caption"><span aria-hidden="true">↻</span> Upprepa steg 2 och 3 tills grupperna är stabila.</p>
          </div>
          <div className="kmeans-classification-panel" role="group" aria-label="Klassificera ett nytt exempel">
            <div className="kmeans-classification-actions">
              <button className="btn btn-secondary" type="button" disabled={run.phase !== "converged"} onClick={handleNewPoint}>Skapa ny datapunkt</button>
              <button className="btn btn-primary" type="button" disabled={!canClassify} onClick={handleClassify}>Klassificera punkten</button>
            </div>
          </div>
          <dl className="kmeans-run-stats">
            <div><dt>Genomförda varv</dt><dd>{run.iterations}</dd></div>
            <div><dt>Bytte grupp senast</dt><dd>{run.changedPoints ?? "—"}</dd></div>
          </dl>
          <p className={`kmeans-feedback${run.phase === "converged" ? " is-converged" : ""}`} role="status" aria-live="polite" aria-atomic="true">{feedback}</p>
        </section>
      </div>

      <div className="kmeans-preparation">
        <section className="section kmeans-example-panel" aria-labelledby="kmeans-examples-heading">
          <h2 id="kmeans-examples-heading">Färdiga undervisningsexempel</h2>
          <p>Välj ett genomtänkt exempel med fasta datapunkter. Samma val ger samma punktmoln varje gång.</p>
          <div className="kmeans-example-list" role="group" aria-label="Färdiga klusterexempel">
            {KMEANS_EXAMPLES.map((example) => (
              <button key={example.count} type="button" className="kmeans-example-button" aria-pressed={exampleCount === example.count} onClick={() => handleExample(example.count)}>
                <strong>{example.count} kluster</strong>
                <span>{example.description}</span>
              </button>
            ))}
          </div>
          <p className="kmeans-hint">Valet ställer k till samma antal som klustren. Du kan sedan ändra k fritt i körpanelen.</p>
        </section>
        <section className="section kmeans-control-panel" aria-labelledby="kmeans-data-heading">
          <div className="kmeans-panel-heading">
            <span className="kmeans-panel-icon" aria-hidden="true">●</span>
            <div><h2 id="kmeans-data-heading">Skapa och redigera data</h2><p>Generera 1–5 normalfördelade punktmoln i två dimensioner.</p></div>
          </div>
          <div className="kmeans-generation-fields">
            <label className="search-field">
              <span>Antal genererade kluster</span>
              <select value={clusterCount} onChange={(event) => setClusterCount(Number(event.target.value))}>
                {[1, 2, 3, 4, 5].map((count) => <option key={count} value={count}>{count}</option>)}
              </select>
            </label>
            <label className="search-field">
              <span>Punkter per kluster</span>
              <input type="number" min="10" max="200" step="1" value={pointCountInput} aria-invalid={!validPointCount} aria-describedby={!validPointCount ? "kmeans-count-error" : undefined} onChange={(event) => setPointCountInput(event.target.value)} />
            </label>
          </div>
          {!validPointCount && <p id="kmeans-count-error" className="kmeans-field-error">Ange ett heltal mellan 10 och 200.</p>}
          <button className="btn btn-primary" type="button" disabled={!validPointCount} onClick={handleGenerate}>Generera nya kluster</button>
          <p className="kmeans-hint">Nya kluster ersätter datapunkterna i ritytan.</p>

          {clusters.length > 0 && (
            <div className="kmeans-editor">
              <h3>Välj ett kluster att redigera</h3>
              <div className="kmeans-cluster-list" role="group" aria-label="Genererade kluster">
                {clusters.map((cluster) => (
                  <button key={cluster.id} type="button" className="kmeans-cluster-button" aria-pressed={selectedClusterId === cluster.id} onClick={() => setSelectedClusterId((previous) => previous === cluster.id ? null : cluster.id)}>
                    Kluster {cluster.id}<small>{cluster.samples.length} punkter</small>
                  </button>
                ))}
              </div>
              {selectedCluster && covariance ? (
                <div className="kmeans-cluster-parameters">
                  <div className="kmeans-parameter-grid">
                    <ParameterSlider label="Medelvärde x (μx)" name="meanX" value={selectedCluster.meanX} min={-8} max={8} step={0.1} onChange={handleParameterChange} />
                    <ParameterSlider label="Medelvärde y (μy)" name="meanY" value={selectedCluster.meanY} min={-8} max={8} step={0.1} onChange={handleParameterChange} />
                    <ParameterSlider label="Standardavvikelse x (σx)" name="stdX" value={selectedCluster.stdX} min={0.2} max={3} step={0.05} onChange={handleParameterChange} />
                    <ParameterSlider label="Standardavvikelse y (σy)" name="stdY" value={selectedCluster.stdY} min={0.2} max={3} step={0.05} onChange={handleParameterChange} />
                  </div>
                  <ParameterSlider label="Korrelation (ρ)" name="correlation" value={selectedCluster.correlation} min={-0.95} max={0.95} step={0.05} onChange={handleParameterChange} />
                  <div className="kmeans-covariance">
                    <span>Kovariansmatris Σ</span>
                    <table aria-label={`Kovariansmatris för kluster ${selectedCluster.id}`}><tbody>
                      {covariance.map((row, index) => <tr key={index}>{row.map((value, column) => <td key={column}>{formatNumber(value)}</td>)}</tr>)}
                    </tbody></table>
                  </div>
                  <p className="kmeans-hint">Diagonalen visar varianserna σx² och σy². Övriga värden är kovariansen ρσxσy. Samma punkter formas om när du ändrar reglagen.</p>
                </div>
              ) : <p className="kmeans-editor-placeholder">Välj ett kluster ovan för att ändra dess läge, spridning och korrelation.</p>}
            </div>
          )}

          <div className="kmeans-data-actions">
            <button className="btn btn-ghost" type="button" disabled={points.length === 0} onClick={handleClear}>Rensa datapunkter</button>
            <button className="btn btn-secondary" type="button" onClick={handleRestore}>Återställ startexempel</button>
          </div>
        </section>

      </div>
      <aside className="kmeans-explanation">
        <h2>Vad hittar k-means?</h2>
        <p>Antalet medelpunkter (k) väljs oberoende av antalet genererade kluster. Ett varv består av en tilldelning och en flytt; gruppbyten räknas från den andra tilldelningen. Algoritmen använder punkternas koordinater och euklidiskt avstånd. Den känner inte till vilka punktmoln som skapade data. En stabil gruppering kan därför skilja sig från de genererade klustren, och olika startpositioner kan ge olika resultat. Ett nytt exempel tilldelas den närmaste medelpunkten utan att medelpunkterna flyttas.</p>
      </aside>
    </div>
  );
}

export default KMeansPage;
