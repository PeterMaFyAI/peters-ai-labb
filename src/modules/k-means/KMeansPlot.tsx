import { useEffect, useId, useMemo, useRef, useState } from "react";
import { CENTROID_COLORS, plotRange } from "./kMeansMath";
import type { ClassificationPoint, DataPoint, KMeansRun } from "./kMeansMath";

interface KMeansPlotProps {
  points: DataPoint[];
  run: KMeansRun;
  selectedClusterId: number | null;
  classificationPoint: ClassificationPoint | null;
}

function axisTicks(min: number, max: number): number[] {
  const roughStep = (max - min) / 6;
  const power = 10 ** Math.floor(Math.log10(roughStep));
  const step = [1, 2, 5, 10].find((value) => value * power >= roughStep)! * power;
  const ticks: number[] = [];
  for (let tick = Math.ceil(min / step) * step; tick <= max; tick += step) {
    ticks.push(Number(tick.toPrecision(10)));
  }
  return ticks;
}

function KMeansPlot({ points, run, selectedClusterId, classificationPoint }: KMeansPlotProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const markerId = useId().replace(/:/g, "");
  const [size, setSize] = useState({ width: 800, height: 400 });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(([entry]) => {
      const width = Math.max(240, entry.contentRect.width);
      const height = Math.max(180, entry.contentRect.height);
      setSize((previous) => Math.abs(previous.width - width) < 0.5 && Math.abs(previous.height - height) < 0.5
        ? previous
        : { width, height });
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const margins = { left: 46, right: 24, top: 25, bottom: 45 };
  const width = size.width - margins.left - margins.right;
  const height = size.height - margins.top - margins.bottom;
  const range = useMemo(() => plotRange(points, width / height), [points, width, height]);
  const toX = (x: number): number => margins.left + (x - range.xMin) / (range.xMax - range.xMin) * width;
  const toY = (y: number): number => margins.top + (range.yMax - y) / (range.yMax - range.yMin) * height;
  const bottom = margins.top + height;
  const classifiedCentroid = classificationPoint?.centroidId != null
    ? run.centroids.find((centroid) => centroid.id === classificationPoint?.centroidId)
    : undefined;
  const newPointColor = classificationPoint?.centroidId != null ? CENTROID_COLORS[classificationPoint.centroidId] : "#17202b";
  const groupCounts = run.centroids.map((centroid) =>
    run.assignments?.filter((assignment) => assignment === centroid.id).length ?? 0
  );

  return (
    <div ref={containerRef} className="kmeans-plot-container">
      <svg className="kmeans-plot" viewBox={`0 0 ${size.width} ${size.height}`} height={size.height} role="img" aria-labelledby={`${markerId}-title ${markerId}-description`}>
        <title id={`${markerId}-title`}>K-means i ett tvådimensionellt koordinatsystem</title>
        <desc id={`${markerId}-description`}>
          {points.length} datapunkter och {run.centroids.length} {run.centroids.length === 1 ? "medelpunkt" : "medelpunkter"}.
          {run.assignments === null ? " Datapunkterna är svarta och ännu inte tilldelade någon grupp." :
            ` Punkterna är färgade efter sin tilldelade medelpunkt. ${groupCounts.map((count, index) => `C${index + 1}: ${count} punkter`).join(". ")}.`}
          {selectedClusterId !== null ? ` Punkterna från genererat kluster ${selectedClusterId} markeras med en kontur.` : ""}
          {classificationPoint ? (classifiedCentroid
            ? ` En större ny datapunkt har klassificerats som C${classifiedCentroid.id + 1} och har en streckad linje till den medelpunkten.`
            : " En större svart datapunkt visar det nya exemplet som ännu inte klassificerats.") : ""}
          Axlarna har samma skala och ligger längs vänster- och nederkanten.
        </desc>
        <defs>
          {run.centroids.map((centroid) => (
            <marker key={centroid.id} id={`${markerId}-arrow-${centroid.id}`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 8 4 L 0 8 Z" fill={CENTROID_COLORS[centroid.id]} />
            </marker>
          ))}
        </defs>
        <rect x={margins.left} y={margins.top} width={width} height={height} fill="white" stroke="#e4eaf0" />
        <path d={`M ${margins.left} ${margins.top} V ${bottom} H ${margins.left + width}`} fill="none" stroke="#697b8c" strokeWidth="1.2" />
        {axisTicks(range.xMin, range.xMax).map((tick) => (
          <g key={`x-${tick}`} className="kmeans-axis-tick">
            <line x1={toX(tick)} x2={toX(tick)} y1={bottom} y2={bottom + 5} />
            <text x={toX(tick)} y={bottom + 22} textAnchor="middle">{tick.toLocaleString("sv-SE")}</text>
          </g>
        ))}
        {axisTicks(range.yMin, range.yMax).map((tick) => (
          <g key={`y-${tick}`} className="kmeans-axis-tick">
            <line x1={margins.left - 5} x2={margins.left} y1={toY(tick)} y2={toY(tick)} />
            <text x={margins.left - 10} y={toY(tick)} textAnchor="end" dominantBaseline="middle">{tick.toLocaleString("sv-SE")}</text>
          </g>
        ))}
        <text className="kmeans-axis-name" x={margins.left + width + 14} y={bottom + 4} textAnchor="middle">x</text>
        <text className="kmeans-axis-name" x={margins.left} y={margins.top - 10} textAnchor="middle">y</text>
        {points.map((point, index) => (
          <g key={point.id}>
            {selectedClusterId === point.sourceClusterId && <circle className="kmeans-selection-ring" cx={toX(point.x)} cy={toY(point.y)} r="6.1" fill="none" stroke="#718399" strokeWidth="1" />}
            <circle className="kmeans-data-point" cx={toX(point.x)} cy={toY(point.y)} r="3.2" fill={run.assignments === null ? "#17202b" : CENTROID_COLORS[run.assignments[index]]} />
          </g>
        ))}
        {classificationPoint && classifiedCentroid && (
          <line className="kmeans-classification-connection" x1={toX(classificationPoint.x)} y1={toY(classificationPoint.y)} x2={toX(classifiedCentroid.x)} y2={toY(classifiedCentroid.y)} stroke={newPointColor} strokeWidth="2" strokeDasharray="6 4" />
        )}
        {run.centroids.map((centroid, index) => {
          const x = toX(centroid.x);
          const y = toY(centroid.y);
          const previous = run.previousCentroids?.[index];
          const deltaX = previous ? x - toX(previous.x) : 0;
          const deltaY = previous ? y - toY(previous.y) : 0;
          const movement = Math.hypot(deltaX, deltaY);
          const moved = previous && movement > 2;
          const arrowX = x - deltaX / (movement || 1) * 14;
          const arrowY = y - deltaY / (movement || 1) * 14;
          const color = CENTROID_COLORS[centroid.id];
          return (
            <g key={centroid.id} className="kmeans-centroid">
              <title>C{centroid.id + 1}: ({centroid.x.toFixed(2)}, {centroid.y.toFixed(2)})</title>
              {moved && (
                <g>
                  <path d={`M ${toX(previous.x) - 5} ${toY(previous.y) - 5} l 10 10 m -10 0 l 10 -10`} stroke={color} strokeWidth="1.5" opacity="0.4" fill="none" />
                  {movement > 18 && <line x1={toX(previous.x)} y1={toY(previous.y)} x2={arrowX} y2={arrowY} stroke={color} strokeWidth="1.5" strokeDasharray="4 3" markerEnd={`url(#${markerId}-arrow-${centroid.id})`} />}
                </g>
              )}
              <path d={`M ${x - 8} ${y - 8} l 16 16 m -16 0 l 16 -16`} stroke="white" strokeWidth="6.5" strokeLinecap="round" fill="none" />
              <path d={`M ${x - 8} ${y - 8} l 16 16 m -16 0 l 16 -16`} stroke={color} strokeWidth="3.5" strokeLinecap="round" fill="none" />
              <text x={x + 12} y={y - 12} fill={color} stroke="white" strokeWidth="4" paintOrder="stroke" fontWeight="700" fontSize="13">C{centroid.id + 1}</text>
            </g>
          );
        })}
        {classificationPoint && (
          <g className="kmeans-classification-point">
            <title>Ny datapunkt{classifiedCentroid ? `: klassificerad som C${classifiedCentroid.id + 1}` : ": ännu inte klassificerad"}</title>
            <circle cx={toX(classificationPoint.x)} cy={toY(classificationPoint.y)} r="7" fill={newPointColor} stroke="white" strokeWidth="3" />
            <text x={toX(classificationPoint.x) + (toX(classificationPoint.x) > margins.left + width - 80 ? -12 : 12)} y={toY(classificationPoint.y) + (toY(classificationPoint.y) < margins.top + 24 ? 22 : -12)} textAnchor={toX(classificationPoint.x) > margins.left + width - 80 ? "end" : "start"} fill={newPointColor} stroke="white" strokeWidth="4" paintOrder="stroke" fontWeight="700" fontSize="12">Ny punkt</text>
          </g>
        )}
        {points.length === 0 && (
          <g className="kmeans-plot-empty" textAnchor="middle">
            <text x={margins.left + width / 2} y={margins.top + height / 2 - 10}>Ritytan är tom</text>
            <text x={margins.left + width / 2} y={margins.top + height / 2 + 15} fontSize="12">Generera nya kluster nedan.</text>
          </g>
        )}
      </svg>
    </div>
  );
}

export default KMeansPlot;
