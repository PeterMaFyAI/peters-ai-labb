import { useEffect, useMemo, useRef, useState } from "react";
import "./kNearestNeighbors.css";

type ViewMode = "2d" | "3d";
type ClassId = 1 | 2 | 3 | 4;

interface Point2D {
  id: number;
  classId: ClassId;
  x: number;
  y: number;
}

interface Point3D extends Point2D {
  z: number;
}

interface QueryPoint2D {
  x: number;
  y: number;
}

interface QueryPoint3D extends QueryPoint2D {
  z: number;
}

interface NeighborInfo {
  id: number;
  classId: ClassId;
  distance: number;
}

interface ClassificationResult {
  neighbors: NeighborInfo[];
  counts: Record<ClassId, number>;
  radius: number;
  usedK: number;
  winner: ClassId | null;
}

interface DragGrid3D {
  x: number[];
  y: number[];
  z: number[];
}

const NEW_POINT_TRACE_NAME = "Ny datapunkt";
const DRAG_GRID_TRACE_NAME = "Draggrid";
const DEFAULT_K_VALUE = 3;
const CLASS_IDS: ClassId[] = [1, 2, 3, 4];

const CLASS_CONFIG: Record<ClassId, { label: string; color: string; highlightColor: string }> = {
  1: { label: "Klass 1", color: "#cf3a34", highlightColor: "#f4a6a3" },
  2: { label: "Klass 2", color: "#2864c7", highlightColor: "#9ec6ff" },
  3: { label: "Klass 3", color: "#c79a1b", highlightColor: "#f5e08f" },
  4: { label: "Klass 4", color: "#2a9a57", highlightColor: "#a8e2bf" }
};

const DEFAULT_CLUSTER_MEANS_BY_CLASS: Record<ClassId, { x: string; y: string }> = {
  1: { x: "-2", y: "0" },
  2: { x: "2", y: "0" },
  3: { x: "0", y: "-4" },
  4: { x: "-3", y: "3" }
};

const DEFAULT_2D_POINTS: Point2D[] = [
  { id: 1, classId: 1, x: -1.543, y: -1.56 },
  { id: 2, classId: 1, x: -0.874, y: 1.411 },
  { id: 3, classId: 1, x: -4.927, y: -1.953 },
  { id: 4, classId: 1, x: -1.808, y: -0.474 },
  { id: 5, classId: 1, x: -2.025, y: -1.28 },
  { id: 6, classId: 1, x: -0.681, y: 1.167 },
  { id: 7, classId: 1, x: -1.901, y: 1.691 },
  { id: 8, classId: 1, x: -1.299, y: -1.289 },
  { id: 9, classId: 1, x: -1.447, y: -1.438 },
  { id: 10, classId: 1, x: -0.682, y: -0.075 },
  { id: 11, classId: 1, x: -2.277, y: -1.021 },
  { id: 12, classId: 1, x: -0.166, y: -0.232 },
  { id: 13, classId: 1, x: -2.642, y: -0.528 },
  { id: 14, classId: 1, x: -1.202, y: 0.548 },
  { id: 15, classId: 1, x: -1.381, y: 0.646 },
  { id: 16, classId: 1, x: 1.212, y: -0.61 },
  { id: 17, classId: 1, x: -2.768, y: -1.221 },
  { id: 18, classId: 1, x: -1.076, y: 1.693 },
  { id: 19, classId: 1, x: -2.171, y: -1.26 },
  { id: 20, classId: 1, x: -3.237, y: 0.976 },
  { id: 21, classId: 2, x: 3.115, y: 0.815 },
  { id: 22, classId: 2, x: 1.002, y: 0.348 },
  { id: 23, classId: 2, x: 2.175, y: 0.328 },
  { id: 24, classId: 2, x: 3.307, y: 0.335 },
  { id: 25, classId: 2, x: 3.018, y: 0.101 },
  { id: 26, classId: 2, x: 2.434, y: 0.947 },
  { id: 27, classId: 2, x: -0.186, y: -0.48 },
  { id: 28, classId: 2, x: 1.294, y: -0.958 },
  { id: 29, classId: 2, x: 1.587, y: 2.242 },
  { id: 30, classId: 2, x: 0.701, y: 1.452 },
  { id: 31, classId: 2, x: -0.524, y: -0.502 },
  { id: 32, classId: 2, x: 2.244, y: 0.879 },
  { id: 33, classId: 2, x: 3.067, y: 1.19 },
  { id: 34, classId: 2, x: 1.477, y: -0.694 },
  { id: 35, classId: 2, x: 3.287, y: -0.287 },
  { id: 36, classId: 2, x: 0.086, y: -1.7 },
  { id: 37, classId: 2, x: 0.621, y: 0.746 },
  { id: 38, classId: 2, x: 2.214, y: 1.036 },
  { id: 39, classId: 2, x: 1.359, y: 0.238 },
  { id: 40, classId: 2, x: 2.938, y: -0.464 }
];

const DEFAULT_3D_POINTS: Point3D[] = [
  { id: 1, classId: 1, x: -0.543, y: -1.06, z: 0.626 },
  { id: 2, classId: 1, x: 0.411, y: -2.427, z: -2.453 },
  { id: 3, classId: 1, x: -0.808, y: 0.026, z: -0.525 },
  { id: 4, classId: 1, x: -2.28, y: 1.819, z: 0.667 },
  { id: 5, classId: 1, x: -0.901, y: 2.191, z: 0.201 },
  { id: 6, classId: 1, x: -2.289, y: 1.053, z: -1.938 },
  { id: 7, classId: 1, x: 0.318, y: 0.425, z: -0.777 },
  { id: 8, classId: 1, x: -2.021, y: 2.334, z: -0.732 },
  { id: 9, classId: 1, x: -1.642, y: -0.028, z: 0.298 },
  { id: 10, classId: 1, x: -0.452, y: 1.119, z: 0.146 },
  { id: 11, classId: 1, x: 2.212, y: -0.11, z: -1.268 },
  { id: 12, classId: 1, x: -2.221, y: 1.424, z: 1.193 },
  { id: 13, classId: 1, x: -1.171, y: -0.76, z: -1.737 },
  { id: 14, classId: 1, x: -0.024, y: 1.615, z: 0.315 },
  { id: 15, classId: 1, x: -1.998, y: 0.848, z: -0.325 },
  { id: 16, classId: 1, x: -0.672, y: 1.807, z: -0.165 },
  { id: 17, classId: 1, x: 0.018, y: 0.601, z: -0.066 },
  { id: 18, classId: 1, x: -0.053, y: -1.686, z: -0.98 },
  { id: 19, classId: 1, x: -1.706, y: -0.458, z: -0.913 },
  { id: 20, classId: 1, x: 1.242, y: -0.799, z: 0.952 },
  { id: 21, classId: 2, x: 0.476, y: -0.002, z: -0.256 },
  { id: 22, classId: 2, x: 3.879, y: 1.567, z: 0.69 },
  { id: 23, classId: 2, x: 2.477, y: -0.194, z: 0.787 },
  { id: 24, classId: 2, x: 2.713, y: -1.414, z: -2.2 },
  { id: 25, classId: 2, x: 1.621, y: 1.246, z: -0.286 },
  { id: 26, classId: 2, x: 4.036, y: -0.141, z: -0.262 },
  { id: 27, classId: 2, x: 3.938, y: 0.036, z: 0.185 },
  { id: 28, classId: 2, x: 2.007, y: -0.045, z: -1.073 },
  { id: 29, classId: 2, x: 1.206, y: 1.23, z: -1.204 },
  { id: 30, classId: 2, x: 3.019, y: 1.221, z: 0.17 },
  { id: 31, classId: 2, x: 3.998, y: 0.352, z: -1.135 },
  { id: 32, classId: 2, x: 2.88, y: -2.031, z: -2.671 },
  { id: 33, classId: 2, x: 1.016, y: -0.996, z: 0.1 },
  { id: 34, classId: 2, x: 1.642, y: -0.067, z: 1.449 },
  { id: 35, classId: 2, x: 2.466, y: 1.606, z: -1.9 },
  { id: 36, classId: 2, x: 2.692, y: -0.925, z: -1.009 },
  { id: 37, classId: 2, x: 4.26, y: -2.091, z: 0.152 },
  { id: 38, classId: 2, x: 3.357, y: -0.391, z: -2.669 },
  { id: 39, classId: 2, x: 3.108, y: -0.294, z: -0.151 },
  { id: 40, classId: 2, x: 3.033, y: 2.903, z: -0.859 }
];

function createInitialCounts(): Record<ClassId, number> {
  return { 1: 0, 2: 0, 3: 0, 4: 0 };
}

function parseFiniteNumber(value: string): number | null {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseIntegerInRange(value: string, min: number, max: number, fallback: number): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function toFixedInput(value: number): string {
  return value.toFixed(3);
}

function nextId(points: Array<{ id: number }>): number {
  if (points.length === 0) {
    return 1;
  }
  return points.reduce((maxId, point) => Math.max(maxId, point.id), points[0].id) + 1;
}

function sampleNormalDistribution(mean: number, standardDeviation: number): number {
  let u1 = 0;
  let u2 = 0;

  while (u1 === 0) {
    u1 = Math.random();
  }
  while (u2 === 0) {
    u2 = Math.random();
  }

  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + standardDeviation * z0;
}

function resolveWinner(neighbors: NeighborInfo[], counts: Record<ClassId, number>): ClassId | null {
  const maxCount = Math.max(...CLASS_IDS.map((classId) => counts[classId]));
  if (maxCount <= 0) {
    return null;
  }

  const candidates = CLASS_IDS.filter((classId) => counts[classId] === maxCount);
  if (candidates.length === 1) {
    return candidates[0];
  }

  let winner = candidates[0];
  let bestAverageDistance = Number.POSITIVE_INFINITY;

  candidates.forEach((classId) => {
    const classDistances = neighbors
      .filter((neighbor) => neighbor.classId === classId)
      .map((neighbor) => neighbor.distance);
    const averageDistance =
      classDistances.reduce((sum, distance) => sum + distance, 0) / classDistances.length;

    if (averageDistance < bestAverageDistance - 1e-12) {
      bestAverageDistance = averageDistance;
      winner = classId;
      return;
    }

    if (Math.abs(averageDistance - bestAverageDistance) <= 1e-12 && classId < winner) {
      winner = classId;
    }
  });

  return winner;
}

function classify2D(points: Point2D[], queryPoint: QueryPoint2D, k: number): ClassificationResult {
  const rankedNeighbors = points
    .map<NeighborInfo>((point) => ({
      id: point.id,
      classId: point.classId,
      distance: Math.hypot(queryPoint.x - point.x, queryPoint.y - point.y)
    }))
    .sort((pointA, pointB) => pointA.distance - pointB.distance);

  const usedK = Math.min(k, rankedNeighbors.length);
  const neighbors = rankedNeighbors.slice(0, usedK);
  const counts = createInitialCounts();
  neighbors.forEach((neighbor) => {
    counts[neighbor.classId] += 1;
  });

  return {
    neighbors,
    counts,
    radius: neighbors.length > 0 ? neighbors[neighbors.length - 1].distance : 0,
    usedK,
    winner: resolveWinner(neighbors, counts)
  };
}

function classify3D(points: Point3D[], queryPoint: QueryPoint3D, k: number): ClassificationResult {
  const rankedNeighbors = points
    .map<NeighborInfo>((point) => ({
      id: point.id,
      classId: point.classId,
      distance: Math.sqrt(
        (queryPoint.x - point.x) ** 2 +
          (queryPoint.y - point.y) ** 2 +
          (queryPoint.z - point.z) ** 2
      )
    }))
    .sort((pointA, pointB) => pointA.distance - pointB.distance);

  const usedK = Math.min(k, rankedNeighbors.length);
  const neighbors = rankedNeighbors.slice(0, usedK);
  const counts = createInitialCounts();
  neighbors.forEach((neighbor) => {
    counts[neighbor.classId] += 1;
  });

  return {
    neighbors,
    counts,
    radius: neighbors.length > 0 ? neighbors[neighbors.length - 1].distance : 0,
    usedK,
    winner: resolveWinner(neighbors, counts)
  };
}

function buildCircleTrace2D(center: QueryPoint2D, radius: number): { x: number[]; y: number[] } {
  const segmentCount = 120;
  const xValues: number[] = [];
  const yValues: number[] = [];

  for (let segment = 0; segment <= segmentCount; segment += 1) {
    const theta = (2 * Math.PI * segment) / segmentCount;
    xValues.push(center.x + radius * Math.cos(theta));
    yValues.push(center.y + radius * Math.sin(theta));
  }

  return { x: xValues, y: yValues };
}

function buildSphereContourTraces(queryPoint: QueryPoint3D, radius: number): any[] {
  const segmentCount = 90;
  const planes: Array<"xy" | "xz" | "yz"> = ["xy", "xz", "yz"];

  return planes.map((plane, index) => {
    const xValues: number[] = [];
    const yValues: number[] = [];
    const zValues: number[] = [];

    for (let segment = 0; segment <= segmentCount; segment += 1) {
      const theta = (2 * Math.PI * segment) / segmentCount;

      if (plane === "xy") {
        xValues.push(queryPoint.x + radius * Math.cos(theta));
        yValues.push(queryPoint.y + radius * Math.sin(theta));
        zValues.push(queryPoint.z);
      } else if (plane === "xz") {
        xValues.push(queryPoint.x + radius * Math.cos(theta));
        yValues.push(queryPoint.y);
        zValues.push(queryPoint.z + radius * Math.sin(theta));
      } else {
        xValues.push(queryPoint.x);
        yValues.push(queryPoint.y + radius * Math.cos(theta));
        zValues.push(queryPoint.z + radius * Math.sin(theta));
      }
    }

    return {
      type: "scatter3d",
      mode: "lines",
      name: index === 0 ? "K-radie" : "K-radie kontur",
      x: xValues,
      y: yValues,
      z: zValues,
      showlegend: index === 0,
      hoverinfo: "skip",
      line: { color: "#151515", width: 5, dash: "dash" }
    };
  });
}

function computeRange2D(
  points: Point2D[],
  queryPoint: QueryPoint2D | null,
  radius: number | null
): { x: [number, number]; y: [number, number] } {
  const xValues = points.map((point) => point.x);
  const yValues = points.map((point) => point.y);

  if (queryPoint) {
    if (radius && radius > 0) {
      xValues.push(queryPoint.x - radius, queryPoint.x + radius);
      yValues.push(queryPoint.y - radius, queryPoint.y + radius);
    } else {
      xValues.push(queryPoint.x);
      yValues.push(queryPoint.y);
    }
  }

  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);

  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const chartSpan = Math.max(spanX, spanY, 4);
  const padding = Math.max(0.55, 0.16 * chartSpan);

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const half = chartSpan / 2 + padding;

  return {
    x: [centerX - half, centerX + half],
    y: [centerY - half, centerY + half]
  };
}

function computeRange3D(
  points: Point3D[],
  queryPoint: QueryPoint3D | null,
  radius: number | null
): { x: [number, number]; y: [number, number]; z: [number, number] } {
  const xValues = points.map((point) => point.x);
  const yValues = points.map((point) => point.y);
  const zValues = points.map((point) => point.z);

  if (queryPoint) {
    if (radius && radius > 0) {
      xValues.push(queryPoint.x - radius, queryPoint.x + radius);
      yValues.push(queryPoint.y - radius, queryPoint.y + radius);
      zValues.push(queryPoint.z - radius, queryPoint.z + radius);
    } else {
      xValues.push(queryPoint.x);
      yValues.push(queryPoint.y);
      zValues.push(queryPoint.z);
    }
  }

  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const minZ = Math.min(...zValues);
  const maxZ = Math.max(...zValues);

  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const spanZ = Math.max(1e-6, maxZ - minZ);

  const padX = Math.max(0.45, spanX * 0.2);
  const padY = Math.max(0.45, spanY * 0.2);
  const padZ = Math.max(0.45, spanZ * 0.2);

  return {
    x: [minX - padX, maxX + padX],
    y: [minY - padY, maxY + padY],
    z: [minZ - padZ, maxZ + padZ]
  };
}

function build3DDragGrid(range3d: {
  x: [number, number];
  y: [number, number];
  z: [number, number];
}): DragGrid3D {
  const steps = 9;
  const xValues: number[] = [];
  const yValues: number[] = [];
  const zValues: number[] = [];

  for (let xi = 0; xi < steps; xi += 1) {
    const x = range3d.x[0] + ((range3d.x[1] - range3d.x[0]) * xi) / (steps - 1);
    for (let yi = 0; yi < steps; yi += 1) {
      const y = range3d.y[0] + ((range3d.y[1] - range3d.y[0]) * yi) / (steps - 1);
      for (let zi = 0; zi < steps; zi += 1) {
        const z = range3d.z[0] + ((range3d.z[1] - range3d.z[0]) * zi) / (steps - 1);
        xValues.push(x);
        yValues.push(y);
        zValues.push(z);
      }
    }
  }

  return { x: xValues, y: yValues, z: zValues };
}

function extract3DPoint(rawPoint: any): QueryPoint3D | null {
  const x = Number(rawPoint?.x);
  const y = Number(rawPoint?.y);
  const z = Number(rawPoint?.z);

  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    return null;
  }

  return { x, y, z };
}

function read2DPointFromMouseEvent(event: MouseEvent, plotElement: HTMLDivElement): QueryPoint2D | null {
  const fullLayout = (plotElement as any)._fullLayout;
  const xAxis = fullLayout?.xaxis;
  const yAxis = fullLayout?.yaxis;

  if (!xAxis || !yAxis) {
    return null;
  }

  const containerRect = plotElement.getBoundingClientRect();
  const xPixel = event.clientX - containerRect.left;
  const yPixel = event.clientY - containerRect.top;

  const xRelative = xPixel - xAxis._offset;
  const yRelative = yPixel - yAxis._offset;

  if (
    xRelative < 0 ||
    yRelative < 0 ||
    xRelative > xAxis._length ||
    yRelative > yAxis._length
  ) {
    return null;
  }

  const x = xAxis.p2l(xRelative);
  const y = yAxis.p2l(yRelative);

  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return null;
  }

  return { x, y };
}

function KNearestNeighborsPage(): JSX.Element {
  const plotElementRef = useRef<HTMLDivElement | null>(null);
  const plotlyRef = useRef<any>(null);
  const cameraRef = useRef({ eye: { x: 1.45, y: 1.35, z: 0.9 } });
  const dragging3dRef = useRef(false);

  const [viewMode, setViewMode] = useState<ViewMode>("2d");

  const [points2d, setPoints2d] = useState<Point2D[]>(() =>
    DEFAULT_2D_POINTS.map((point) => ({ ...point }))
  );
  const [points3d, setPoints3d] = useState<Point3D[]>(() =>
    DEFAULT_3D_POINTS.map((point) => ({ ...point }))
  );

  const [queryPoint2d, setQueryPoint2d] = useState<QueryPoint2D | null>(null);
  const [queryPoint3d, setQueryPoint3d] = useState<QueryPoint3D | null>(null);

  const [queryInputs2d, setQueryInputs2d] = useState({ x: "0", y: "0" });
  const [queryInputs3d, setQueryInputs3d] = useState({ x: "0", y: "0", z: "0" });

  const [classification2d, setClassification2d] = useState<ClassificationResult | null>(null);
  const [classification3d, setClassification3d] = useState<ClassificationResult | null>(null);

  const [kInput, setKInput] = useState(String(DEFAULT_K_VALUE));

  const [clusterClassInput, setClusterClassInput] = useState<ClassId>(3);
  const [clusterCountInput, setClusterCountInput] = useState("15");
  const [clusterMeanInputs, setClusterMeanInputs] = useState(() => ({
    ...DEFAULT_CLUSTER_MEANS_BY_CLASS[3],
    z: "0"
  }));
  const [clusterStdInput, setClusterStdInput] = useState("0.8");

  const [feedback, setFeedback] = useState("");

  const activeClassification = viewMode === "2d" ? classification2d : classification3d;
  const activeRange2d = useMemo(
    () => computeRange2D(points2d, queryPoint2d, classification2d?.radius ?? null),
    [points2d, queryPoint2d, classification2d]
  );
  const activeRange3d = useMemo(
    () => computeRange3D(points3d, queryPoint3d, classification3d?.radius ?? null),
    [points3d, queryPoint3d, classification3d]
  );
  const dragGrid3d = useMemo(() => build3DDragGrid(activeRange3d), [activeRange3d]);
  const maxKValue = Math.max(1, viewMode === "2d" ? points2d.length : points3d.length);

  const setQueryAndInputs2d = (queryPoint: QueryPoint2D): void => {
    setQueryPoint2d(queryPoint);
    setQueryInputs2d({
      x: toFixedInput(queryPoint.x),
      y: toFixedInput(queryPoint.y)
    });
    setClassification2d(null);
  };

  const setQueryAndInputs3d = (queryPoint: QueryPoint3D): void => {
    setQueryPoint3d(queryPoint);
    setQueryInputs3d({
      x: toFixedInput(queryPoint.x),
      y: toFixedInput(queryPoint.y),
      z: toFixedInput(queryPoint.z)
    });
    setClassification3d(null);
  };

  const toggleViewMode = (): void => {
    setViewMode((previousMode) => (previousMode === "2d" ? "3d" : "2d"));
    setFeedback("");
  };

  const handleAddQueryPoint = (): void => {
    if (viewMode === "2d") {
      const x = parseFiniteNumber(queryInputs2d.x);
      const y = parseFiniteNumber(queryInputs2d.y);

      if (x === null || y === null) {
        setFeedback("Ange giltiga x- och y-koordinater för den nya punkten.");
        return;
      }

      setQueryAndInputs2d({ x, y });
      setFeedback("Ny datapunkt tillagd i 2D.");
      return;
    }

    const x = parseFiniteNumber(queryInputs3d.x);
    const y = parseFiniteNumber(queryInputs3d.y);
    const z = parseFiniteNumber(queryInputs3d.z);

    if (x === null || y === null || z === null) {
      setFeedback("Ange giltiga x-, y- och z-koordinater för den nya punkten.");
      return;
    }

    setQueryAndInputs3d({ x, y, z });
    setFeedback("Ny datapunkt tillagd i 3D.");
  };

  const handleClassifyPoint = (): void => {
    const safeK = parseIntegerInRange(
      kInput,
      1,
      maxKValue,
      Math.min(DEFAULT_K_VALUE, maxKValue)
    );
    setKInput(String(safeK));

    if (viewMode === "2d") {
      if (!queryPoint2d) {
        setFeedback("Lägg till en ny datapunkt i 2D innan klassificering.");
        return;
      }
      const result = classify2D(points2d, queryPoint2d, safeK);
      setClassification2d(result);
      setFeedback(`Klassificering klar i 2D med k = ${result.usedK}.`);
      return;
    }

    if (!queryPoint3d) {
      setFeedback("Lägg till en ny datapunkt i 3D innan klassificering.");
      return;
    }

    const result = classify3D(points3d, queryPoint3d, safeK);
    setClassification3d(result);
    setFeedback(`Klassificering klar i 3D med k = ${result.usedK}.`);
  };

  const handleGenerateCluster = (): void => {
    const count = parseIntegerInRange(clusterCountInput, 1, 300, 15);
    const standardDeviation = parseFiniteNumber(clusterStdInput);
    const meanX = parseFiniteNumber(clusterMeanInputs.x);
    const meanY = parseFiniteNumber(clusterMeanInputs.y);
    const meanZ = parseFiniteNumber(clusterMeanInputs.z);

    if (standardDeviation === null || standardDeviation <= 0) {
      setFeedback("Standardavvikelsen måste vara ett positivt tal.");
      return;
    }

    if (meanX === null || meanY === null) {
      setFeedback("Ange giltiga medelvärden för x och y.");
      return;
    }

    if (viewMode === "2d") {
      setPoints2d((previousPoints) => {
        let idCounter = nextId(previousPoints);
        const generatedPoints: Point2D[] = Array.from({ length: count }, () => {
          const generatedPoint: Point2D = {
            id: idCounter,
            classId: clusterClassInput,
            x: sampleNormalDistribution(meanX, standardDeviation),
            y: sampleNormalDistribution(meanY, standardDeviation)
          };
          idCounter += 1;
          return generatedPoint;
        });

        return [...previousPoints, ...generatedPoints];
      });
      setClassification2d(null);
      setFeedback(
        `Genererade ${count} nya 2D-punkter i ${CLASS_CONFIG[clusterClassInput].label}.`
      );
      return;
    }

    if (meanZ === null) {
      setFeedback("Ange ett giltigt medelvärde för z i 3D.");
      return;
    }

    setPoints3d((previousPoints) => {
      let idCounter = nextId(previousPoints);
      const generatedPoints: Point3D[] = Array.from({ length: count }, () => {
        const generatedPoint: Point3D = {
          id: idCounter,
          classId: clusterClassInput,
          x: sampleNormalDistribution(meanX, standardDeviation),
          y: sampleNormalDistribution(meanY, standardDeviation),
          z: sampleNormalDistribution(meanZ, standardDeviation)
        };
        idCounter += 1;
        return generatedPoint;
      });

      return [...previousPoints, ...generatedPoints];
    });
    setClassification3d(null);
    setFeedback(`Genererade ${count} nya 3D-punkter i ${CLASS_CONFIG[clusterClassInput].label}.`);
  };

  const handleResetAll = (): void => {
    setViewMode("2d");
    setPoints2d(DEFAULT_2D_POINTS.map((point) => ({ ...point })));
    setPoints3d(DEFAULT_3D_POINTS.map((point) => ({ ...point })));
    setQueryPoint2d(null);
    setQueryPoint3d(null);
    setClassification2d(null);
    setClassification3d(null);
    setQueryInputs2d({ x: "0", y: "0" });
    setQueryInputs3d({ x: "0", y: "0", z: "0" });
    setKInput(String(DEFAULT_K_VALUE));
    setClusterClassInput(3);
    setClusterCountInput("15");
    setClusterMeanInputs({ ...DEFAULT_CLUSTER_MEANS_BY_CLASS[3], z: "0" });
    setClusterStdInput("0.8");
    setFeedback("Diagrammen är återställda till ursprungsdata.");
  };

  useEffect(() => {
    let cancelled = false;
    const plotElement = plotElementRef.current;

    if (!plotElement) {
      return;
    }

    let onPlotClick2d: ((event: MouseEvent) => void) | null = null;
    let onMouseUp: (() => void) | null = null;

    const renderPlot = async (): Promise<void> => {
      if (!plotlyRef.current) {
        const plotlyModule = await import("plotly.js-dist-min");
        if (cancelled) {
          return;
        }
        plotlyRef.current = plotlyModule.default;
      }

      const Plotly = plotlyRef.current;
      const graphDiv = plotElement as any;

      graphDiv.removeAllListeners?.("plotly_click");
      graphDiv.removeAllListeners?.("plotly_hover");

      if (viewMode === "2d") {
        const traces: any[] = [];

        CLASS_IDS.forEach((classId) => {
          const classPoints = points2d.filter((point) => point.classId === classId);
          if (classPoints.length === 0) {
            return;
          }

          traces.push({
            type: "scatter",
            mode: "markers",
            name: CLASS_CONFIG[classId].label,
            x: classPoints.map((point) => point.x),
            y: classPoints.map((point) => point.y),
            marker: {
              color: CLASS_CONFIG[classId].color,
              size: 10,
              line: {
                color: "#ffffff",
                width: 0.8
              }
            },
            hovertemplate: "x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"
          });
        });

        if (queryPoint2d && classification2d && classification2d.radius > 0) {
          const circle = buildCircleTrace2D(queryPoint2d, classification2d.radius);
          traces.push({
            type: "scatter",
            mode: "lines",
            name: "K-radie",
            x: circle.x,
            y: circle.y,
            hoverinfo: "skip",
            line: {
              color: "#151515",
              width: 2,
              dash: "dash"
            }
          });
        }

        if (classification2d) {
          CLASS_IDS.forEach((classId) => {
            const neighborIds = new Set(
              classification2d.neighbors
                .filter((neighbor) => neighbor.classId === classId)
                .map((neighbor) => neighbor.id)
            );

            if (neighborIds.size === 0) {
              return;
            }

            const highlightedPoints = points2d.filter((point) => neighborIds.has(point.id));

            traces.push({
              type: "scatter",
              mode: "markers",
              name: `${CLASS_CONFIG[classId].label} i k-grannar`,
              showlegend: false,
              x: highlightedPoints.map((point) => point.x),
              y: highlightedPoints.map((point) => point.y),
              marker: {
                symbol: "circle-open",
                size: 18,
                color: CLASS_CONFIG[classId].highlightColor,
                line: {
                  color: CLASS_CONFIG[classId].highlightColor,
                  width: 3
                }
              },
              hoverinfo: "skip"
            });
          });
        }

        if (queryPoint2d) {
          traces.push({
            type: "scatter",
            mode: "markers",
            name: NEW_POINT_TRACE_NAME,
            x: [queryPoint2d.x],
            y: [queryPoint2d.y],
            marker: {
              size: 13,
              symbol: "x",
              color: classification2d?.winner ? CLASS_CONFIG[classification2d.winner].color : "#111111",
              line: {
                color: "#111111",
                width: 1.2
              }
            },
            hovertemplate: "Ny punkt<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"
          });
        }

        await Plotly.react(
          plotElement,
          traces,
          {
            margin: { l: 55, r: 10, t: 10, b: 50 },
            paper_bgcolor: "#ffffff",
            plot_bgcolor: "#f8fbfe",
            hovermode: "closest",
            legend: {
              orientation: "h",
              y: 1.12
            },
            xaxis: {
              title: "x",
              range: activeRange2d.x,
              fixedrange: true,
              zeroline: true,
              zerolinewidth: 1.5,
              zerolinecolor: "#9fb3c7",
              gridcolor: "#dde7f1"
            },
            yaxis: {
              title: "y",
              range: activeRange2d.y,
              fixedrange: true,
              scaleanchor: "x",
              scaleratio: 1,
              zeroline: true,
              zerolinewidth: 1.5,
              zerolinecolor: "#9fb3c7",
              gridcolor: "#dde7f1"
            }
          },
          {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: [
              "lasso2d",
              "select2d",
              "zoom2d",
              "pan2d",
              "autoScale2d",
              "resetScale2d"
            ],
            scrollZoom: false
          }
        );

        onPlotClick2d = (event: MouseEvent): void => {
          const queryPoint = read2DPointFromMouseEvent(event, plotElement);
          if (!queryPoint) {
            return;
          }

          setQueryAndInputs2d(queryPoint);
          setFeedback("Ny datapunkt satt via klick i 2D-diagrammet.");
        };

        plotElement.addEventListener("click", onPlotClick2d);
        return;
      }

      const layoutCamera = graphDiv.layout?.scene?.camera;
      if (layoutCamera) {
        cameraRef.current = layoutCamera;
      }

      const traces: any[] = [];

      CLASS_IDS.forEach((classId) => {
        const classPoints = points3d.filter((point) => point.classId === classId);
        if (classPoints.length === 0) {
          return;
        }

        traces.push({
          type: "scatter3d",
          mode: "markers",
          name: CLASS_CONFIG[classId].label,
          x: classPoints.map((point) => point.x),
          y: classPoints.map((point) => point.y),
          z: classPoints.map((point) => point.z),
          marker: {
            color: CLASS_CONFIG[classId].color,
            size: 4.5
          },
          hovertemplate: "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>"
        });
      });

      if (classification3d) {
        CLASS_IDS.forEach((classId) => {
          const neighborIds = new Set(
            classification3d.neighbors
              .filter((neighbor) => neighbor.classId === classId)
              .map((neighbor) => neighbor.id)
          );

          if (neighborIds.size === 0) {
            return;
          }

          const highlightedPoints = points3d.filter((point) => neighborIds.has(point.id));
          traces.push({
            type: "scatter3d",
            mode: "markers",
            name: `${CLASS_CONFIG[classId].label} i k-grannar`,
            showlegend: false,
            x: highlightedPoints.map((point) => point.x),
            y: highlightedPoints.map((point) => point.y),
            z: highlightedPoints.map((point) => point.z),
            marker: {
              symbol: "circle",
              size: 8.5,
              color: CLASS_CONFIG[classId].highlightColor,
              line: {
                color: CLASS_CONFIG[classId].highlightColor,
                width: 1
              }
            },
            hoverinfo: "skip"
          });
        });
      }

      if (queryPoint3d) {
        traces.push({
          type: "scatter3d",
          mode: "markers",
          name: NEW_POINT_TRACE_NAME,
          x: [queryPoint3d.x],
          y: [queryPoint3d.y],
          z: [queryPoint3d.z],
          marker: {
            symbol: "x",
            size: 9,
            color: classification3d?.winner ? CLASS_CONFIG[classification3d.winner].color : "#111111",
            line: {
              color: "#111111",
              width: 1.5
            }
          },
          hovertemplate: "Ny punkt<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>"
        });
      }

      if (queryPoint3d && classification3d && classification3d.radius > 0) {
        traces.push(...buildSphereContourTraces(queryPoint3d, classification3d.radius));
      }

      traces.push({
        type: "scatter3d",
        mode: "markers",
        name: DRAG_GRID_TRACE_NAME,
        showlegend: false,
        x: dragGrid3d.x,
        y: dragGrid3d.y,
        z: dragGrid3d.z,
        marker: {
          size: 2,
          color: "rgba(90, 108, 126, 0.1)"
        },
        hovertemplate: "<extra></extra>"
      });

      await Plotly.react(
        plotElement,
        traces,
        {
          margin: { l: 0, r: 0, t: 10, b: 0 },
          paper_bgcolor: "#ffffff",
          uirevision: "knn-3d-camera",
          legend: {
            orientation: "h",
            y: 1.05
          },
          scene: {
            uirevision: "knn-3d-camera",
            xaxis: { title: "x", range: activeRange3d.x, backgroundcolor: "#f8fbfe" },
            yaxis: { title: "y", range: activeRange3d.y, backgroundcolor: "#f8fbfe" },
            zaxis: { title: "z", range: activeRange3d.z, backgroundcolor: "#f8fbfe" },
            camera: cameraRef.current
          }
        },
        {
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ["lasso2d", "select2d"]
        }
      );

      graphDiv.on("plotly_click", (eventData: any) => {
        const rawPoint = eventData?.points?.[0];
        const clickedPoint = extract3DPoint(rawPoint);
        if (!clickedPoint) {
          return;
        }

        setQueryAndInputs3d(clickedPoint);

        const traceName = String(rawPoint?.data?.name ?? "");
        dragging3dRef.current =
          traceName === NEW_POINT_TRACE_NAME || traceName === DRAG_GRID_TRACE_NAME;

        if (traceName === DRAG_GRID_TRACE_NAME || traceName === NEW_POINT_TRACE_NAME) {
          setFeedback("Dra musen över draggriden för att flytta punkten i 3D.");
        } else {
          setFeedback("Ny datapunkt satt via klick i 3D-diagrammet.");
        }
      });

      graphDiv.on("plotly_hover", (eventData: any) => {
        if (!dragging3dRef.current) {
          return;
        }

        const rawPoint = eventData?.points?.[0];
        const traceName = String(rawPoint?.data?.name ?? "");
        if (traceName !== DRAG_GRID_TRACE_NAME) {
          return;
        }

        const hoveredPoint = extract3DPoint(rawPoint);
        if (!hoveredPoint) {
          return;
        }

        setQueryAndInputs3d(hoveredPoint);
      });

      onMouseUp = () => {
        dragging3dRef.current = false;
      };
      window.addEventListener("mouseup", onMouseUp);
    };

    void renderPlot();

    return () => {
      cancelled = true;
      dragging3dRef.current = false;

      if (onPlotClick2d) {
        plotElement.removeEventListener("click", onPlotClick2d);
      }

      if (onMouseUp) {
        window.removeEventListener("mouseup", onMouseUp);
      }
    };
  }, [
    activeRange2d,
    activeRange3d,
    classification2d,
    classification3d,
    dragGrid3d,
    points2d,
    points3d,
    queryPoint2d,
    queryPoint3d,
    viewMode
  ]);

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
        <div className="knn-header">
          <div>
            <h1 className="knn-title">K-närmaste grannar</h1>
            <p className="knn-lead">
              Klicka i koordinatsystemet eller ange koordinater manuellt, välj k och klassificera
              punkten med KNN.
            </p>
          </div>
          <div className="knn-header-actions">
            <button className="btn btn-secondary" type="button" onClick={toggleViewMode}>
              {viewMode === "2d" ? "Skifta till 3D" : "Skifta till 2D"}
            </button>
            <button className="btn btn-ghost" type="button" onClick={handleResetAll}>
              Återställ diagrammen
            </button>
          </div>
        </div>
      </section>

      <main className="knn-layout">
        <section className="section">
          <h2 className="knn-section-title">
            {viewMode === "2d" ? "2D-koordinatsystem" : "3D-koordinatsystem"}
          </h2>
          <div
            ref={plotElementRef}
            className={viewMode === "2d" ? "knn-plot-2d" : "knn-plot-3d"}
          />

          <div className="knn-legend">
            {CLASS_IDS.map((classId) => (
              <span key={classId} className="knn-legend-item">
                <i style={{ backgroundColor: CLASS_CONFIG[classId].color }} />
                {CLASS_CONFIG[classId].label}
              </span>
            ))}
          </div>

          <div className="knn-control-card">
            <h3>Ny datapunkt</h3>
            <p className="lead">
              Klicka i diagrammet för att placera punkten, eller ange koordinater och klicka på
              knappen.
            </p>

            <div className="knn-query-controls">
              <label className="knn-query-inline">
                <span>x =</span>
                <input
                  className="knn-query-input"
                  type="number"
                  step="0.001"
                  value={viewMode === "2d" ? queryInputs2d.x : queryInputs3d.x}
                  onChange={(event) => {
                    if (viewMode === "2d") {
                      setQueryInputs2d((previous) => ({ ...previous, x: event.target.value }));
                      return;
                    }
                    setQueryInputs3d((previous) => ({ ...previous, x: event.target.value }));
                  }}
                />
              </label>

              <label className="knn-query-inline">
                <span>y =</span>
                <input
                  className="knn-query-input"
                  type="number"
                  step="0.001"
                  value={viewMode === "2d" ? queryInputs2d.y : queryInputs3d.y}
                  onChange={(event) => {
                    if (viewMode === "2d") {
                      setQueryInputs2d((previous) => ({ ...previous, y: event.target.value }));
                      return;
                    }
                    setQueryInputs3d((previous) => ({ ...previous, y: event.target.value }));
                  }}
                />
              </label>

              {viewMode === "3d" && (
                <label className="knn-query-inline">
                  <span>z =</span>
                  <input
                    className="knn-query-input"
                    type="number"
                    step="0.001"
                    value={queryInputs3d.z}
                    onChange={(event) =>
                      setQueryInputs3d((previous) => ({ ...previous, z: event.target.value }))
                    }
                  />
                </label>
              )}
              <button className="btn btn-secondary" type="button" onClick={handleAddQueryPoint}>
                Lägg till
              </button>
            </div>

            <div className="knn-row">
              <label className="knn-inline-field knn-k-field">
                <span>Antal grannar k = :</span>
                <input
                  type="number"
                  min={1}
                  max={maxKValue}
                  step={1}
                  value={kInput}
                  onChange={(event) => setKInput(event.target.value)}
                />
              </label>

              <button className="btn btn-primary" type="button" onClick={handleClassifyPoint}>
                Klassificera punkten
              </button>
            </div>
          </div>
        </section>

        <section className="section knn-side-column">
          <div className="knn-control-card">
            <h3>Klassificeringsresultat</h3>
            {activeClassification ? (
              <>
                <p>
                  <strong>Använd k:</strong> {activeClassification.usedK}
                </p>
                <p style={{ color: CLASS_CONFIG[1].color }}>
                  <strong>Klass 1:</strong> {activeClassification.counts[1]}{" "}
                  {activeClassification.counts[1] === 1 ? "granne" : "grannar"}
                </p>
                <p style={{ color: CLASS_CONFIG[2].color }}>
                  <strong>Klass 2:</strong> {activeClassification.counts[2]}{" "}
                  {activeClassification.counts[2] === 1 ? "granne" : "grannar"}
                </p>
                <p style={{ color: CLASS_CONFIG[3].color }}>
                  <strong>Klass 3:</strong> {activeClassification.counts[3]}{" "}
                  {activeClassification.counts[3] === 1 ? "granne" : "grannar"}
                </p>
                <p style={{ color: CLASS_CONFIG[4].color }}>
                  <strong>Klass 4:</strong> {activeClassification.counts[4]}{" "}
                  {activeClassification.counts[4] === 1 ? "granne" : "grannar"}
                </p>
                <p>
                  <strong>Klassificering:</strong>{" "}
                  {activeClassification.winner
                    ? (
                      <span style={{ color: CLASS_CONFIG[activeClassification.winner].color }}>
                        {CLASS_CONFIG[activeClassification.winner].label}
                      </span>
                    )
                    : "Ingen klass hittades"}
                </p>
                <p>
                  <strong>Radie:</strong> {activeClassification.radius.toFixed(3)}
                </p>
              </>
            ) : (
              <p className="lead">
                När du klassificerar en punkt visas antal grannar per klass och vilken klass
                punkten tillhör.
              </p>
            )}
          </div>

          <div className="knn-control-card">
            <h3>Generera kluster</h3>
            <p className="lead">
              Skapa nya punkter med valbar medelpunkt, standardavvikelse och antal.
            </p>

            <div className="knn-input-grid">
              <label className="knn-field">
                <span>Klass</span>
                <select
                  value={clusterClassInput}
                  onChange={(event) => {
                    const selectedClass = Number(event.target.value) as ClassId;
                    setClusterClassInput(selectedClass);
                    setClusterMeanInputs((previous) => ({
                      ...previous,
                      ...DEFAULT_CLUSTER_MEANS_BY_CLASS[selectedClass]
                    }));
                  }}
                >
                  {CLASS_IDS.map((classId) => (
                    <option key={classId} value={classId}>
                      {CLASS_CONFIG[classId].label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="knn-field">
                <span>Antal</span>
                <input
                  type="number"
                  min={1}
                  max={300}
                  step={1}
                  value={clusterCountInput}
                  onChange={(event) => setClusterCountInput(event.target.value)}
                />
              </label>

              <label className="knn-field">
                <span>Medel x</span>
                <input
                  type="number"
                  step="0.1"
                  value={clusterMeanInputs.x}
                  onChange={(event) =>
                    setClusterMeanInputs((previous) => ({ ...previous, x: event.target.value }))
                  }
                />
              </label>

              <label className="knn-field">
                <span>Medel y</span>
                <input
                  type="number"
                  step="0.1"
                  value={clusterMeanInputs.y}
                  onChange={(event) =>
                    setClusterMeanInputs((previous) => ({ ...previous, y: event.target.value }))
                  }
                />
              </label>

              {viewMode === "3d" && (
                <label className="knn-field">
                  <span>Medel z</span>
                  <input
                    type="number"
                    step="0.1"
                    value={clusterMeanInputs.z}
                    onChange={(event) =>
                      setClusterMeanInputs((previous) => ({ ...previous, z: event.target.value }))
                    }
                  />
                </label>
              )}

              <label className="knn-field">
                <span>Standardavvikelse</span>
                <input
                  type="number"
                  min={0.01}
                  step="0.05"
                  value={clusterStdInput}
                  onChange={(event) => setClusterStdInput(event.target.value)}
                />
              </label>
            </div>

            <button className="btn btn-secondary" type="button" onClick={handleGenerateCluster}>
              Generera kluster
            </button>
          </div>

          <p className="knn-feedback">{feedback || " "}</p>
        </section>
      </main>
    </div>
  );
}

export default KNearestNeighborsPage;

