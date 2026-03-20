import { useEffect, useMemo, useRef, useState } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import "./avstand.css";

type ViewMode = "2d" | "3d";
type MetricType = "none" | "euclidean" | "manhattan" | "cosine";
type DragPoint = "a" | "b" | null;

interface Point2D {
  x: number;
  y: number;
}

interface Point3D extends Point2D {
  z: number;
}

interface Points2DState {
  a: Point2D;
  b: Point2D;
}

interface Points3DState {
  a: Point3D;
  b: Point3D;
}

interface Inputs2D {
  ax: string;
  ay: string;
  bx: string;
  by: string;
}

interface Inputs3D extends Inputs2D {
  az: string;
  bz: string;
}

interface MetricValues {
  euclidean: number;
  manhattan: number;
  cosine: number | null;
  angleDeg: number | null;
  components: number[];
}

interface Range2D {
  x: [number, number];
  y: [number, number];
  span: number;
}

interface Range3D {
  x: [number, number];
  y: [number, number];
  z: [number, number];
  span: number;
}

interface Vec3 {
  x: number;
  y: number;
  z: number;
}

const EPSILON = 1e-9;
const POINT_A_TRACE_NAME = "Punkt A";
const POINT_B_TRACE_NAME = "Punkt B";
const DEFAULT_3D_CAMERA = {
  eye: { x: 1.45, y: 1.35, z: 0.95 },
  center: { x: 0, y: 0, z: 0 },
  up: { x: 0, y: 0, z: 1 }
};
const COSINE_FIXED_RANGE_3D: Range3D = {
  x: [-12, 12],
  y: [-12, 12],
  z: [-12, 12],
  span: 24
};

const DEFAULT_POINTS_2D: Points2DState = {
  a: { x: 2, y: 1 },
  b: { x: 6, y: 4 }
};

const DEFAULT_POINTS_3D: Points3DState = {
  a: { x: 2, y: 1, z: 1 },
  b: { x: -2, y: 8, z: -1 }
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function parseFiniteNumber(value: string): number | null {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatCoordinate(value: number): string {
  return Number(value.toFixed(3)).toString();
}

function dot2d(pointA: Point2D, pointB: Point2D): number {
  return pointA.x * pointB.x + pointA.y * pointB.y;
}

function norm2d(point: Point2D): number {
  return Math.hypot(point.x, point.y);
}

function cosineSimilarity2d(pointA: Point2D, pointB: Point2D): number | null {
  const normA = norm2d(pointA);
  const normB = norm2d(pointB);

  if (normA < EPSILON || normB < EPSILON) {
    return null;
  }

  const cosine = dot2d(pointA, pointB) / (normA * normB);
  return clamp(cosine, -1, 1);
}

function dot3d(vectorA: Vec3, vectorB: Vec3): number {
  return vectorA.x * vectorB.x + vectorA.y * vectorB.y + vectorA.z * vectorB.z;
}

function add3d(vectorA: Vec3, vectorB: Vec3): Vec3 {
  return { x: vectorA.x + vectorB.x, y: vectorA.y + vectorB.y, z: vectorA.z + vectorB.z };
}

function subtract3d(vectorA: Vec3, vectorB: Vec3): Vec3 {
  return { x: vectorA.x - vectorB.x, y: vectorA.y - vectorB.y, z: vectorA.z - vectorB.z };
}

function scale3d(vector: Vec3, scalar: number): Vec3 {
  return { x: vector.x * scalar, y: vector.y * scalar, z: vector.z * scalar };
}

function cross3d(vectorA: Vec3, vectorB: Vec3): Vec3 {
  return {
    x: vectorA.y * vectorB.z - vectorA.z * vectorB.y,
    y: vectorA.z * vectorB.x - vectorA.x * vectorB.z,
    z: vectorA.x * vectorB.y - vectorA.y * vectorB.x
  };
}

function norm3d(vector: Vec3): number {
  return Math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2);
}

function normalize3d(vector: Vec3): Vec3 | null {
  const length = norm3d(vector);
  if (length < EPSILON) {
    return null;
  }
  return scale3d(vector, 1 / length);
}

function cosineSimilarity3d(pointA: Point3D, pointB: Point3D): number | null {
  const normA = norm3d(pointA);
  const normB = norm3d(pointB);

  if (normA < EPSILON || normB < EPSILON) {
    return null;
  }

  const cosine = dot3d(pointA, pointB) / (normA * normB);
  return clamp(cosine, -1, 1);
}

function createPerpendicularUnit3d(unitVector: Vec3): Vec3 {
  const reference: Vec3 =
    Math.abs(unitVector.x) < 0.8 ? { x: 1, y: 0, z: 0 } : { x: 0, y: 1, z: 0 };
  const perpendicular = cross3d(reference, unitVector);
  const normalized = normalize3d(perpendicular);

  if (normalized) {
    return normalized;
  }

  return { x: 0, y: 0, z: 1 };
}

function angleFromCosine(cosineValue: number | null): number | null {
  if (cosineValue === null) {
    return null;
  }

  return (Math.acos(clamp(cosineValue, -1, 1)) * 180) / Math.PI;
}

function rotatePointBToAngle2d(pointA: Point2D, pointB: Point2D, angleDeg: number): Point2D {
  const normA = norm2d(pointA);
  const normB = norm2d(pointB);

  if (normA < EPSILON || normB < EPSILON) {
    return pointB;
  }

  const unitA = { x: pointA.x / normA, y: pointA.y / normA };
  const perpendicular = { x: -unitA.y, y: unitA.x };

  const orientation = pointA.x * pointB.y - pointA.y * pointB.x;
  const sign = orientation < 0 ? -1 : 1;

  const angleRad = (clamp(angleDeg, 0, 180) * Math.PI) / 180;
  return {
    x: normB * (Math.cos(angleRad) * unitA.x + Math.sin(angleRad) * sign * perpendicular.x),
    y: normB * (Math.cos(angleRad) * unitA.y + Math.sin(angleRad) * sign * perpendicular.y)
  };
}

function rotatePointBToAngle3d(pointA: Point3D, pointB: Point3D, angleDeg: number): Point3D {
  const unitA = normalize3d(pointA);
  const normB = norm3d(pointB);

  if (!unitA || normB < EPSILON) {
    return pointB;
  }

  const projectionLength = dot3d(pointB, unitA);
  const projectedOnA = scale3d(unitA, projectionLength);
  const orthogonalRaw = subtract3d(pointB, projectedOnA);
  const unitOrthogonal = normalize3d(orthogonalRaw) ?? createPerpendicularUnit3d(unitA);

  const angleRad = (clamp(angleDeg, 0, 180) * Math.PI) / 180;
  const adjusted = add3d(
    scale3d(unitA, Math.cos(angleRad) * normB),
    scale3d(unitOrthogonal, Math.sin(angleRad) * normB)
  );

  return { x: adjusted.x, y: adjusted.y, z: adjusted.z };
}

function computeRange2d(points: Points2DState): Range2D {
  const xValues = [0, points.a.x, points.b.x];
  const yValues = [0, points.a.y, points.b.y];

  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);

  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const chartSpan = Math.max(spanX, spanY, 5);
  const padding = Math.max(0.9, 0.18 * chartSpan);

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const half = chartSpan / 2 + padding;

  return {
    x: [centerX - half, centerX + half],
    y: [centerY - half, centerY + half],
    span: 2 * half
  };
}

function computeRange3d(points: Points3DState): Range3D {
  const xValues = [0, points.a.x, points.b.x];
  const yValues = [0, points.a.y, points.b.y];
  const zValues = [0, points.a.z, points.b.z];

  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const minZ = Math.min(...zValues);
  const maxZ = Math.max(...zValues);

  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const spanZ = Math.max(1e-6, maxZ - minZ);

  const maxSpan = Math.max(spanX, spanY, spanZ, 5);
  const padding = Math.max(0.8, 0.2 * maxSpan);

  return {
    x: [minX - padding, maxX + padding],
    y: [minY - padding, maxY + padding],
    z: [minZ - padding, maxZ + padding],
    span: maxSpan + 2 * padding
  };
}

function computeCosineRange2d(points: Points2DState): Range2D {
  const radiusRaw = Math.max(norm2d(points.a), norm2d(points.b), 1);
  const radius = Number(radiusRaw.toFixed(6));
  const padding = Math.max(1, 0.28 * radius);
  const half = radius + padding;

  return {
    x: [-half, half],
    y: [-half, half],
    span: 2 * half
  };
}

function buildVectorArrowHead2d(tip: Point2D, size: number): { x: number[]; y: number[] } | null {
  const unit = normalize3d({ x: tip.x, y: tip.y, z: 0 });
  if (!unit) {
    return null;
  }

  const clampedSize = Math.max(0.18, size);
  const wing = 0.55 * clampedSize;
  const base = {
    x: tip.x - clampedSize * unit.x,
    y: tip.y - clampedSize * unit.y
  };

  const perpendicular = { x: -unit.y, y: unit.x };
  const left = {
    x: base.x + wing * perpendicular.x,
    y: base.y + wing * perpendicular.y
  };
  const right = {
    x: base.x - wing * perpendicular.x,
    y: base.y - wing * perpendicular.y
  };

  return {
    x: [left.x, tip.x, right.x],
    y: [left.y, tip.y, right.y]
  };
}

function buildVectorArrowHead3d(
  tip: Point3D,
  size: number
): { x: number[]; y: number[]; z: number[] } | null {
  const unit = normalize3d(tip);
  if (!unit) {
    return null;
  }

  const perpendicular = createPerpendicularUnit3d(unit);
  const clampedSize = Math.max(0.22, size);
  const wing = 0.58 * clampedSize;
  const base = subtract3d(tip, scale3d(unit, clampedSize));
  const left = add3d(base, scale3d(perpendicular, wing));
  const right = add3d(base, scale3d(perpendicular, -wing));

  return {
    x: [left.x, tip.x, right.x],
    y: [left.y, tip.y, right.y],
    z: [left.z, tip.z, right.z]
  };
}

function read2dPointFromMouseEvent(event: MouseEvent, plotElement: HTMLDivElement): Point2D | null {
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

function detect2dDragTarget(
  event: MouseEvent,
  plotElement: HTMLDivElement,
  points: Points2DState,
  pixelThreshold = 18
): DragPoint {
  const fullLayout = (plotElement as any)._fullLayout;
  const xAxis = fullLayout?.xaxis;
  const yAxis = fullLayout?.yaxis;

  if (!xAxis || !yAxis) {
    return null;
  }

  const containerRect = plotElement.getBoundingClientRect();
  const mouseX = event.clientX - containerRect.left;
  const mouseY = event.clientY - containerRect.top;

  const pointAX = xAxis.l2p(points.a.x) + xAxis._offset;
  const pointAY = yAxis.l2p(points.a.y) + yAxis._offset;
  const pointBX = xAxis.l2p(points.b.x) + xAxis._offset;
  const pointBY = yAxis.l2p(points.b.y) + yAxis._offset;

  const distanceA = Math.hypot(mouseX - pointAX, mouseY - pointAY);
  const distanceB = Math.hypot(mouseX - pointBX, mouseY - pointBY);
  const minDistance = Math.min(distanceA, distanceB);

  if (minDistance > pixelThreshold) {
    return null;
  }

  return distanceA <= distanceB ? "a" : "b";
}

function computeMetricValues2d(points: Points2DState): MetricValues {
  const deltaX = points.b.x - points.a.x;
  const deltaY = points.b.y - points.a.y;
  const euclidean = Math.hypot(deltaX, deltaY);
  const manhattan = Math.abs(deltaX) + Math.abs(deltaY);
  const cosine = cosineSimilarity2d(points.a, points.b);

  return {
    euclidean,
    manhattan,
    cosine,
    angleDeg: angleFromCosine(cosine),
    components: [Math.abs(deltaX), Math.abs(deltaY)]
  };
}

function computeMetricValues3d(points: Points3DState): MetricValues {
  const deltaX = points.b.x - points.a.x;
  const deltaY = points.b.y - points.a.y;
  const deltaZ = points.b.z - points.a.z;

  const euclidean = Math.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2);
  const manhattan = Math.abs(deltaX) + Math.abs(deltaY) + Math.abs(deltaZ);
  const cosine = cosineSimilarity3d(points.a, points.b);

  return {
    euclidean,
    manhattan,
    cosine,
    angleDeg: angleFromCosine(cosine),
    components: [Math.abs(deltaX), Math.abs(deltaY), Math.abs(deltaZ)]
  };
}

function buildCosineArc2d(pointA: Point2D, pointB: Point2D, angleDeg: number): {
  x: number[];
  y: number[];
  labelX: number;
  labelY: number;
} | null {
  const normA = norm2d(pointA);
  const normB = norm2d(pointB);

  if (normA < EPSILON || normB < EPSILON) {
    return null;
  }

  const startAngle = Math.atan2(pointA.y, pointA.x);
  const orientation = pointA.x * pointB.y - pointA.y * pointB.x;
  const direction = orientation < 0 ? -1 : 1;

  const angleRad = (angleDeg * Math.PI) / 180;
  const radius = Math.max(0.45, 0.27 * Math.min(normA, normB));
  const steps = 80;

  const x: number[] = [];
  const y: number[] = [];

  for (let step = 0; step <= steps; step += 1) {
    const t = startAngle + direction * (angleRad * step) / steps;
    x.push(radius * Math.cos(t));
    y.push(radius * Math.sin(t));
  }

  const labelAngle = startAngle + direction * angleRad / 2;

  return {
    x,
    y,
    labelX: 1.12 * radius * Math.cos(labelAngle),
    labelY: 1.12 * radius * Math.sin(labelAngle)
  };
}

function buildCosineArc3d(pointA: Point3D, pointB: Point3D, angleDeg: number): {
  x: number[];
  y: number[];
  z: number[];
  label: Point3D;
} | null {
  const unitA = normalize3d(pointA);
  const normB = norm3d(pointB);

  if (!unitA || normB < EPSILON) {
    return null;
  }

  const projectionLength = dot3d(pointB, unitA);
  const projectedOnA = scale3d(unitA, projectionLength);
  const orthogonalRaw = subtract3d(pointB, projectedOnA);
  const unitOrthogonal = normalize3d(orthogonalRaw) ?? createPerpendicularUnit3d(unitA);

  const angleRad = (angleDeg * Math.PI) / 180;
  const radius = Math.max(0.5, 0.27 * Math.min(norm3d(pointA), norm3d(pointB)));
  const steps = 90;

  const x: number[] = [];
  const y: number[] = [];
  const z: number[] = [];

  for (let step = 0; step <= steps; step += 1) {
    const t = (angleRad * step) / steps;
    const arcPoint = add3d(
      scale3d(unitA, radius * Math.cos(t)),
      scale3d(unitOrthogonal, radius * Math.sin(t))
    );

    x.push(arcPoint.x);
    y.push(arcPoint.y);
    z.push(arcPoint.z);
  }

  const middle = add3d(
    scale3d(unitA, 1.12 * radius * Math.cos(angleRad / 2)),
    scale3d(unitOrthogonal, 1.12 * radius * Math.sin(angleRad / 2))
  );

  return {
    x,
    y,
    z,
    label: { x: middle.x, y: middle.y, z: middle.z }
  };
}

function renderMath(expression: string): { __html: string } {
  return {
    __html: katex.renderToString(expression, {
      throwOnError: false,
      displayMode: true
    })
  };
}

function cloneCamera(camera: { eye: Vec3; center: Vec3; up: Vec3 }): {
  eye: Vec3;
  center: Vec3;
  up: Vec3;
} {
  return {
    eye: { ...camera.eye },
    center: { ...camera.center },
    up: { ...camera.up }
  };
}

function AvstandPage(): JSX.Element {
  const plotElementRef = useRef<HTMLDivElement | null>(null);
  const plotlyRef = useRef<any>(null);
  const cameraRef = useRef(DEFAULT_3D_CAMERA);
  const dragging2dRef = useRef<DragPoint>(null);

  const [viewMode, setViewMode] = useState<ViewMode>("2d");
  const [metric, setMetric] = useState<MetricType>("none");

  const [points2d, setPoints2d] = useState<Points2DState>(DEFAULT_POINTS_2D);
  const [points3d, setPoints3d] = useState<Points3DState>(DEFAULT_POINTS_3D);

  const [inputs2d, setInputs2d] = useState<Inputs2D>({
    ax: formatCoordinate(DEFAULT_POINTS_2D.a.x),
    ay: formatCoordinate(DEFAULT_POINTS_2D.a.y),
    bx: formatCoordinate(DEFAULT_POINTS_2D.b.x),
    by: formatCoordinate(DEFAULT_POINTS_2D.b.y)
  });

  const [inputs3d, setInputs3d] = useState<Inputs3D>({
    ax: formatCoordinate(DEFAULT_POINTS_3D.a.x),
    ay: formatCoordinate(DEFAULT_POINTS_3D.a.y),
    az: formatCoordinate(DEFAULT_POINTS_3D.a.z),
    bx: formatCoordinate(DEFAULT_POINTS_3D.b.x),
    by: formatCoordinate(DEFAULT_POINTS_3D.b.y),
    bz: formatCoordinate(DEFAULT_POINTS_3D.b.z)
  });

  const range2d = useMemo(() => computeRange2d(points2d), [points2d]);
  const range3d = useMemo(() => computeRange3d(points3d), [points3d]);
  const cosineRange2d = useMemo(() => computeCosineRange2d(points2d), [points2d]);
  const activeRange2d = metric === "cosine" ? cosineRange2d : range2d;
  const activeRange3d = metric === "cosine" ? COSINE_FIXED_RANGE_3D : range3d;

  const metricValues2d = useMemo(() => computeMetricValues2d(points2d), [points2d]);
  const metricValues3d = useMemo(() => computeMetricValues3d(points3d), [points3d]);
  const activeValues = viewMode === "2d" ? metricValues2d : metricValues3d;

  const metricMath = useMemo(() => {
    if (metric === "none") {
      return null;
    }

    if (metric === "euclidean") {
      return viewMode === "2d"
        ? renderMath(String.raw`d_E(\mathbf{p},\mathbf{q})=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}`)
        : renderMath(
            String.raw`d_E(\mathbf{p},\mathbf{q})=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2+(z_2-z_1)^2}`
          );
    }

    if (metric === "manhattan") {
      return viewMode === "2d"
        ? renderMath(String.raw`d_M(\mathbf{p},\mathbf{q})=|x_2-x_1|+|y_2-y_1|`)
        : renderMath(
            String.raw`d_M(\mathbf{p},\mathbf{q})=|x_2-x_1|+|y_2-y_1|+|z_2-z_1|`
          );
    }

    return renderMath(
      String.raw`s_{\cos}(\mathbf{p},\mathbf{q})=\frac{\mathbf{p}\cdot\mathbf{q}}{\|\mathbf{p}\|\|\mathbf{q}\|}=\cos(\theta)`
    );
  }, [metric, viewMode]);

  useEffect(() => {
    setInputs2d({
      ax: formatCoordinate(points2d.a.x),
      ay: formatCoordinate(points2d.a.y),
      bx: formatCoordinate(points2d.b.x),
      by: formatCoordinate(points2d.b.y)
    });
  }, [points2d]);

  useEffect(() => {
    setInputs3d({
      ax: formatCoordinate(points3d.a.x),
      ay: formatCoordinate(points3d.a.y),
      az: formatCoordinate(points3d.a.z),
      bx: formatCoordinate(points3d.b.x),
      by: formatCoordinate(points3d.b.y),
      bz: formatCoordinate(points3d.b.z)
    });
  }, [points3d]);

  const updatePointFrom2dInput = (key: keyof Inputs2D, rawValue: string): void => {
    setInputs2d((previous) => ({ ...previous, [key]: rawValue }));

    const parsed = parseFiniteNumber(rawValue);
    if (parsed === null) {
      return;
    }

    setPoints2d((previous) => {
      if (key === "ax") {
        return { ...previous, a: { ...previous.a, x: parsed } };
      }
      if (key === "ay") {
        return { ...previous, a: { ...previous.a, y: parsed } };
      }
      if (key === "bx") {
        return { ...previous, b: { ...previous.b, x: parsed } };
      }
      return { ...previous, b: { ...previous.b, y: parsed } };
    });
  };

  const restore2dInputIfInvalid = (key: keyof Inputs2D): void => {
    const value = inputs2d[key];
    if (parseFiniteNumber(value) !== null) {
      return;
    }

    const fallback =
      key === "ax"
        ? points2d.a.x
        : key === "ay"
          ? points2d.a.y
          : key === "bx"
            ? points2d.b.x
            : points2d.b.y;

    setInputs2d((previous) => ({ ...previous, [key]: formatCoordinate(fallback) }));
  };

  const updatePointFrom3dInput = (key: keyof Inputs3D, rawValue: string): void => {
    setInputs3d((previous) => ({ ...previous, [key]: rawValue }));

    const parsed = parseFiniteNumber(rawValue);
    if (parsed === null) {
      return;
    }

    setPoints3d((previous) => {
      if (key === "ax") {
        return { ...previous, a: { ...previous.a, x: parsed } };
      }
      if (key === "ay") {
        return { ...previous, a: { ...previous.a, y: parsed } };
      }
      if (key === "az") {
        return { ...previous, a: { ...previous.a, z: parsed } };
      }
      if (key === "bx") {
        return { ...previous, b: { ...previous.b, x: parsed } };
      }
      if (key === "by") {
        return { ...previous, b: { ...previous.b, y: parsed } };
      }
      return { ...previous, b: { ...previous.b, z: parsed } };
    });
  };

  const restore3dInputIfInvalid = (key: keyof Inputs3D): void => {
    const value = inputs3d[key];
    if (parseFiniteNumber(value) !== null) {
      return;
    }

    const fallback =
      key === "ax"
        ? points3d.a.x
        : key === "ay"
          ? points3d.a.y
          : key === "az"
            ? points3d.a.z
            : key === "bx"
              ? points3d.b.x
              : key === "by"
                ? points3d.b.y
                : points3d.b.z;

    setInputs3d((previous) => ({ ...previous, [key]: formatCoordinate(fallback) }));
  };

  const handleCosineAngleChange = (nextAngle: number): void => {
    if (viewMode === "2d") {
      setPoints2d((previous) => ({
        ...previous,
        b: rotatePointBToAngle2d(previous.a, previous.b, nextAngle)
      }));
      return;
    }

    setPoints3d((previous) => ({
      ...previous,
      b: rotatePointBToAngle3d(previous.a, previous.b, nextAngle)
    }));
  };

  const toggleViewMode = (): void => {
    setViewMode((previous) => (previous === "2d" ? "3d" : "2d"));
    dragging2dRef.current = null;
  };

  const handleReset = (): void => {
    setMetric("none");
    setViewMode("2d");
    setPoints2d(DEFAULT_POINTS_2D);
    setPoints3d(DEFAULT_POINTS_3D);
    dragging2dRef.current = null;
  };

  useEffect(() => {
    let cancelled = false;
    const plotElement = plotElementRef.current;

    if (!plotElement) {
      return;
    }

    let onMouseDown2d: ((event: MouseEvent) => void) | null = null;
    let onMouseMove2d: ((event: MouseEvent) => void) | null = null;
    let onMouseUpGlobal: (() => void) | null = null;

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

        if (metric === "euclidean") {
          const corner = { x: points2d.b.x, y: points2d.a.y };
          const midpoint = {
            x: (points2d.a.x + points2d.b.x) / 2,
            y: (points2d.a.y + points2d.b.y) / 2
          };
          const labelOffset = activeRange2d.span * 0.025;

          traces.push({
            type: "scatter",
            mode: "lines",
            x: [points2d.a.x, points2d.b.x],
            y: [points2d.a.y, points2d.b.y],
            hoverinfo: "skip",
            line: { color: "#111111", width: 3 }
          });

          traces.push({
            type: "scatter",
            mode: "lines",
            x: [points2d.a.x, corner.x, points2d.b.x],
            y: [points2d.a.y, corner.y, points2d.b.y],
            hoverinfo: "skip",
            line: { color: "#111111", width: 2, dash: "dash" }
          });

          traces.push({
            type: "scatter",
            mode: "text",
            x: [(points2d.a.x + corner.x) / 2, corner.x + labelOffset],
            y: [corner.y + labelOffset, (corner.y + points2d.b.y) / 2],
            text: [
              `|Δx| = ${activeValues.components[0].toFixed(3)}`,
              `|Δy| = ${activeValues.components[1].toFixed(3)}`
            ],
            textfont: { size: 13, color: "#111111" },
            hoverinfo: "skip"
          });

          traces.push({
            type: "scatter",
            mode: "text",
            x: [midpoint.x],
            y: [midpoint.y + labelOffset],
            text: [`d = ${activeValues.euclidean.toFixed(3)}`],
            textfont: { size: 14, color: "#111111" },
            hoverinfo: "skip"
          });
        }

        if (metric === "manhattan") {
          const corner = { x: points2d.b.x, y: points2d.a.y };
          const labelOffset = activeRange2d.span * 0.025;

          traces.push({
            type: "scatter",
            mode: "lines",
            x: [points2d.a.x, corner.x, points2d.b.x],
            y: [points2d.a.y, corner.y, points2d.b.y],
            hoverinfo: "skip",
            line: { color: "#111111", width: 3 }
          });

          traces.push({
            type: "scatter",
            mode: "text",
            x: [(points2d.a.x + corner.x) / 2, corner.x + labelOffset],
            y: [corner.y + labelOffset, (corner.y + points2d.b.y) / 2],
            text: [
              `|Δx| = ${activeValues.components[0].toFixed(3)}`,
              `|Δy| = ${activeValues.components[1].toFixed(3)}`
            ],
            textfont: { size: 13, color: "#111111" },
            hoverinfo: "skip"
          });

          traces.push({
            type: "scatter",
            mode: "text",
            x: [points2d.b.x + labelOffset],
            y: [points2d.a.y + labelOffset],
            text: [`d_M = ${activeValues.manhattan.toFixed(3)}`],
            textfont: { size: 14, color: "#111111" },
            hoverinfo: "skip"
          });
        }

        if (metric === "cosine") {
          const arrowSize = 0.034 * activeRange2d.span;
          const arrowHeadA = buildVectorArrowHead2d(points2d.a, arrowSize);
          const arrowHeadB = buildVectorArrowHead2d(points2d.b, arrowSize);

          traces.push({
            type: "scatter",
            mode: "lines",
            x: [0, points2d.a.x],
            y: [0, points2d.a.y],
            hoverinfo: "skip",
            line: { color: "#2864c7", width: 4 }
          });

          traces.push({
            type: "scatter",
            mode: "lines",
            x: [0, points2d.b.x],
            y: [0, points2d.b.y],
            hoverinfo: "skip",
            line: { color: "#3f87ff", width: 4 }
          });

          if (arrowHeadA) {
            traces.push({
              type: "scatter",
              mode: "lines",
              x: arrowHeadA.x,
              y: arrowHeadA.y,
              hoverinfo: "skip",
              line: { color: "#2864c7", width: 4 }
            });
          }

          if (arrowHeadB) {
            traces.push({
              type: "scatter",
              mode: "lines",
              x: arrowHeadB.x,
              y: arrowHeadB.y,
              hoverinfo: "skip",
              line: { color: "#3f87ff", width: 4 }
            });
          }

          if (activeValues.angleDeg !== null && activeValues.cosine !== null) {
            const arc = buildCosineArc2d(points2d.a, points2d.b, activeValues.angleDeg);

            if (arc) {
              const labelOffset = activeRange2d.span * 0.025;

              traces.push({
                type: "scatter",
                mode: "lines",
                x: arc.x,
                y: arc.y,
                hoverinfo: "skip",
                line: { color: "#111111", width: 2.5 }
              });

              traces.push({
                type: "scatter",
                mode: "text",
                x: [arc.labelX],
                y: [arc.labelY],
                text: [`${activeValues.angleDeg.toFixed(1)}°`],
                textfont: { size: 14, color: "#111111" },
                hoverinfo: "skip"
              });

              traces.push({
                type: "scatter",
                mode: "text",
                x: [(points2d.a.x + points2d.b.x) * 0.33],
                y: [(points2d.a.y + points2d.b.y) * 0.33 + labelOffset],
                text: [`cosinuslikhet = ${activeValues.cosine.toFixed(3)}`],
                textfont: { size: 14, color: "#111111" },
                hoverinfo: "skip"
              });
            }
          }
        }

        traces.push(
          {
            type: "scatter",
            mode: "markers+text",
            name: POINT_A_TRACE_NAME,
            x: [points2d.a.x],
            y: [points2d.a.y],
            text: ["A"],
            textposition: "top center",
            marker: {
              size: 14,
              color: "#2864c7",
              line: { color: "#ffffff", width: 1.2 }
            },
            hovertemplate: "Punkt A<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"
          },
          {
            type: "scatter",
            mode: "markers+text",
            name: POINT_B_TRACE_NAME,
            x: [points2d.b.x],
            y: [points2d.b.y],
            text: ["B"],
            textposition: "top center",
            marker: {
              size: 14,
              color: "#2864c7",
              line: { color: "#ffffff", width: 1.2 }
            },
            hovertemplate: "Punkt B<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"
          }
        );

        await Plotly.react(
          plotElement,
          traces,
          {
            margin: { l: 55, r: 12, t: 12, b: 50 },
            paper_bgcolor: "#ffffff",
            plot_bgcolor: "#f8fbfe",
            showlegend: false,
            dragmode: false,
            hovermode: "closest",
            xaxis: {
              title: "x",
              range: activeRange2d.x,
              fixedrange: true,
              zeroline: true,
              zerolinewidth: 1.4,
              zerolinecolor: "#8fa9be",
              gridcolor: "#dce8f2"
            },
            yaxis: {
              title: "y",
              range: activeRange2d.y,
              fixedrange: true,
              scaleanchor: "x",
              scaleratio: 1,
              zeroline: true,
              zerolinewidth: 1.4,
              zerolinecolor: "#8fa9be",
              gridcolor: "#dce8f2"
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

        onMouseDown2d = (event: MouseEvent): void => {
          if (event.button !== 0) {
            return;
          }

          dragging2dRef.current = detect2dDragTarget(event, plotElement, points2d);
          if (dragging2dRef.current) {
            event.preventDefault();
            event.stopPropagation();
          }
        };

        onMouseMove2d = (event: MouseEvent): void => {
          if (!dragging2dRef.current) {
            return;
          }

          if ((event.buttons & 1) === 0) {
            dragging2dRef.current = null;
            return;
          }

          const nextPoint = read2dPointFromMouseEvent(event, plotElement);
          if (!nextPoint) {
            return;
          }

          setPoints2d((previous) =>
            dragging2dRef.current === "a"
              ? { ...previous, a: nextPoint }
              : { ...previous, b: nextPoint }
          );
        };

        plotElement.addEventListener("mousedown", onMouseDown2d);
        window.addEventListener("mousemove", onMouseMove2d);

        onMouseUpGlobal = () => {
          dragging2dRef.current = null;
        };
        window.addEventListener("mouseup", onMouseUpGlobal);
        return;
      }

      const layoutCamera = graphDiv.layout?.scene?.camera;
      if (layoutCamera && metric !== "cosine") {
        cameraRef.current = layoutCamera;
      }
      const sceneCamera = metric === "cosine" ? cloneCamera(DEFAULT_3D_CAMERA) : cameraRef.current;

      const traces: any[] = [];

      if (metric === "euclidean") {
        const corner1 = { x: points3d.b.x, y: points3d.a.y, z: points3d.a.z };
        const corner2 = { x: points3d.b.x, y: points3d.b.y, z: points3d.a.z };
        const textOffset = 0.035 * activeRange3d.span;

        traces.push({
          type: "scatter3d",
          mode: "lines",
          x: [points3d.a.x, points3d.b.x],
          y: [points3d.a.y, points3d.b.y],
          z: [points3d.a.z, points3d.b.z],
          hoverinfo: "skip",
          line: { color: "#111111", width: 6 }
        });

        traces.push({
          type: "scatter3d",
          mode: "lines",
          x: [points3d.a.x, corner1.x, corner2.x, points3d.b.x],
          y: [points3d.a.y, corner1.y, corner2.y, points3d.b.y],
          z: [points3d.a.z, corner1.z, corner2.z, points3d.b.z],
          hoverinfo: "skip",
          line: { color: "#111111", width: 4, dash: "dash" }
        });

        traces.push({
          type: "scatter3d",
          mode: "text",
          x: [
            (points3d.a.x + corner1.x) / 2,
            (corner1.x + corner2.x) / 2,
            (corner2.x + points3d.b.x) / 2
          ],
          y: [
            (points3d.a.y + corner1.y) / 2,
            (corner1.y + corner2.y) / 2,
            (corner2.y + points3d.b.y) / 2
          ],
          z: [
            (points3d.a.z + corner1.z) / 2 + textOffset,
            (corner1.z + corner2.z) / 2 + textOffset,
            (corner2.z + points3d.b.z) / 2 + textOffset
          ],
          text: [
            `|Δx| = ${activeValues.components[0].toFixed(3)}`,
            `|Δy| = ${activeValues.components[1].toFixed(3)}`,
            `|Δz| = ${activeValues.components[2].toFixed(3)}`
          ],
          textfont: { size: 12, color: "#111111" },
          hoverinfo: "skip"
        });

        traces.push({
          type: "scatter3d",
          mode: "text",
          x: [(points3d.a.x + points3d.b.x) / 2],
          y: [(points3d.a.y + points3d.b.y) / 2],
          z: [(points3d.a.z + points3d.b.z) / 2 + textOffset],
          text: [`d = ${activeValues.euclidean.toFixed(3)}`],
          textfont: { size: 14, color: "#111111" },
          hoverinfo: "skip"
        });
      }

      if (metric === "manhattan") {
        const corner1 = { x: points3d.b.x, y: points3d.a.y, z: points3d.a.z };
        const corner2 = { x: points3d.b.x, y: points3d.b.y, z: points3d.a.z };
        const textOffset = 0.035 * activeRange3d.span;

        traces.push({
          type: "scatter3d",
          mode: "lines",
          x: [points3d.a.x, corner1.x, corner2.x, points3d.b.x],
          y: [points3d.a.y, corner1.y, corner2.y, points3d.b.y],
          z: [points3d.a.z, corner1.z, corner2.z, points3d.b.z],
          hoverinfo: "skip",
          line: { color: "#111111", width: 6 }
        });

        traces.push({
          type: "scatter3d",
          mode: "text",
          x: [
            (points3d.a.x + corner1.x) / 2,
            (corner1.x + corner2.x) / 2,
            (corner2.x + points3d.b.x) / 2
          ],
          y: [
            (points3d.a.y + corner1.y) / 2,
            (corner1.y + corner2.y) / 2,
            (corner2.y + points3d.b.y) / 2
          ],
          z: [
            (points3d.a.z + corner1.z) / 2 + textOffset,
            (corner1.z + corner2.z) / 2 + textOffset,
            (corner2.z + points3d.b.z) / 2 + textOffset
          ],
          text: [
            `|Δx| = ${activeValues.components[0].toFixed(3)}`,
            `|Δy| = ${activeValues.components[1].toFixed(3)}`,
            `|Δz| = ${activeValues.components[2].toFixed(3)}`
          ],
          textfont: { size: 12, color: "#111111" },
          hoverinfo: "skip"
        });

        traces.push({
          type: "scatter3d",
          mode: "text",
          x: [corner2.x],
          y: [corner2.y],
          z: [corner2.z + 2 * textOffset],
          text: [`d_M = ${activeValues.manhattan.toFixed(3)}`],
          textfont: { size: 14, color: "#111111" },
          hoverinfo: "skip"
        });
      }

      if (metric === "cosine") {
        const arrowSize = 0.048 * activeRange3d.span;
        const arrowHeadA = buildVectorArrowHead3d(points3d.a, arrowSize);
        const arrowHeadB = buildVectorArrowHead3d(points3d.b, arrowSize);

        traces.push({
          type: "scatter3d",
          mode: "lines",
          x: [0, points3d.a.x],
          y: [0, points3d.a.y],
          z: [0, points3d.a.z],
          hoverinfo: "skip",
          line: { color: "#2864c7", width: 7 }
        });

        traces.push({
          type: "scatter3d",
          mode: "lines",
          x: [0, points3d.b.x],
          y: [0, points3d.b.y],
          z: [0, points3d.b.z],
          hoverinfo: "skip",
          line: { color: "#3f87ff", width: 7 }
        });

        if (arrowHeadA) {
          traces.push({
            type: "scatter3d",
            mode: "lines",
            x: arrowHeadA.x,
            y: arrowHeadA.y,
            z: arrowHeadA.z,
            hoverinfo: "skip",
            line: { color: "#2864c7", width: 7 }
          });
        }

        if (arrowHeadB) {
          traces.push({
            type: "scatter3d",
            mode: "lines",
            x: arrowHeadB.x,
            y: arrowHeadB.y,
            z: arrowHeadB.z,
            hoverinfo: "skip",
            line: { color: "#3f87ff", width: 7 }
          });
        }

        if (activeValues.angleDeg !== null && activeValues.cosine !== null) {
          const arc = buildCosineArc3d(points3d.a, points3d.b, activeValues.angleDeg);
          if (arc) {
            const textOffset = 0.03 * activeRange3d.span;

            traces.push({
              type: "scatter3d",
              mode: "lines",
              x: arc.x,
              y: arc.y,
              z: arc.z,
              hoverinfo: "skip",
              line: { color: "#111111", width: 5 }
            });

            traces.push({
              type: "scatter3d",
              mode: "text",
              x: [arc.label.x],
              y: [arc.label.y],
              z: [arc.label.z],
              text: [`${activeValues.angleDeg.toFixed(1)}°`],
              textfont: { size: 13, color: "#111111" },
              hoverinfo: "skip"
            });

            traces.push({
              type: "scatter3d",
              mode: "text",
              x: [(points3d.a.x + points3d.b.x) * 0.33],
              y: [(points3d.a.y + points3d.b.y) * 0.33],
              z: [(points3d.a.z + points3d.b.z) * 0.33 + textOffset],
              text: [`cosinuslikhet = ${activeValues.cosine.toFixed(3)}`],
              textfont: { size: 13, color: "#111111" },
              hoverinfo: "skip"
            });
          }
        }
      }

      traces.push(
        {
          type: "scatter3d",
          mode: "markers+text",
          name: POINT_A_TRACE_NAME,
          x: [points3d.a.x],
          y: [points3d.a.y],
          z: [points3d.a.z],
          text: ["A"],
          textposition: "top center",
          marker: {
            size: 6.8,
            color: "#2864c7",
            line: { color: "#ffffff", width: 1 }
          },
          hovertemplate: "Punkt A<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>"
        },
        {
          type: "scatter3d",
          mode: "markers+text",
          name: POINT_B_TRACE_NAME,
          x: [points3d.b.x],
          y: [points3d.b.y],
          z: [points3d.b.z],
          text: ["B"],
          textposition: "top center",
          marker: {
            size: 6.8,
            color: "#2864c7",
            line: { color: "#ffffff", width: 1 }
          },
          hovertemplate: "Punkt B<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>"
        }
      );

      await Plotly.react(
        plotElement,
        traces,
        {
          margin: { l: 0, r: 0, t: 10, b: 0 },
          paper_bgcolor: "#ffffff",
          dragmode: metric === "cosine" ? false : "orbit",
          uirevision: "avstand-3d-camera",
          showlegend: false,
          scene: {
            uirevision: "avstand-3d-camera",
            aspectmode: "cube",
            xaxis: {
              title: "x",
              autorange: false,
              range: activeRange3d.x,
              backgroundcolor: "#f8fbfe"
            },
            yaxis: {
              title: "y",
              autorange: false,
              range: activeRange3d.y,
              backgroundcolor: "#f8fbfe"
            },
            zaxis: {
              title: "z",
              autorange: false,
              range: activeRange3d.z,
              backgroundcolor: "#f8fbfe"
            },
            camera: sceneCamera
          }
        },
        {
          responsive: true,
          displaylogo: false,
          scrollZoom: false,
          modeBarButtonsToRemove: ["lasso2d", "select2d"]
        }
      );
    };

    void renderPlot();

    return () => {
      cancelled = true;

      if (onMouseDown2d) {
        plotElement.removeEventListener("mousedown", onMouseDown2d);
      }

      if (onMouseMove2d) {
        window.removeEventListener("mousemove", onMouseMove2d);
      }

      if (onMouseUpGlobal) {
        window.removeEventListener("mouseup", onMouseUpGlobal);
      }
    };
  }, [activeRange2d, activeRange3d, activeValues, metric, points2d, points3d, viewMode]);

  useEffect(() => {
    return () => {
      const plotElement = plotElementRef.current;
      if (plotElement && plotlyRef.current) {
        plotlyRef.current.purge(plotElement);
      }
    };
  }, []);

  const metricTitle =
    metric === "none"
      ? "Inget mått valt"
      : metric === "euclidean"
        ? "Euklidiskt avstånd"
        : metric === "manhattan"
          ? "Manhattan-avstånd"
          : "Cosinuslikhet";

  const metricDescription =
    metric === "euclidean"
      ? "Euklidiskt avstånd mäter den kortaste raka vägen mellan två punkter."
      : metric === "manhattan"
        ? "Manhattan-avstånd summerar förflyttningar längs axlarna, som kvarter i ett rutnät."
        : metric === "cosine"
          ? "Cosinuslikhet mäter hur lika riktning två vektorer har, oberoende av deras längder."
          : "";

  return (
    <div className="container page-flow">
      <section className="section">
        <div className="avstand-header">
          <div>
            <h1 className="avstand-title">Avstånd</h1>
            <p className="avstand-lead">
              Utforska euklidiskt avstånd, Manhattan-avstånd och cosinuslikhet i 2D och 3D.
            </p>
          </div>
          <div className="avstand-header-actions">
            <button className="btn btn-secondary" type="button" onClick={toggleViewMode}>
              {viewMode === "2d" ? "Skifta till 3D" : "Skifta till 2D"}
            </button>
            <button className="btn btn-ghost" type="button" onClick={handleReset}>
              Återställ
            </button>
          </div>
        </div>
      </section>

      <main className="avstand-layout">
        <section className="section">
          <h2 className="avstand-section-title">
            {viewMode === "2d" ? "2D-koordinatsystem" : "3D-koordinatsystem"}
          </h2>
          <div
            ref={plotElementRef}
            className={viewMode === "2d" ? "avstand-plot-2d" : "avstand-plot-3d"}
          />

          <p className="avstand-drag-hint">
            {viewMode === "2d"
              ? "Dra punkt A eller B i diagrammet för att uppdatera koordinaterna."
              : "I 3D ändrar du koordinaterna i fälten under diagrammet."}
          </p>

          <div className="avstand-coordinate-grid">
            <div className="avstand-point-card">
              <h3>Punkt A</h3>
              <div className="avstand-point-fields">
                <label>
                  <span>x =</span>
                  <input
                    type="number"
                    step="0.001"
                    value={viewMode === "2d" ? inputs2d.ax : inputs3d.ax}
                    onChange={(event) =>
                      viewMode === "2d"
                        ? updatePointFrom2dInput("ax", event.target.value)
                        : updatePointFrom3dInput("ax", event.target.value)
                    }
                    onBlur={() =>
                      viewMode === "2d"
                        ? restore2dInputIfInvalid("ax")
                        : restore3dInputIfInvalid("ax")
                    }
                  />
                </label>

                <label>
                  <span>y =</span>
                  <input
                    type="number"
                    step="0.001"
                    value={viewMode === "2d" ? inputs2d.ay : inputs3d.ay}
                    onChange={(event) =>
                      viewMode === "2d"
                        ? updatePointFrom2dInput("ay", event.target.value)
                        : updatePointFrom3dInput("ay", event.target.value)
                    }
                    onBlur={() =>
                      viewMode === "2d"
                        ? restore2dInputIfInvalid("ay")
                        : restore3dInputIfInvalid("ay")
                    }
                  />
                </label>

                {viewMode === "3d" && (
                  <label>
                    <span>z =</span>
                    <input
                      type="number"
                      step="0.001"
                      value={inputs3d.az}
                      onChange={(event) => updatePointFrom3dInput("az", event.target.value)}
                      onBlur={() => restore3dInputIfInvalid("az")}
                    />
                  </label>
                )}
              </div>
            </div>

            <div className="avstand-point-card">
              <h3>Punkt B</h3>
              <div className="avstand-point-fields">
                <label>
                  <span>x =</span>
                  <input
                    type="number"
                    step="0.001"
                    value={viewMode === "2d" ? inputs2d.bx : inputs3d.bx}
                    onChange={(event) =>
                      viewMode === "2d"
                        ? updatePointFrom2dInput("bx", event.target.value)
                        : updatePointFrom3dInput("bx", event.target.value)
                    }
                    onBlur={() =>
                      viewMode === "2d"
                        ? restore2dInputIfInvalid("bx")
                        : restore3dInputIfInvalid("bx")
                    }
                  />
                </label>

                <label>
                  <span>y =</span>
                  <input
                    type="number"
                    step="0.001"
                    value={viewMode === "2d" ? inputs2d.by : inputs3d.by}
                    onChange={(event) =>
                      viewMode === "2d"
                        ? updatePointFrom2dInput("by", event.target.value)
                        : updatePointFrom3dInput("by", event.target.value)
                    }
                    onBlur={() =>
                      viewMode === "2d"
                        ? restore2dInputIfInvalid("by")
                        : restore3dInputIfInvalid("by")
                    }
                  />
                </label>

                {viewMode === "3d" && (
                  <label>
                    <span>z =</span>
                    <input
                      type="number"
                      step="0.001"
                      value={inputs3d.bz}
                      onChange={(event) => updatePointFrom3dInput("bz", event.target.value)}
                      onBlur={() => restore3dInputIfInvalid("bz")}
                    />
                  </label>
                )}
              </div>
            </div>
          </div>
        </section>

        <section className="section avstand-side-column">
          <div className="avstand-control-card">
            <h3>Välj mått</h3>
            <fieldset className="avstand-radio-group">
              <label>
                <input
                  type="radio"
                  name="metric"
                  value="none"
                  checked={metric === "none"}
                  onChange={() => setMetric("none")}
                />
                Inget
              </label>
              <label>
                <input
                  type="radio"
                  name="metric"
                  value="euclidean"
                  checked={metric === "euclidean"}
                  onChange={() => setMetric("euclidean")}
                />
                Euklidiskt avstånd
              </label>
              <label>
                <input
                  type="radio"
                  name="metric"
                  value="manhattan"
                  checked={metric === "manhattan"}
                  onChange={() => setMetric("manhattan")}
                />
                Manhattan-avstånd
              </label>
              <label>
                <input
                  type="radio"
                  name="metric"
                  value="cosine"
                  checked={metric === "cosine"}
                  onChange={() => setMetric("cosine")}
                />
                Cosinuslikhet
              </label>
            </fieldset>
          </div>

          {metric === "cosine" && (
            <div className="avstand-control-card">
              <h3>Justera vinkel</h3>
              <label className="avstand-slider-field">
                <span>
                  Vinkel:{" "}
                  {activeValues.angleDeg !== null
                    ? `${activeValues.angleDeg.toFixed(1)}°`
                    : "ej definierad"}
                </span>
                <input
                  type="range"
                  min={0}
                  max={180}
                  step={1}
                  value={activeValues.angleDeg !== null ? activeValues.angleDeg : 0}
                  disabled={activeValues.angleDeg === null}
                  onChange={(event) => handleCosineAngleChange(Number(event.target.value))}
                />
              </label>
              <p className="lead">
                Glidaren roterar punkt B kring origo så att vinkeln mellan vektorerna ändras.
              </p>
            </div>
          )}

          <div className="avstand-control-card">
            <h3>Beräknat värde</h3>
            {metric === "none" ? (
              <p className="lead">Välj ett mått för att se visualisering och beräkning.</p>
            ) : (
              <>
                <p>
                  <strong>{metricTitle}:</strong>{" "}
                  {(metric === "euclidean"
                    ? activeValues.euclidean
                    : metric === "manhattan"
                      ? activeValues.manhattan
                      : activeValues.cosine ?? Number.NaN
                  ).toFixed(3)}
                </p>

                {metric === "euclidean" && (
                  <>
                    <p>
                      <strong>|Δx|:</strong> {activeValues.components[0].toFixed(3)}
                    </p>
                    <p>
                      <strong>|Δy|:</strong> {activeValues.components[1].toFixed(3)}
                    </p>
                    {viewMode === "3d" && (
                      <p>
                        <strong>|Δz|:</strong> {activeValues.components[2].toFixed(3)}
                      </p>
                    )}
                  </>
                )}

                {metric === "manhattan" && (
                  <>
                    <p>
                      <strong>|Δx|:</strong> {activeValues.components[0].toFixed(3)}
                    </p>
                    <p>
                      <strong>|Δy|:</strong> {activeValues.components[1].toFixed(3)}
                    </p>
                    {viewMode === "3d" && (
                      <p>
                        <strong>|Δz|:</strong> {activeValues.components[2].toFixed(3)}
                      </p>
                    )}
                    <p>
                      <strong>Summa:</strong> {activeValues.manhattan.toFixed(3)}
                    </p>
                  </>
                )}

                {metric === "cosine" && (
                  <>
                    {activeValues.cosine === null ? (
                      <p className="lead">
                        Cosinuslikhet är inte definierad när någon vektor har längden 0.
                      </p>
                    ) : (
                      <>
                        <p>
                          <strong>Cosinuslikhet:</strong> {activeValues.cosine.toFixed(3)}
                        </p>
                        <p>
                          <strong>Vinkel:</strong> {activeValues.angleDeg?.toFixed(1)}°
                        </p>
                      </>
                    )}
                  </>
                )}
              </>
            )}
          </div>

          {metric !== "none" && metricMath && (
            <div className="avstand-control-card">
              <h3>Så fungerar måttet</h3>
              <p className="lead">{metricDescription}</p>
              <div className="avstand-math-block" dangerouslySetInnerHTML={metricMath} />
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default AvstandPage;
