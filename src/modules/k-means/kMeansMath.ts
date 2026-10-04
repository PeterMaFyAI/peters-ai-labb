export interface Coordinates {
  x: number;
  y: number;
}

export interface ClusterParameters {
  meanX: number;
  meanY: number;
  stdX: number;
  stdY: number;
  correlation: number;
}

export interface GeneratedCluster extends ClusterParameters {
  id: number;
  samples: Coordinates[];
}

export interface DataPoint extends Coordinates {
  id: string;
  sourceClusterId: number;
}

export interface Centroid extends Coordinates {
  id: number;
}

export interface ClassificationPoint extends Coordinates {
  centroidId: number | null;
}

export type KMeansPhase = "uninitialized" | "assign" | "update" | "converged";

export interface KMeansRun {
  phase: KMeansPhase;
  centroids: Centroid[];
  assignments: number[] | null;
  previousCentroids: Centroid[] | null;
  iterations: number;
  changedPoints: number | null;
  emptyCentroidIds: number[];
}

type RandomSource = () => number;

export const CENTROID_COLORS = ["#2864c7", "#c43f3a", "#238453", "#9455bf", "#b77708"];
export const DEFAULT_POINTS_PER_CLUSTER = 60;

export function seededRandom(seed: number): RandomSource {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
}

function normalPair(random: RandomSource): Coordinates {
  const radius = Math.sqrt(-2 * Math.log(Math.max(Number.MIN_VALUE, random())));
  const angle = 2 * Math.PI * random();
  return { x: radius * Math.cos(angle), y: radius * Math.sin(angle) };
}

function makeCluster(
  id: number,
  count: number,
  parameters: ClusterParameters,
  random: RandomSource
): GeneratedCluster {
  return { id, ...parameters, samples: Array.from({ length: count }, () => normalPair(random)) };
}

export function createDefaultClusters(): GeneratedCluster[] {
  const random = seededRandom(20261004);
  const parameters: ClusterParameters[] = [
    { meanX: -3, meanY: -1, stdX: 0.85, stdY: 0.65, correlation: 0.25 },
    { meanX: 2, meanY: -0.8, stdX: 1.35, stdY: 0.9, correlation: -0.35 },
    { meanX: 0.4, meanY: 2.8, stdX: 1, stdY: 1.3, correlation: 0.45 }
  ];
  return parameters.map((parameters, index) =>
    makeCluster(index + 1, DEFAULT_POINTS_PER_CLUSTER, parameters, random)
  );
}

export const KMEANS_EXAMPLES = [
  { count: 1, description: "Ett sammanhängande punktmoln" },
  { count: 2, description: "Två tydligt skilda grupper" },
  { count: 3, description: "Standard · viss överlappning" },
  { count: 4, description: "Fyra olika spridningar" },
  { count: 5, description: "Fem grupper att jämföra" }
];

export function createExampleClusters(count: number): GeneratedCluster[] {
  if (!Number.isInteger(count) || count < 1 || count > 5) {
    throw new RangeError("Välj ett färdigt exempel med 1–5 kluster.");
  }
  // Preserve the original default example, including its exact sampled points.
  if (count === 3) return createDefaultClusters();

  const exampleParameters: Record<number, ClusterParameters[]> = {
    1: [
      { meanX: 0, meanY: 0, stdX: 1.6, stdY: 0.9, correlation: 0.4 }
    ],
    2: [
      { meanX: -2.8, meanY: -0.5, stdX: 0.9, stdY: 0.8, correlation: 0.2 },
      { meanX: 2.8, meanY: 0.7, stdX: 1.1, stdY: 0.7, correlation: -0.25 }
    ],
    4: [
      { meanX: -3.2, meanY: -2, stdX: 0.7, stdY: 0.6, correlation: 0.2 },
      { meanX: 3.2, meanY: -2, stdX: 1, stdY: 0.75, correlation: -0.3 },
      { meanX: -3.2, meanY: 2.2, stdX: 0.65, stdY: 1, correlation: -0.15 },
      { meanX: 3.2, meanY: 2.2, stdX: 0.9, stdY: 0.8, correlation: 0.35 }
    ],
    5: [
      { meanX: -4, meanY: -2, stdX: 0.65, stdY: 0.7, correlation: 0.2 },
      { meanX: 0, meanY: -2.1, stdX: 0.8, stdY: 0.65, correlation: -0.25 },
      { meanX: 4, meanY: -1.9, stdX: 0.7, stdY: 0.85, correlation: 0.3 },
      { meanX: -2, meanY: 2.2, stdX: 0.85, stdY: 0.7, correlation: -0.2 },
      { meanX: 2.2, meanY: 2.3, stdX: 0.7, stdY: 0.8, correlation: 0.15 }
    ]
  };
  const random = seededRandom(20261004 + count * 97);
  return exampleParameters[count].map((parameters, index) =>
    makeCluster(index + 1, DEFAULT_POINTS_PER_CLUSTER, parameters, random)
  );
}

function squaredDistance(a: Coordinates, b: Coordinates): number {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}

// A random first position followed by the best-separated of several random candidates.
// The bounded search also works when the requested groups overlap heavily.
function spreadPositions(
  count: number,
  sample: () => Coordinates,
  candidateCount: number
): Coordinates[] {
  const positions = [sample()];
  for (let index = 1; index < count; index += 1) {
    let best = sample();
    let bestDistance = -1;
    for (let candidate = 0; candidate < candidateCount; candidate += 1) {
      const position = sample();
      const separation = Math.min(...positions.map((other) => squaredDistance(position, other)));
      if (separation > bestDistance) {
        best = position;
        bestDistance = separation;
      }
    }
    positions.push(best);
  }
  return positions;
}

export function generateClusters(
  count: number,
  pointsPerCluster: number,
  random: RandomSource = Math.random
): GeneratedCluster[] {
  if (!Number.isInteger(count) || count < 1 || count > 5) {
    throw new RangeError("Antalet kluster måste vara ett heltal mellan 1 och 5.");
  }
  if (!Number.isInteger(pointsPerCluster) || pointsPerCluster < 10 || pointsPerCluster > 200) {
    throw new RangeError("Antalet punkter per kluster måste vara ett heltal mellan 10 och 200.");
  }

  const means = spreadPositions(
    count,
    () => ({ x: -4.2 + random() * 8.4, y: -2.8 + random() * 5.6 }),
    24
  );
  return means.map((mean, index) =>
    makeCluster(
      index + 1,
      pointsPerCluster,
      {
        meanX: mean.x,
        meanY: mean.y,
        stdX: 0.6 + random() * 0.85,
        stdY: 0.6 + random() * 0.85,
        correlation: -0.65 + random() * 1.3
      },
      random
    )
  );
}

export function clusterPoints(cluster: GeneratedCluster): DataPoint[] {
  const { meanX, meanY, stdX, stdY, correlation } = cluster;
  if (
    ![meanX, meanY, stdX, stdY, correlation].every(Number.isFinite) ||
    stdX <= 0 || stdY <= 0 || Math.abs(correlation) > 1
  ) {
    throw new RangeError("Klusterparametrarna måste ge en giltig kovariansmatris.");
  }
  const independentYScale = Math.sqrt(1 - correlation ** 2);
  // Reuse the standard-normal samples so parameter edits never resample the data.
  return cluster.samples.map((sample, index) => ({
    id: `${cluster.id}-${index}`,
    sourceClusterId: cluster.id,
    x: meanX + stdX * sample.x,
    y: meanY + stdY * (correlation * sample.x + independentYScale * sample.y)
  }));
}

export function covarianceMatrix(parameters: ClusterParameters): [[number, number], [number, number]] {
  const covariance = parameters.correlation * parameters.stdX * parameters.stdY;
  return [[parameters.stdX ** 2, covariance], [covariance, parameters.stdY ** 2]];
}

export function emptyRun(): KMeansRun {
  return {
    phase: "uninitialized",
    centroids: [],
    assignments: null,
    previousCentroids: null,
    iterations: 0,
    changedPoints: null,
    emptyCentroidIds: []
  };
}

export function initializeRun(
  points: Coordinates[],
  k: number,
  random: RandomSource = Math.random
): KMeansRun {
  const distinctPoints = Array.from(new Map(points.map((point) => [`${point.x},${point.y}`, point])).values());
  if (!Number.isInteger(k) || k < 1 || k > 5 || distinctPoints.length < k) {
    throw new RangeError("Det behövs minst k olika datapunkter för att placera ut medelpunkter.");
  }
  // Start at data positions, without using the source clusters or their means.
  const positions = spreadPositions(k, () => {
    return distinctPoints[Math.floor(random() * distinctPoints.length)];
  }, Math.max(32, Math.min(128, distinctPoints.length)));

  // The first selected position may be sampled again. Replace any duplicates with
  // the best-separated remaining point, including for very small datasets.
  const uniquePositions: Coordinates[] = [];
  for (const position of positions) {
    let next = position;
    if (uniquePositions.some((previous) => squaredDistance(previous, position) === 0)) {
      let bestDistance = -1;
      for (const candidate of distinctPoints) {
        const distance = Math.min(...uniquePositions.map((previous) => squaredDistance(previous, candidate)));
        if (distance > bestDistance) {
          next = candidate;
          bestDistance = distance;
        }
      }
    }
    uniquePositions.push(next);
  }

  return {
    ...emptyRun(),
    phase: "assign",
    centroids: uniquePositions.map(({ x, y }, id) => ({ id, x, y }))
  };
}

export function assignPoints(points: Coordinates[], centroids: Centroid[]): number[] {
  if (centroids.length === 0) {
    throw new RangeError("Placera ut medelpunkter före tilldelningen.");
  }
  return points.map((point) => {
    let nearest = centroids[0];
    let nearestDistance = squaredDistance(point, nearest);
    for (const centroid of centroids.slice(1)) {
      const distance = squaredDistance(point, centroid);
      // Equal distances always go to the first centroid, making ties stable.
      if (distance < nearestDistance) {
        nearest = centroid;
        nearestDistance = distance;
      }
    }
    return nearest.id;
  });
}

export function createClassificationPoint(
  points: Coordinates[],
  random: RandomSource = Math.random
): ClassificationPoint {
  if (points.length === 0) {
    throw new RangeError("Det behövs datapunkter för att skapa ett nytt exempel.");
  }
  // Sample inside the data bounds included by plotRange, so the axes stay fixed.
  let xMin = -5;
  let xMax = 5;
  let yMin = -4;
  let yMax = 4;
  for (const point of points) {
    xMin = Math.min(xMin, point.x);
    xMax = Math.max(xMax, point.x);
    yMin = Math.min(yMin, point.y);
    yMax = Math.max(yMax, point.y);
  }
  return {
    x: xMin + random() * (xMax - xMin),
    y: yMin + random() * (yMax - yMin),
    centroidId: null
  };
}

export function classifyPoint(point: Coordinates, run: KMeansRun): number {
  if (run.phase !== "converged") {
    throw new RangeError("Grupperna måste vara stabila före klassificeringen.");
  }
  return assignPoints([point], run.centroids)[0];
}

export function advanceRun(points: Coordinates[], run: KMeansRun): KMeansRun {
  if (points.length === 0 || run.phase === "uninitialized" || run.phase === "converged") {
    return run;
  }
  if (run.phase === "assign") {
    const assignments = assignPoints(points, run.centroids);
    const changedPoints = run.assignments === null
      ? null
      : assignments.reduce((count, assignment, index) => count + Number(assignment !== run.assignments![index]), 0);
    return {
      ...run,
      assignments,
      changedPoints,
      previousCentroids: null,
      emptyCentroidIds: run.centroids.filter((centroid) => !assignments.includes(centroid.id)).map((centroid) => centroid.id),
      phase: changedPoints === 0 && run.iterations > 0 ? "converged" : "update"
    };
  }

  if (run.assignments === null || run.assignments.length !== points.length) {
    throw new RangeError("Tilldela alla datapunkter före flytten av medelpunkterna.");
  }
  const emptyCentroidIds: number[] = [];
  const centroids = run.centroids.map((centroid) => {
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    points.forEach((point, index) => {
      if (run.assignments![index] === centroid.id) {
        sumX += point.x;
        sumY += point.y;
        count += 1;
      }
    });
    if (count === 0) {
      emptyCentroidIds.push(centroid.id);
      return { ...centroid };
    }
    return { id: centroid.id, x: sumX / count, y: sumY / count };
  });
  return {
    ...run,
    phase: "assign",
    centroids,
    previousCentroids: run.centroids,
    iterations: run.iterations + 1,
    emptyCentroidIds
  };
}

export interface PlotRange {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

export function plotRange(points: Coordinates[], aspectRatio: number): PlotRange {
  let xMin = -5;
  let xMax = 5;
  let yMin = -4;
  let yMax = 4;
  for (const point of points) {
    xMin = Math.min(xMin, point.x);
    xMax = Math.max(xMax, point.x);
    yMin = Math.min(yMin, point.y);
    yMax = Math.max(yMax, point.y);
  }
  const ySpan = Math.max(10, (yMax - yMin) * 1.16, (xMax - xMin) * 1.16 / aspectRatio);
  const xSpan = ySpan * aspectRatio;
  const centerX = (xMin + xMax) / 2;
  const centerY = (yMin + yMax) / 2;
  return {
    xMin: centerX - xSpan / 2,
    xMax: centerX + xSpan / 2,
    yMin: centerY - ySpan / 2,
    yMax: centerY + ySpan / 2
  };
}
