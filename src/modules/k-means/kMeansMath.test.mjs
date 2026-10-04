import assert from "node:assert/strict";
import test from "node:test";
import {
  advanceRun,
  assignPoints,
  classifyPoint,
  clusterPoints,
  covarianceMatrix,
  createDefaultClusters,
  createClassificationPoint,
  createExampleClusters,
  emptyRun,
  generateClusters,
  initializeRun,
  plotRange,
  seededRandom
} from "./kMeansMath.ts";

const close = (actual, expected, tolerance = 1e-10) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);

const defaultPoints = () => createDefaultClusters().flatMap(clusterPoints);

function objective(points, run) {
  return points.reduce((sum, point, index) => {
    const centroid = run.centroids.find((centroid) => centroid.id === run.assignments[index]);
    return sum + (point.x - centroid.x) ** 2 + (point.y - centroid.y) ** 2;
  }, 0);
}

test("start example is reproducible and has three different 2D distributions", () => {
  const clusters = createDefaultClusters();
  assert.deepEqual(clusters, createDefaultClusters());
  assert.equal(clusters.length, 3);
  assert.equal(defaultPoints().length, 180);
  assert.equal(new Set(defaultPoints().map((point) => point.id)).size, 180);
  assert.equal(new Set(clusters.map((cluster) => JSON.stringify(covarianceMatrix(cluster)))).size, 3);
});

test("teaching examples provide reproducible editable clouds for every count, retaining the original default", () => {
  for (let count = 1; count <= 5; count += 1) {
    const first = createExampleClusters(count);
    const second = createExampleClusters(count);
    assert.deepEqual(first, second);
    assert.equal(first.length, count);
    assert.equal(first.flatMap(clusterPoints).length, count * 60);
    assert.notEqual(first[0].samples, second[0].samples);
    const savedPoint = { ...second[0].samples[0] };
    first[0].samples[0].x += 1;
    assert.deepEqual(second[0].samples[0], savedPoint);
  }
  assert.deepEqual(createExampleClusters(3), createDefaultClusters());
  for (const invalidCount of [0, 6, 2.5, NaN]) {
    assert.throws(() => createExampleClusters(invalidCount), RangeError);
  }
});

test("every teaching example has a clear spatial structure and converges with its matching k", () => {
  for (let count = 1; count <= 5; count += 1) {
    const clusters = createExampleClusters(count);
    const points = clusters.flatMap(clusterPoints);
    const centers = clusters.map((cluster) => ({ id: cluster.id, x: cluster.meanX, y: cluster.meanY }));
    const nearestSource = assignPoints(points, centers);
    const correctlyPlaced = points.filter((point, index) => point.sourceClusterId === nearestSource[index]).length;
    assert.ok(correctlyPlaced / points.length > 0.85, `example ${count} is too mixed to distinguish its groups`);
    for (let seed = 1; seed <= 5; seed += 1) {
      let run = initializeRun(points, count, seededRandom(seed));
      let steps = 0;
      while (run.phase !== "converged" && steps < 100) {
        run = advanceRun(points, run);
        steps += 1;
      }
      assert.equal(run.phase, "converged", `example ${count} did not converge for start ${seed}`);
      assert.ok(run.centroids.every((centroid) => run.assignments.includes(centroid.id)));
    }
  }
});

test("parameter edits preserve samples, identities, other clusters, and exact translations", () => {
  const clusters = createDefaultClusters();
  const original = clusterPoints(clusters[0]);
  const edited = { ...clusters[0], meanX: clusters[0].meanX + 2, meanY: clusters[0].meanY - 1 };
  const moved = clusterPoints(edited);
  assert.equal(edited.samples, clusters[0].samples);
  moved.forEach((point, index) => {
    assert.equal(point.id, original[index].id);
    close(point.x - original[index].x, 2);
    close(point.y - original[index].y, -1);
  });
  assert.deepEqual(clusterPoints(clusters[1]), clusterPoints(createDefaultClusters()[1]));
});

test("full covariance shapes points correctly, including correlated and independent axes", () => {
  const cluster = { id: 1, meanX: 5, meanY: 10, stdX: 2, stdY: 3, correlation: 0.6, samples: [{ x: 1, y: 2 }] };
  const [point] = clusterPoints(cluster);
  close(point.x, 7);
  close(point.y, 16.6);
  const matrix = covarianceMatrix(cluster);
  close(matrix[0][0], 4);
  close(matrix[1][1], 9);
  close(matrix[0][1], 3.6);
  close(matrix[1][0], 3.6);
  close(clusterPoints({ ...cluster, correlation: 0 })[0].y, 16);
  close(clusterPoints({ ...cluster, correlation: -1 })[0].y, 7);
  assert.throws(() => clusterPoints({ ...cluster, correlation: 1.1 }), RangeError);
  assert.throws(() => clusterPoints({ ...cluster, stdX: 0 }), RangeError);
  assert.throws(() => clusterPoints({ ...cluster, meanY: NaN }), RangeError);
});

test("sampled 2D clouds reproduce the requested mean and covariance statistically", () => {
  const random = seededRandom(481);
  const parameters = { meanX: 5, meanY: -2, stdX: 2, stdY: 3, correlation: -0.6 };
  const points = Array.from({ length: 150 }, () =>
    clusterPoints({ ...generateClusters(1, 200, random)[0], ...parameters })
  ).flat();
  const meanX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
  const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
  const varianceX = points.reduce((sum, point) => sum + (point.x - meanX) ** 2, 0) / points.length;
  const varianceY = points.reduce((sum, point) => sum + (point.y - meanY) ** 2, 0) / points.length;
  const covariance = points.reduce((sum, point) => sum + (point.x - meanX) * (point.y - meanY), 0) / points.length;
  close(meanX, 5, 0.06);
  close(meanY, -2, 0.06);
  close(varianceX, 4, 0.12);
  close(varianceY, 9, 0.25);
  close(covariance, -3.6, 0.15);
});

test("generate 1–5 clusters and reject invalid counts without unbounded sampling", () => {
  for (let count = 1; count <= 5; count += 1) {
    const clusters = generateClusters(count, 10, seededRandom(count));
    assert.equal(clusters.length, count);
    assert.equal(clusters.flatMap(clusterPoints).length, count * 10);
  }
  assert.throws(() => generateClusters(0, 60), RangeError);
  assert.throws(() => generateClusters(6, 60), RangeError);
  assert.throws(() => generateClusters(2, 9), RangeError);
  assert.throws(() => generateClusters(2, 201), RangeError);
  assert.throws(() => generateClusters(2, 10.5), RangeError);
});

test("initialization uses distinct well-separated data positions and leaves assignments unset", () => {
  const points = defaultPoints();
  const run = initializeRun(points, 5, seededRandom(42));
  assert.equal(run.phase, "assign");
  assert.equal(run.assignments, null);
  assert.equal(run.centroids.length, 5);
  assert.equal(new Set(run.centroids.map((point) => `${point.x},${point.y}`)).size, 5);
  for (const centroid of run.centroids) {
    assert.ok(points.some((point) => point.x === centroid.x && point.y === centroid.y));
  }
  const minSeparation = Math.min(...run.centroids.flatMap((point, index) =>
    run.centroids.slice(index + 1).map((other) => Math.hypot(point.x - other.x, point.y - other.y))
  ));
  assert.ok(minSeparation > 2);
  assert.throws(() => initializeRun([], 1), RangeError);
  assert.throws(() => initializeRun([{ x: 0, y: 0 }, { x: 0, y: 0 }], 2), RangeError);
  const tiny = [{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }];
  assert.equal(new Set(initializeRun(tiny, 3, () => 0).centroids.map((point) => point.x)).size, 3);
});

test("assignment uses Euclidean distance and resolves ties consistently", () => {
  const centroids = [{ id: 0, x: -1, y: 0 }, { id: 1, x: 1, y: 0 }];
  const assignments = assignPoints([{ x: -2, y: 2 }, { x: 0, y: 4 }, { x: 3, y: 0 }], centroids);
  assert.deepEqual(assignments, [0, 0, 1]);
  assert.throws(() => assignPoints([{ x: 0, y: 0 }], []), RangeError);
});

test("new examples stay inside the existing plot at any aspect ratio and do not change training data", () => {
  const points = [...defaultPoints(), { x: 18, y: -12 }];
  const original = structuredClone(points);
  const random = seededRandom(62);
  for (let index = 0; index < 100; index += 1) {
    const point = createClassificationPoint(points, random);
    assert.equal(point.centroidId, null);
    for (const ratio of [0.7, 1, 1.8, 3]) {
      const range = plotRange(points, ratio);
      assert.ok(point.x > range.xMin && point.x < range.xMax);
      assert.ok(point.y > range.yMin && point.y < range.yMax);
    }
  }
  assert.deepEqual(points, original);
  assert.throws(() => createClassificationPoint([]), RangeError);
});

test("new examples use the nearest stable centroid without changing the learned groups", () => {
  const points = [{ x: -4, y: 0 }, { x: -2, y: 0 }, { x: 2, y: 0 }, { x: 4, y: 0 }];
  let run = { ...emptyRun(), phase: "assign", centroids: [{ id: 0, x: -4, y: 0 }, { id: 1, x: 4, y: 0 }] };
  assert.throws(() => classifyPoint({ x: 0, y: 0 }, run), RangeError);
  for (let step = 0; step < 10 && run.phase !== "converged"; step += 1) {
    run = advanceRun(points, run);
  }
  assert.equal(run.phase, "converged");
  const original = structuredClone(run);
  assert.equal(classifyPoint({ x: -1, y: 2 }, run), 0);
  assert.equal(classifyPoint({ x: 1, y: 2 }, run), 1);
  assert.equal(classifyPoint({ x: 0, y: 3 }, run), 0);
  assert.deepEqual(run, original);
});

test("each click performs exactly one phase and retains colors during the mean update", () => {
  const points = [{ x: 0, y: 0 }, { x: 2, y: 2 }, { x: 8, y: 0 }, { x: 10, y: 2 }];
  const initial = { ...emptyRun(), phase: "assign", centroids: [{ id: 0, x: 0, y: 0 }, { id: 1, x: 10, y: 2 }] };
  const assigned = advanceRun(points, initial);
  assert.equal(assigned.phase, "update");
  assert.deepEqual(assigned.centroids, initial.centroids);
  assert.deepEqual(assigned.assignments, [0, 0, 1, 1]);
  assert.equal(assigned.changedPoints, null);
  assert.equal(assigned.iterations, 0);
  const moved = advanceRun(points, assigned);
  assert.equal(moved.phase, "assign");
  assert.deepEqual(moved.centroids, [{ id: 0, x: 1, y: 1 }, { id: 1, x: 9, y: 1 }]);
  assert.equal(moved.assignments, assigned.assignments);
  assert.deepEqual(moved.previousCentroids, initial.centroids);
  assert.equal(moved.iterations, 1);
  const stable = advanceRun(points, moved);
  assert.equal(stable.phase, "converged");
  assert.equal(stable.changedPoints, 0);
  assert.equal(stable.previousCentroids, null);
  assert.equal(advanceRun(points, stable), stable);
});

test("empty groups retain their positions and are reported without invalid means", () => {
  const points = [{ x: 0, y: 0 }, { x: 2, y: 0 }];
  const run = { ...emptyRun(), phase: "update", centroids: [{ id: 0, x: 0, y: 0 }, { id: 1, x: 100, y: 100 }], assignments: [0, 0] };
  const moved = advanceRun(points, run);
  assert.deepEqual(moved.centroids, [{ id: 0, x: 1, y: 0 }, { id: 1, x: 100, y: 100 }]);
  assert.deepEqual(moved.emptyCentroidIds, [1]);
  assert.equal(advanceRun(points, moved).phase, "converged");
});

test("reset and empty data are safe; k can differ from the generating cluster count", () => {
  const points = generateClusters(1, 20, seededRandom(17)).flatMap(clusterPoints);
  assert.equal(initializeRun(points, 5, seededRandom(18)).centroids.length, 5);
  const reset = emptyRun();
  assert.equal(advanceRun(points, reset), reset);
  const run = initializeRun(points, 2);
  assert.equal(advanceRun([], run), run);
});

test("many independent starts converge and squared error never increases in either phase", () => {
  for (let seed = 1; seed <= 20; seed += 1) {
    const points = generateClusters(1 + seed % 5, 30, seededRandom(seed)).flatMap(clusterPoints);
    let run = initializeRun(points, 1 + (seed * 3) % 5, seededRandom(seed + 100));
    let previousError = Infinity;
    let steps = 0;
    while (run.phase !== "converged" && steps < 200) {
      run = advanceRun(points, run);
      const error = objective(points, run);
      assert.ok(error <= previousError + 1e-8, `error increased at seed ${seed}, phase ${run.phase}`);
      previousError = error;
      steps += 1;
    }
    assert.equal(run.phase, "converged", `did not converge for seed ${seed}`);
    assert.deepEqual(assignPoints(points, run.centroids), run.assignments);
  }
});

test("plot scales stay equal and contain all points, even for narrow screens and extreme edits", () => {
  const clusters = createDefaultClusters();
  const points = clusterPoints({ ...clusters[0], meanX: 8, meanY: -8, stdX: 3, stdY: 3, correlation: -0.95 });
  for (const ratio of [0.7, 1, 1.8, 3]) {
    const range = plotRange(points, ratio);
    close((range.xMax - range.xMin) / (range.yMax - range.yMin), ratio);
    assert.ok(points.every((point) => point.x > range.xMin && point.x < range.xMax && point.y > range.yMin && point.y < range.yMax));
  }
});
