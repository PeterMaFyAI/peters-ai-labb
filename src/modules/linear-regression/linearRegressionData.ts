export interface DataPoint {
  index: number;
  area: number;
  price: number;
}

interface FeatureScalingStats {
  meanX: number;
  stdX: number;
}

const csvData = `index,Area (kvm),Pris (miljoner kr)
0,111.0,3.57
1,135.0,3.99
2,73.0,2.35
3,99.0,2.65
4,88.0,3.18
5,119.0,3.44
6,23.0,0.71
7,143.0,4.66
8,86.0,2.71
9,124.0,4.53
10,86.0,2.54
11,99.0,3.4
12,83.0,2.83
13,45.0,1.35
14,53.0,2.1
15,74.0,2.52
16,77.0,2.89
17,117.0,4.11
18,36.0,2.08
19,98.0,3.69
20,112.0,3.6
21,75.0,2.53
22,40.0,1.88
23,132.0,4.51
24,113.0,3.93
25,51.0,1.39
26,77.0,2.65
27,79.0,2.5
28,37.0,1.54
29,95.0,3.17
30,102.0,3.68
31,61.0,2.28
32,79.0,2.96
33,122.0,3.87
34,64.0,1.99
35,43.0,1.52
36,54.0,1.96
37,54.0,2.08
38,122.0,4.69
39,74.0,2.56
40,42.0,1.32
41,51.0,1.78
42,101.0,3.24
43,63.0,2.38
44,41.0,1.12
45,40.0,1.57
46,105.0,3.55
47,98.0,3.36
48,45.0,1.52
49,81.0,2.71`;

function parseCSV(csv: string): DataPoint[] {
  const lines = csv.trim().split("\n");
  const parsed: DataPoint[] = [];

  for (let index = 1; index < lines.length; index += 1) {
    const parts = lines[index].split(",");
    parsed.push({
      index: Number(parts[0]),
      area: Number(parts[1]),
      price: Number(parts[2])
    });
  }

  return parsed;
}

export const dataPoints = parseCSV(csvData);

export function calculateMSE(weight: number, bias: number, points: DataPoint[] = dataPoints): number {
  const squaredErrorSum = points.reduce((sum, point) => {
    const predicted = weight * point.area + bias;
    const error = predicted - point.price;
    return sum + error ** 2;
  }, 0);

  return squaredErrorSum / points.length;
}

export function calculateOptimalParameters(points: DataPoint[] = dataPoints): {
  weight: number;
  bias: number;
} {
  const n = points.length;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;

  points.forEach((point) => {
    sumX += point.area;
    sumY += point.price;
    sumXY += point.area * point.price;
    sumXX += point.area * point.area;
  });

  const weight = (n * sumXY - sumX * sumY) / (n * sumXX - sumX ** 2);
  const bias = (sumY - weight * sumX) / n;

  return { weight, bias };
}

function calculateFeatureScalingStats(points: DataPoint[] = dataPoints): FeatureScalingStats {
  const n = points.length;
  const meanX = points.reduce((sum, point) => sum + point.area, 0) / n;
  const varianceX =
    points.reduce((sum, point) => {
      const centered = point.area - meanX;
      return sum + centered * centered;
    }, 0) / n;

  const stdX = Math.sqrt(varianceX) || 1;

  return { meanX, stdX };
}

export const featureScalingStats = calculateFeatureScalingStats(dataPoints);

export function toStandardizedWeightBias(
  weight: number,
  bias: number,
  stats: FeatureScalingStats = featureScalingStats
): { weightStd: number; biasStd: number } {
  const weightStd = weight * stats.stdX;
  const biasStd = weight * stats.meanX + bias;
  return { weightStd, biasStd };
}

export function toOriginalWeightBias(
  weightStd: number,
  biasStd: number,
  stats: FeatureScalingStats = featureScalingStats
): { weight: number; bias: number } {
  const weight = weightStd / stats.stdX;
  const bias = biasStd - (weightStd * stats.meanX) / stats.stdX;
  return { weight, bias };
}

export function calculateGradientStandardized(
  weightStd: number,
  biasStd: number,
  stats: FeatureScalingStats = featureScalingStats,
  points: DataPoint[] = dataPoints
): { gradientWeightStd: number; gradientBiasStd: number } {
  const n = points.length;
  let gradientWeightStdAccumulator = 0;
  let gradientBiasStdAccumulator = 0;

  points.forEach((point) => {
    const standardizedArea = (point.area - stats.meanX) / stats.stdX;
    const prediction = weightStd * standardizedArea + biasStd;
    const error = prediction - point.price;

    gradientWeightStdAccumulator += error * standardizedArea;
    gradientBiasStdAccumulator += error;
  });

  return {
    gradientWeightStd: (2 / n) * gradientWeightStdAccumulator,
    gradientBiasStd: (2 / n) * gradientBiasStdAccumulator
  };
}
