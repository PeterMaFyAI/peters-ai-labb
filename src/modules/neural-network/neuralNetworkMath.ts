import {
  type IrrigationSample,
  type Label,
  type NetworkParams,
  LABEL_TO_TARGET,
  trainingData
} from "./neuralNetworkData";

export interface NormalizationStats {
  moistureMean: number;
  moistureStd: number;
  temperatureMean: number;
  temperatureStd: number;
}

export interface ForwardPass {
  preHidden: number[];
  actHidden: number[];
  output: number;
}

export interface ForwardPassModel extends ForwardPass {
  rawInputs: [number, number];
  modelInputs: [number, number];
}

export interface Gradients {
  W_IH: number[][];
  B_H: number[];
  W_HO: number[];
  B_O: number;
  loss: number;
}

export interface BatchGradients extends Gradients {
  lossSum: number;
  batchSize: number;
}

export interface EvaluationResult {
  mse: number;
  accuracy: number;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function variance(values: number[], mu: number): number {
  return values.reduce((sum, value) => sum + (value - mu) * (value - mu), 0) / values.length;
}

function computeNormalizationStats(): NormalizationStats {
  const moistureValues = trainingData.map((example) => example.moisture);
  const temperatureValues = trainingData.map((example) => example.temperature);
  const moistureMean = mean(moistureValues);
  const temperatureMean = mean(temperatureValues);
  const moistureStd = Math.sqrt(variance(moistureValues, moistureMean)) || 1;
  const temperatureStd = Math.sqrt(variance(temperatureValues, temperatureMean)) || 1;

  return {
    moistureMean,
    moistureStd,
    temperatureMean,
    temperatureStd
  };
}

export const NORMALIZATION_STATS = computeNormalizationStats();

export function cloneParams(source: NetworkParams): NetworkParams {
  return {
    W_IH: source.W_IH.map((row) => row.slice()),
    B_H: source.B_H.slice(),
    W_HO: source.W_HO.slice(),
    B_O: source.B_O
  };
}

export function normalizeInputs(
  inputs: [number, number],
  stats: NormalizationStats = NORMALIZATION_STATS
): [number, number] {
  return [
    (inputs[0] - stats.moistureMean) / stats.moistureStd,
    (inputs[1] - stats.temperatureMean) / stats.temperatureStd
  ];
}

export function forwardPassCore(inputs: [number, number], params: NetworkParams): ForwardPass {
  const preHidden: number[] = [];
  const actHidden: number[] = [];

  for (let j = 0; j < 3; j += 1) {
    const pre = inputs[0] * params.W_IH[0][j] + inputs[1] * params.W_IH[1][j] + params.B_H[j];
    preHidden.push(pre);
    actHidden.push(Math.max(0, pre));
  }

  const output = actHidden.reduce((sum, activation, index) => sum + activation * params.W_HO[index], params.B_O);

  return { preHidden, actHidden, output };
}

export function forwardPassModel(rawInputs: [number, number], params: NetworkParams): ForwardPassModel {
  const modelInputs = normalizeInputs(rawInputs);
  const forward = forwardPassCore(modelInputs, params);
  return {
    ...forward,
    rawInputs,
    modelInputs
  };
}

export function computeGradients(
  forward: ForwardPass,
  modelInputs: [number, number],
  target: number,
  params: NetworkParams
): Gradients {
  const gradients: Gradients = {
    W_IH: [new Array(3).fill(0), new Array(3).fill(0)],
    B_H: new Array(3).fill(0),
    W_HO: new Array(3).fill(0),
    B_O: 0,
    loss: 0
  };

  const error = forward.output - target;
  gradients.B_O = error;
  gradients.loss = 0.5 * error * error;

  for (let j = 0; j < 3; j += 1) {
    gradients.W_HO[j] = error * forward.actHidden[j];
    const reluDerivative = forward.preHidden[j] > 0 ? 1 : 0;
    const deltaH = error * params.W_HO[j] * reluDerivative;
    gradients.B_H[j] = deltaH;
    gradients.W_IH[0][j] = deltaH * modelInputs[0];
    gradients.W_IH[1][j] = deltaH * modelInputs[1];
  }

  return gradients;
}

export function computeBatchGradients(
  batch: IrrigationSample[],
  params: NetworkParams
): BatchGradients {
  const gradients: BatchGradients = {
    W_IH: [new Array(3).fill(0), new Array(3).fill(0)],
    B_H: new Array(3).fill(0),
    W_HO: new Array(3).fill(0),
    B_O: 0,
    loss: 0,
    lossSum: 0,
    batchSize: batch.length
  };

  if (batch.length === 0) {
    return gradients;
  }

  batch.forEach((example) => {
    const forward = forwardPassModel([example.moisture, example.temperature], params);
    const localGradients = computeGradients(
      forward,
      forward.modelInputs,
      LABEL_TO_TARGET[example.label],
      params
    );

    gradients.B_O += localGradients.B_O;
    gradients.lossSum += localGradients.loss;

    for (let j = 0; j < 3; j += 1) {
      gradients.W_HO[j] += localGradients.W_HO[j];
      gradients.B_H[j] += localGradients.B_H[j];
      gradients.W_IH[0][j] += localGradients.W_IH[0][j];
      gradients.W_IH[1][j] += localGradients.W_IH[1][j];
    }
  });

  gradients.B_O /= batch.length;
  gradients.loss = gradients.lossSum / batch.length;

  for (let j = 0; j < 3; j += 1) {
    gradients.W_HO[j] /= batch.length;
    gradients.B_H[j] /= batch.length;
    gradients.W_IH[0][j] /= batch.length;
    gradients.W_IH[1][j] /= batch.length;
  }

  return gradients;
}

export function applyGradients(params: NetworkParams, gradients: Gradients, learningRate: number): NetworkParams {
  const next = cloneParams(params);

  next.B_O -= learningRate * gradients.B_O;

  for (let j = 0; j < 3; j += 1) {
    next.W_HO[j] -= learningRate * gradients.W_HO[j];
    next.B_H[j] -= learningRate * gradients.B_H[j];
    next.W_IH[0][j] -= learningRate * gradients.W_IH[0][j];
    next.W_IH[1][j] -= learningRate * gradients.W_IH[1][j];
  }

  return next;
}

export function evaluateDataset(data: IrrigationSample[], params: NetworkParams): EvaluationResult {
  let totalError = 0;
  let correct = 0;

  data.forEach((example) => {
    const forward = forwardPassModel([example.moisture, example.temperature], params);
    const target = LABEL_TO_TARGET[example.label];
    const error = forward.output - target;
    totalError += error * error;

    const predicted: Label = forward.output > 0 ? "ja" : "nej";
    if (predicted === example.label) {
      correct += 1;
    }
  });

  return {
    mse: totalError / data.length,
    accuracy: (correct / data.length) * 100
  };
}

export function shuffle<T>(items: T[]): T[] {
  const result = items.slice();

  for (let i = result.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }

  return result;
}
