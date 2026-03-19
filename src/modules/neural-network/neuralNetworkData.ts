export type Label = "ja" | "nej";

export interface IrrigationSample {
  id: number;
  moisture: number;
  temperature: number;
  label: Label;
}

export interface NetworkParams {
  W_IH: number[][];
  B_H: number[];
  W_HO: number[];
  B_O: number;
}

export const INITIAL_PARAMS: NetworkParams = {
  W_IH: [
    [0.63, -0.92, 0.47],
    [-0.58, 0.31, -0.79]
  ],
  B_H: [0.28, -0.41, 0.22],
  W_HO: [1.17, -0.54, 0.83],
  B_O: -0.46
};

export const LEARNING_RATE_MANUAL = 0.02;
export const LEARNING_RATE_AUTO = 0.005;
export const BATCH_SIZE = 1;

export const trainingData: IrrigationSample[] = [
  { id: 1, moisture: 51, temperature: 19, label: "nej" },
  { id: 2, moisture: 60, temperature: 35, label: "ja" },
  { id: 3, moisture: 16, temperature: 17, label: "ja" },
  { id: 4, moisture: 78, temperature: 18, label: "nej" },
  { id: 5, moisture: 56, temperature: 33, label: "ja" },
  { id: 6, moisture: 17, temperature: 31, label: "ja" },
  { id: 7, moisture: 37, temperature: 16, label: "nej" },
  { id: 8, moisture: 21, temperature: 28, label: "ja" },
  { id: 9, moisture: 63, temperature: 17, label: "nej" },
  { id: 10, moisture: 40, temperature: 17, label: "nej" },
  { id: 11, moisture: 80, temperature: 28, label: "nej" },
  { id: 12, moisture: 17, temperature: 33, label: "ja" },
  { id: 13, moisture: 25, temperature: 22, label: "ja" },
  { id: 14, moisture: 76, temperature: 28, label: "nej" },
  { id: 15, moisture: 46, temperature: 20, label: "nej" },
  { id: 16, moisture: 50, temperature: 24, label: "nej" },
  { id: 17, moisture: 76, temperature: 28, label: "nej" },
  { id: 18, moisture: 27, temperature: 17, label: "ja" },
  { id: 19, moisture: 21, temperature: 19, label: "ja" },
  { id: 20, moisture: 22, temperature: 23, label: "ja" },
  { id: 21, moisture: 65, temperature: 20, label: "nej" },
  { id: 22, moisture: 28, temperature: 23, label: "ja" },
  { id: 23, moisture: 35, temperature: 34, label: "ja" },
  { id: 24, moisture: 62, temperature: 26, label: "nej" },
  { id: 25, moisture: 46, temperature: 31, label: "ja" },
  { id: 26, moisture: 24, temperature: 21, label: "ja" },
  { id: 27, moisture: 61, temperature: 24, label: "nej" },
  { id: 28, moisture: 64, temperature: 25, label: "nej" },
  { id: 29, moisture: 59, temperature: 21, label: "nej" },
  { id: 30, moisture: 56, temperature: 34, label: "ja" },
  { id: 31, moisture: 67, temperature: 21, label: "nej" },
  { id: 32, moisture: 67, temperature: 17, label: "nej" },
  { id: 33, moisture: 67, temperature: 23, label: "nej" },
  { id: 34, moisture: 58, temperature: 34, label: "ja" },
  { id: 35, moisture: 53, temperature: 19, label: "nej" },
  { id: 36, moisture: 69, temperature: 30, label: "nej" },
  { id: 37, moisture: 55, temperature: 17, label: "nej" },
  { id: 38, moisture: 69, temperature: 30, label: "nej" },
  { id: 39, moisture: 60, temperature: 24, label: "nej" },
  { id: 40, moisture: 56, temperature: 20, label: "nej" },
  { id: 41, moisture: 29, temperature: 34, label: "ja" },
  { id: 42, moisture: 63, temperature: 20, label: "nej" },
  { id: 43, moisture: 38, temperature: 22, label: "nej" },
  { id: 44, moisture: 56, temperature: 18, label: "nej" },
  { id: 45, moisture: 20, temperature: 15, label: "ja" },
  { id: 46, moisture: 14, temperature: 16, label: "ja" },
  { id: 47, moisture: 16, temperature: 35, label: "ja" },
  { id: 48, moisture: 58, temperature: 27, label: "ja" },
  { id: 49, moisture: 61, temperature: 32, label: "ja" },
  { id: 50, moisture: 59, temperature: 26, label: "ja" }
];

export const testData: IrrigationSample[] = [
  { id: 1, moisture: 12, temperature: 29, label: "ja" },
  { id: 2, moisture: 55, temperature: 20, label: "nej" },
  { id: 3, moisture: 24, temperature: 30, label: "ja" },
  { id: 4, moisture: 17, temperature: 21, label: "ja" },
  { id: 5, moisture: 46, temperature: 19, label: "nej" },
  { id: 6, moisture: 41, temperature: 27, label: "ja" },
  { id: 7, moisture: 60, temperature: 30, label: "ja" },
  { id: 8, moisture: 20, temperature: 20, label: "ja" },
  { id: 9, moisture: 67, temperature: 27, label: "nej" },
  { id: 10, moisture: 80, temperature: 23, label: "nej" }
];

export const LABEL_TO_TARGET: Record<Label, 1 | -1> = {
  ja: 1,
  nej: -1
};

export const labelFromOutput = (output: number): Label => (output > 0 ? "ja" : "nej");
