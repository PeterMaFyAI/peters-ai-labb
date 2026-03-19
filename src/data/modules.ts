import type { LearningModule } from "../types/module";
import knnImage from "../assets/module-images/knn.png";
import linearRegressionImage from "../assets/module-images/linear-regression.png";
import neuralNetworkImage from "../assets/module-images/neural-network.png";

export const visualizationModules: LearningModule[] = [
  {
    id: "knn",
    title: "K-närmaste grannar",
    slug: "k-narmaste-grannar",
    imageUrl: knnImage,
    description:
      "Interaktiv KNN i 2D och 3D med klassificering, k-val och klustergenerering.",
    category: "Klassificering",
    targetGroup: "Gymnasiet / vuxenutbildning",
    route: "/visualiseringar/k-narmaste-grannar",
    type: "visualization",
    status: "ready",
    keywords: ["klassificering", "avstånd", "etiketter", "data"]
  },
  {
    id: "linear-regression",
    title: "Linjär regression",
    slug: "linjar-regression",
    imageUrl: linearRegressionImage,
    description:
      "Utforska regressionslinje, MSE och en separat undersida för gradientnedstigning.",
    category: "Regression",
    targetGroup: "Gymnasiet / vuxenutbildning",
    route: "/visualiseringar/linjar-regression",
    type: "visualization",
    status: "ready",
    keywords: ["trend", "prediktion", "modell", "linje", "gradientnedstigning"]
  },
  {
    id: "neural-network",
    title: "Neuralt nätverk",
    slug: "neuralt-natverk",
    imageUrl: neuralNetworkImage,
    description:
      "Interaktiv modul med framåtpass, bakåtpropagering och automatisk träning för bevattningsprediktion.",
    category: "Djupinlärning",
    targetGroup: "Gymnasiet / vuxenutbildning",
    route: "/visualiseringar/neuralt-natverk",
    type: "visualization",
    status: "ready",
    keywords: ["noder", "lager", "vikter", "träning"]
  }
];

export const labModules: LearningModule[] = [];

export const plannedLabModules: LearningModule[] = [
  {
    id: "future-lab-1",
    title: "Kommande laboration",
    slug: "kommande-laboration",
    description:
      "Strukturen är förberedd för interaktiva laborationer som publiceras i nästa steg.",
    category: "Laboration",
    targetGroup: "Gymnasiet / vuxenutbildning",
    route: "/laborationer",
    type: "lab",
    status: "planned",
    keywords: ["övning", "experiment", "interaktiv"]
  }
];

export const featuredResources: LearningModule[] = [...visualizationModules, plannedLabModules[0]];
