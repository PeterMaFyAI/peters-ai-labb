import type { LearningModule } from "../types/module";
import knnImage from "../assets/module-images/knn.png";
import linearRegressionImage from "../assets/module-images/linear-regression.png";
import neuralNetworkImage from "../assets/module-images/neural-network.png";
import kMeansImage from "../assets/module-images/k-means.png";
import avstandImage from "../assets/module-images/avstand.png";
import normaliseringImage from "../assets/module-images/normalisering.png";

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
  },
  {
    id: "k-means",
    title: "K-means",
    slug: "k-means",
    imageUrl: kMeansImage,
    description: "Visualiseringen kommer snart.",
    category: "Klustring",
    targetGroup: "Gymnasiet / vuxenutbildning",
    route: "/visualiseringar/k-means",
    type: "visualization",
    status: "planned",
    keywords: ["klustring", "centroider", "gruppering", "distance"]
  },
  {
    id: "avstand",
    title: "Avstånd",
    slug: "avstand",
    imageUrl: avstandImage,
    description:
      "Interaktiv visualisering av euklidiskt avstånd, Manhattan-avstånd och cosinuslikhet i 2D/3D.",
    category: "Matematik",
    targetGroup: "Gymnasiet / vuxenutbildning",
    route: "/visualiseringar/avstand",
    type: "visualization",
    status: "ready",
    keywords: ["avstånd", "distans", "geometri", "mått"]
  },
  {
    id: "normalisering",
    title: "Normalisering",
    slug: "normalisering",
    imageUrl: normaliseringImage,
    description:
      "Jämför egenskaper före och efter standard score-normalisering med punktdiagram och tabell.",
    category: "Matematik",
    targetGroup: "Gymnasiet / vuxenutbildning",
    route: "/visualiseringar/normalisering",
    type: "visualization",
    status: "ready",
    keywords: ["skalning", "standardisering", "värden", "intervall"]
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

export const featuredResources: LearningModule[] = [
  ...visualizationModules,
  plannedLabModules[0]
];
