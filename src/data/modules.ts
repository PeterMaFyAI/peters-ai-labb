import type { LearningModule } from "../types/module";
import knnImage from "../assets/module-images/knn.png";
import linearRegressionImage from "../assets/module-images/linear-regression.png";
import neuralNetworkImage from "../assets/module-images/neural-network.png";
import kMeansImage from "../assets/module-images/k-means.png";
import avstandImage from "../assets/module-images/avstand.png";
import normaliseringImage from "../assets/module-images/normalisering.png";
import qLearningImage from "../assets/module-images/q-learning.png";
import nimImage from "../assets/module-images/nim.svg";
import languageModelImage from "../assets/module-images/language-model.svg";

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
    description:
      "Skapa punktmoln och följ k-means steg för steg med tilldelning och flytt av medelpunkter.",
    category: "Klustring",
    targetGroup: "Gymnasiet / vuxenutbildning",
    route: "/visualiseringar/k-means",
    type: "visualization",
    status: "ready",
    keywords: ["klustring", "centroider", "gruppering", "avstånd", "kovarians", "k-means"]
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
  },
  {
    id: "q-learning",
    title: "Q-learning",
    slug: "q-learning",
    imageUrl: qLearningImage,
    description:
      "Räkna Q-värden för hand i fem enkla miljöer och följ Q-tabellen steg för steg.",
    category: "Förstärkningsinlärning",
    targetGroup: "Gymnasiet / vuxenutbildning",
    route: "/visualiseringar/q-learning",
    type: "visualization",
    status: "ready",
    keywords: ["q-table", "agent", "belöning", "epsilon-greedy", "episoder"]
  }
];

export const labModules: LearningModule[] = [
  {
    id: "language-model",
    title: "En token i taget",
    slug: "sprakmodell",
    imageUrl: languageModelImage,
    description:
      "Följ hur en språkmodell genererar ett svar token för token. Utforska sannolikheter, slump och temperatur.",
    category: "Språkmodeller",
    targetGroup: "Gymnasiet / vuxenutbildning",
    route: "/laborationer/sprakmodell",
    type: "lab",
    status: "ready",
    keywords: ["språkmodell", "LLM", "tokens", "sannolikhet", "temperatur", "generering"]
  },
  {
    id: "nim",
    title: "Nim – träna en AI",
    slug: "nim",
    imageUrl: nimImage,
    description:
      "Spela Nim med 12 pinnar, träna en AI med Q-learning och följ hur dess Q-tabell utvecklas.",
    category: "Förstärkningsinlärning",
    targetGroup: "Gymnasiet / vuxenutbildning",
    route: "/laborationer/nim",
    type: "lab",
    status: "ready",
    keywords: ["nim", "q-learning", "q-tabell", "strategi", "belöning", "träning"]
  }
];

export const featuredResources: LearningModule[] = [
  ...visualizationModules,
  ...labModules
];
