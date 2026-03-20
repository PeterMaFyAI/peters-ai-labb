import { Navigate, Route, Routes } from "react-router-dom";
import MainLayout from "./components/layout/MainLayout";
import AboutPage from "./pages/AboutPage";
import HomePage from "./pages/HomePage";
import LabsPage from "./pages/LabsPage";
import NotFoundPage from "./pages/NotFoundPage";
import VisualizationsPage from "./pages/VisualizationsPage";
import KNearestNeighborsPage from "./modules/k-nearest-neighbors/KNearestNeighborsPage";
import LinearRegressionPage from "./modules/linear-regression/LinearRegressionPage";
import LinearRegressionGradientPage from "./modules/linear-regression/LinearRegressionGradientPage";
import NeuralNetworkPage from "./modules/neural-network/NeuralNetworkPage";
import KMeansPage from "./modules/k-means/KMeansPage";
import AvstandPage from "./modules/avstand/AvstandPage";
import NormaliseringPage from "./modules/normalisering/NormaliseringPage";

function App(): JSX.Element {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<HomePage />} />
        <Route path="visualiseringar" element={<VisualizationsPage />} />
        <Route
          path="visualiseringar/k-narmaste-grannar"
          element={<KNearestNeighborsPage />}
        />
        <Route
          path="visualiseringar/linjar-regression"
          element={<LinearRegressionPage />}
        />
        <Route
          path="visualiseringar/linjar-regression/gradientnedstigning"
          element={<LinearRegressionGradientPage />}
        />
        <Route
          path="visualiseringar/neuralt-natverk"
          element={<NeuralNetworkPage />}
        />
        <Route path="visualiseringar/k-means" element={<KMeansPage />} />
        <Route path="visualiseringar/avstand" element={<AvstandPage />} />
        <Route
          path="visualiseringar/normalisering"
          element={<NormaliseringPage />}
        />
        <Route path="laborationer" element={<LabsPage />} />
        <Route path="om-portalen" element={<AboutPage />} />
        <Route path="hem" element={<Navigate to="/" replace />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}

export default App;
