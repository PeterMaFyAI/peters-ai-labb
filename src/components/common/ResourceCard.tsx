import { Link } from "react-router-dom";
import type { LearningModule } from "../../types/module";

interface ResourceCardProps {
  module: LearningModule;
}

function ResourceCard({ module }: ResourceCardProps): JSX.Element {
  const hasPreviewImage = Boolean(module.imageUrl);

  return (
    <article className="resource-card">
      <div className="resource-meta">
        <span className="pill">{module.type === "visualization" ? "Visualisering" : "Laboration"}</span>
        <span className="category">{module.category}</span>
      </div>
      <h3>{module.title}</h3>
      {hasPreviewImage ? (
        <div className="resource-preview">
          <img src={module.imageUrl} alt={`Förhandsbild för ${module.title}`} loading="lazy" />
        </div>
      ) : (
        <>
          <p>{module.description}</p>
          <p className="resource-target">{module.targetGroup}</p>
        </>
      )}
      <Link className="btn btn-secondary" to={module.route}>
        Öppna modul
      </Link>
    </article>
  );
}

export default ResourceCard;
