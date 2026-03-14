import { Link } from "react-router-dom";

interface ModulePlaceholderProps {
  title: string;
  summary: string;
}

function ModulePlaceholder({ title, summary }: ModulePlaceholderProps): JSX.Element {
  return (
    <section className="module-placeholder">
      <p className="section-eyebrow">Modul förberedd</p>
      <h1>{title}</h1>
      <p>{summary}</p>
      <p className="placeholder-note">
        Den här sidan är strukturerad och klar för implementation av själva visualiseringen.
      </p>
      <div className="placeholder-actions">
        <Link className="btn btn-primary" to="/visualiseringar">
          Tillbaka till visualiseringar
        </Link>
        <Link className="btn btn-ghost" to="/">
          Till startsidan
        </Link>
      </div>
    </section>
  );
}

export default ModulePlaceholder;

