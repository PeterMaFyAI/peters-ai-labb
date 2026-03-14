import { Link } from "react-router-dom";
import ResourceCard from "../components/common/ResourceCard";
import SectionHeader from "../components/common/SectionHeader";
import { featuredResources } from "../data/modules";

function HomePage(): JSX.Element {
  return (
    <div className="container page-flow">
      <section className="hero">
        <div className="hero-badge">AI, vetenskap och lärande</div>
        <h1>Peters AI-labb</h1>
        <p className="hero-text">
          En pedagogisk portal där visualiseringar och framtida laborationer gör AI begripligt
          steg för steg.
        </p>
        <div className="hero-actions">
          <Link className="btn btn-primary" to="/visualiseringar">
            Utforska visualiseringar
          </Link>
          <Link className="btn btn-ghost" to="/laborationer">
            Se laborationer
          </Link>
        </div>
      </section>

      <section className="section">
        <SectionHeader
          eyebrow="Utvalda resurser"
          title="Startklara moduler i portalen"
          text="Portalen är byggd för att växa. Nya moduler kan läggas till via datalistan utan att routing eller layout behöver skrivas om."
        />
        <div className="resource-grid">
          {featuredResources.map((module) => (
            <ResourceCard key={module.id} module={module} />
          ))}
        </div>
      </section>

      <section className="section split-section">
        <article className="info-card">
          <h2>Syftet med portalen</h2>
          <p>
            Portalen samlar AI-material på ett sätt som är tydligt för klassrummet. Fokus är
            tydlig begreppsförståelse, progression och snabb åtkomst till rätt resurs.
          </p>
        </article>
        <article className="info-card">
          <h2>Interaktivt och pedagogiskt</h2>
          <p>
            Sidorna är strukturerade för interaktivt innehåll med fokus på visualisering,
            laboration och reflektion. Det gör det enkelt att bygga vidare med nya övningar.
          </p>
        </article>
      </section>
    </div>
  );
}

export default HomePage;

