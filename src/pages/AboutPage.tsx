function AboutPage(): JSX.Element {
  return (
    <div className="container page-flow">
      <section className="section">
        <p className="section-eyebrow">Om portalen</p>
        <h1>För lärare och elever</h1>
        <p className="lead">
          Peters AI-labb är byggd för att ge en tydlig, pedagogisk och skalbar struktur för
          undervisning i AI.
        </p>
      </section>

      <section className="split-section">
        <article className="info-card">
          <h2>Pedagogisk riktning</h2>
          <p>
            Innehållet riktar sig till gymnasiet och vuxenutbildning med fokus på begrepp,
            resonemang och praktisk förståelse.
          </p>
        </article>
        <article className="info-card">
          <h2>Byggd för tillväxt</h2>
          <p>
            Varje modul definieras i en gemensam datastruktur och kan få egen sida, route och
            interaktiv implementation utan att resten av appen ändras.
          </p>
        </article>
      </section>
    </div>
  );
}

export default AboutPage;

