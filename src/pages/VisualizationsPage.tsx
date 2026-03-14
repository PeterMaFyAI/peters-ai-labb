import { useMemo, useState } from "react";
import ResourceCard from "../components/common/ResourceCard";
import SectionHeader from "../components/common/SectionHeader";
import { visualizationModules } from "../data/modules";

function VisualizationsPage(): JSX.Element {
  const [searchTerm, setSearchTerm] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("Alla kategorier");

  const categories = useMemo(
    () => ["Alla kategorier", ...new Set(visualizationModules.map((item) => item.category))],
    []
  );

  const filteredModules = useMemo(() => {
    return visualizationModules.filter((module) => {
      const matchesSearch =
        module.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        module.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        module.keywords.some((keyword) => keyword.toLowerCase().includes(searchTerm.toLowerCase()));

      const matchesCategory =
        categoryFilter === "Alla kategorier" || module.category === categoryFilter;

      return matchesSearch && matchesCategory;
    });
  }, [searchTerm, categoryFilter]);

  return (
    <div className="container page-flow">
      <section className="section">
        <SectionHeader
          eyebrow="Visualiseringar"
          title="Utforska AI-begrepp visuellt"
          text="Här hittar du alla visualiseringar som är förberedda i portalen. Varje modul har egen route och egen komponent, redo för implementation."
        />

        <div className="filter-row">
          <label className="search-field">
            <span>Sök visualisering</span>
            <input
              type="search"
              placeholder="Exempel: regression, noder..."
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
            />
          </label>

          <label className="search-field">
            <span>Filtrera kategori</span>
            <select
              value={categoryFilter}
              onChange={(event) => setCategoryFilter(event.target.value)}
            >
              {categories.map((category) => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>
          </label>
        </div>

        {filteredModules.length > 0 ? (
          <div className="resource-grid">
            {filteredModules.map((module) => (
              <ResourceCard key={module.id} module={module} />
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <h3>Ingen träff just nu</h3>
            <p>Prova en annan sökning eller återställ filtret.</p>
          </div>
        )}
      </section>
    </div>
  );
}

export default VisualizationsPage;

