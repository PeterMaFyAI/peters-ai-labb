import ResourceCard from "../components/common/ResourceCard";
import SectionHeader from "../components/common/SectionHeader";
import { labModules } from "../data/modules";

function LabsPage(): JSX.Element {
  return (
    <div className="container page-flow">
      <section className="section">
        <SectionHeader
          eyebrow="Interaktiva laborationer"
          title="Experimentera, träna och upptäck"
          text="Undersök hur AI lär sig genom egna experiment. Spela, ändra inställningar och följ vad som händer steg för steg."
        />
        <div className="resource-grid">
          {labModules.map((module) => (
            <ResourceCard key={module.id} module={module} />
          ))}
        </div>
      </section>
    </div>
  );
}

export default LabsPage;

