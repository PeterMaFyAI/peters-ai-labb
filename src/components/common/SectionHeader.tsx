interface SectionHeaderProps {
  eyebrow?: string;
  title: string;
  text?: string;
}

function SectionHeader({ eyebrow, title, text }: SectionHeaderProps): JSX.Element {
  return (
    <header className="section-header">
      {eyebrow ? <p className="section-eyebrow">{eyebrow}</p> : null}
      <h2>{title}</h2>
      {text ? <p>{text}</p> : null}
    </header>
  );
}

export default SectionHeader;

