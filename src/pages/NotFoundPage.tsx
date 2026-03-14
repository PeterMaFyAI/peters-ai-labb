import { Link } from "react-router-dom";

function NotFoundPage(): JSX.Element {
  return (
    <div className="container page-flow">
      <section className="empty-state">
        <h1>Sidan hittades inte</h1>
        <p>Kontrollera adressen eller gå tillbaka till startsidan.</p>
        <Link className="btn btn-primary" to="/">
          Till startsidan
        </Link>
      </section>
    </div>
  );
}

export default NotFoundPage;

