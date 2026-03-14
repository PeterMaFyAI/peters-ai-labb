import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/", label: "Startsida", end: true },
  { to: "/visualiseringar", label: "Visualiseringar" },
  { to: "/laborationer", label: "Interaktiva laborationer" },
  { to: "/om-portalen", label: "Om portalen" }
];

function Header(): JSX.Element {
  return (
    <header className="site-header">
      <div className="container topbar">
        <NavLink to="/" className="brand" end>
          <span className="brand-dot" aria-hidden="true" />
          <span>
            <strong>Peters AI-labb</strong>
            <small>Pedagogisk portal för AI</small>
          </span>
        </NavLink>

        <nav className="site-nav" aria-label="Huvudnavigation">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `nav-link${isActive ? " nav-link-active" : ""}`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}

export default Header;

