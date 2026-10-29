import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import Landing from "./landing/Landing";
import "./styles.css";

// Routes: the landing page by default, the simulation at #sim.
function Root() {
  const [route, setRoute] = useState(() => (location.hash === "#sim" ? "sim" : "landing"));
  useEffect(() => {
    const on = () => setRoute(location.hash === "#sim" ? "sim" : "landing");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  // one bundle (no lazy chunk) so the single-file build keeps working; the simulation worker starts only here
  if (route === "sim") return <App />;
  return <Landing onEnter={() => (location.hash = "sim")} />;
}

createRoot(document.getElementById("root")!).render(<Root />);
