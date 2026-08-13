import "@fontsource-variable/jetbrains-mono";
import "@fontsource-variable/manrope";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <App />,
);
