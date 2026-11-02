// Self-hosted fonts (latin subset only), inlined as data URLs at build time so the static site, the Docker image
// and the single-file build all carry them: no request to Google Fonts.
//   Fraunces   - display: a soft, organic "old style" serif with an optical-size axis
//   Inter      - body and interface text (optical sizes 14-32)
//   JetBrains Mono - numbers, steps, code
import frauncesUrl from "@fontsource-variable/fraunces/files/fraunces-latin-opsz-normal.woff2?url";
import interUrl from "@fontsource-variable/inter/files/inter-latin-opsz-normal.woff2?url";
import mono400 from "@fontsource/jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff2?url";
import mono500 from "@fontsource/jetbrains-mono/files/jetbrains-mono-latin-500-normal.woff2?url";

const LATIN = "U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2190-2199,U+2212,U+2215,U+FEFF,U+FFFD";
const face = (family: string, url: string, weight: string, variable: boolean) =>
  `@font-face{font-family:'${family}';font-style:normal;font-display:swap;font-weight:${weight};` +
  `src:url(${url}) format('${variable ? "woff2-variations" : "woff2"}');unicode-range:${LATIN}}`;

const css = [
  face("Fraunces", frauncesUrl, "100 900", true),
  face("Inter", interUrl, "100 900", true),
  face("JetBrains Mono", mono400, "400", false),
  face("JetBrains Mono", mono500, "500", false),
].join("\n");

const style = document.createElement("style");
style.dataset.fonts = "ecoarena";
style.textContent = css;
document.head.prepend(style);
