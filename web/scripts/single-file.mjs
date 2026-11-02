// Packs the Vite build into one self-contained HTML body (CSS + JS inline, worker inlined as a blob) for
// publishing as a claude.ai artifact: dist/ecoarena.html. Run after `vite build`.
import { readFileSync, readdirSync, writeFileSync } from "node:fs";

const dir = new URL("../dist/", import.meta.url);
const assets = readdirSync(new URL("assets/", dir));
const css = assets.filter((f) => f.endsWith(".css")).map((f) => readFileSync(new URL(`assets/${f}`, dir), "utf8")).join("\n");
const js = assets.filter((f) => f.endsWith(".js")).map((f) => readFileSync(new URL(`assets/${f}`, dir), "utf8")).join("\n");
const safe = (s) => s.replace(/<\/(script|style)/gi, "<\\/$1");
const html = `<meta charset="utf-8">
<title>EcoArena</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
${safe(css)}
</style>
<div id="root"></div>
<script type="module">
${safe(js)}
</script>
`;
writeFileSync(new URL("ecoarena.html", dir), html);
console.log(`dist/ecoarena.html ${(html.length / 1e6).toFixed(2)} MB`);
