// Print docs/handbook/handbook.html to EcoArena-Handbook.pdf (A4, page numbers in the footer).
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + path.resolve(__dirname, 'handbook.html'), { waitUntil: 'networkidle' });
  await p.evaluate(() => document.fonts.ready);
  await p.pdf({
    path: path.resolve(__dirname, process.argv[2] || 'EcoArena-Handbook.pdf'),
    format: 'A4', printBackground: true, preferCSSPageSize: true, displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="width:100%;font:7.5pt Inter,sans-serif;color:#7a8279;padding:0 19mm;display:flex;justify-content:space-between"><span>EcoArena team handbook</span><span class="pageNumber"></span></div>',
  });
  await b.close();
})();
