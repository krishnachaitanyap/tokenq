import puppeteer from 'puppeteer-core';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const URL = 'http://127.0.0.1:8090/';
const OUT = '/Users/roopapalivela/Documents/AI-casestudies/tokenq/screenshots';

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  defaultViewport: { width: 1280, height: 1800, deviceScaleFactor: 2 },
});

const page = await browser.newPage();
page.on('console', m => console.log('[page]', m.text()));

await page.goto(URL, { waitUntil: 'networkidle0', timeout: 30000 });
console.log('loaded');

// click the Last 7d preset
await page.evaluate(() => {
  const btn = document.querySelector('button.preset[data-preset="7d"]');
  if (btn) btn.click();
});
console.log('clicked 7d');

// wait for the data to populate
await new Promise(r => setTimeout(r, 4000));

// helper: switch tab and screenshot
const tabs = ['overview', 'spend', 'activity', 'pipeline'];

for (const tab of tabs) {
  await page.evaluate((t) => {
    const btn = document.querySelector(`button.tab[data-tab="${t}"]`);
    if (btn) btn.click();
  }, tab);
  await new Promise(r => setTimeout(r, 1500));
  const path = `${OUT}/dash-7d-${tab}.png`;
  await page.screenshot({ path, fullPage: true });
  console.log('saved', path);
}

await browser.close();
console.log('done');
