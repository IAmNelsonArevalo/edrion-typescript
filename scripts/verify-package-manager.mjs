import { readFile } from 'node:fs/promises';

const packageDocument = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
const specification = packageDocument.packageManager;
const exactPnpmPattern = /^pnpm@\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/u;

if (typeof specification !== 'string' || !exactPnpmPattern.test(specification)) {
    throw new Error(
        `packageManager must pin an exact pnpm semantic version. Received ${JSON.stringify(specification)}.`,
    );
}

console.log(`Verified reproducible package manager: ${specification}`);
