import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vitest/config';

export default defineConfig({
    resolve: {
        alias: {
            '@core': fileURLToPath(new URL('./src/core', import.meta.url)),
            '@nest': fileURLToPath(new URL('./src/nest', import.meta.url)),
            '@test': fileURLToPath(new URL('./test', import.meta.url)),
        },
    },
    test: {
        environment: 'node',
        include: ['test/**/*.spec.ts'],
        clearMocks: true,
        restoreMocks: true,
        mockReset: true,
    },
});
