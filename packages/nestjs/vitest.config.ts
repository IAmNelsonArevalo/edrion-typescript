import { fileURLToPath, URL } from 'node:url';
import { defineConfig, type UserConfig } from 'vitest/config';

const nestPath: string = fileURLToPath(new URL('./src', import.meta.url));

const config: UserConfig = {
    resolve: {
        alias: {
            '@nest': nestPath,
        },
    },
    test: {
        environment: 'node',
        include: ['test/**/*.spec.ts'],
        clearMocks: true,
        restoreMocks: true,
        mockReset: true,
    },
};

export default defineConfig(config);
