import { fileURLToPath, URL } from 'node:url';
import { defineConfig, type UserConfig } from 'vitest/config';

const corePath: string = fileURLToPath(new URL('./src', import.meta.url));

const config: UserConfig = {
    resolve: {
        alias: {
            '@core': corePath,
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
