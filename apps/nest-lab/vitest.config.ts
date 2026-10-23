import { defineConfig, type UserConfig } from 'vitest/config';

const config: UserConfig = {
    test: {
        environment: 'node',
        include: ['test/**/*.spec.ts'],
    },
};

export default defineConfig(config);
