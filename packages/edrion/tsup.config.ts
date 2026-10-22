import { defineConfig, type Options } from 'tsup';

const config: Options = {
    entry: ['src/index.ts'],
    format: ['esm', 'cjs'],
    dts: true,
    sourcemap: true,
    clean: true,
    splitting: false,
    treeshake: true,
    target: 'es2023',
    outDir: 'dist',
    external: ['@edrion/core', '@edrion/nestjs'],
};

export default defineConfig(config);
