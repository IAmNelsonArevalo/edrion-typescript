import { defineConfig } from 'tsup';

export default defineConfig({
    entry: ['src/index.ts'],
    format: ['esm', 'cjs'],
    dts: true,
    clean: true,
    splitting: false,
    treeshake: true,
    sourcemap: true,
    minify: false,
    outDir: 'dist',
    tsconfig: 'tsconfig.build.json',
    external: ['@nestjs/common', '@nestjs/core', 'reflect-metadata', 'rxjs'],
});
