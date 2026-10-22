import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

const sharedRules = {
    '@typescript-eslint/no-explicit-any': 'error',
    '@typescript-eslint/explicit-function-return-type': 'error',
    '@typescript-eslint/explicit-module-boundary-types': 'error',
    '@typescript-eslint/no-unnecessary-type-assertion': 'error',
    '@typescript-eslint/no-floating-promises': 'error',
    '@typescript-eslint/no-misused-promises': 'error',
    '@typescript-eslint/consistent-type-imports': [
        'error',
        {
            prefer: 'type-imports',
            fixStyle: 'inline-type-imports',
        },
    ],
};

export default tseslint.config(
    {
        ignores: ['**/dist/**', '**/node_modules/**', '**/coverage/**'],
    },
    eslint.configs.recommended,
    ...tseslint.configs.recommendedTypeChecked,
    {
        files: ['packages/core/src/**/*.ts'],
        languageOptions: {
            parserOptions: {
                project: './packages/core/tsconfig.json',
                tsconfigRootDir: import.meta.dirname,
            },
        },
        rules: sharedRules,
    },
    {
        files: ['packages/core/test/**/*.ts'],
        languageOptions: {
            parserOptions: {
                project: './packages/core/tsconfig.eslint.json',
                tsconfigRootDir: import.meta.dirname,
            },
        },
        rules: sharedRules,
    },
    {
        files: ['packages/nestjs/src/**/*.ts'],
        languageOptions: {
            parserOptions: {
                project: './packages/nestjs/tsconfig.json',
                tsconfigRootDir: import.meta.dirname,
            },
        },
        rules: sharedRules,
    },
    {
        files: ['packages/nestjs/test/**/*.ts'],
        languageOptions: {
            parserOptions: {
                project: './packages/nestjs/tsconfig.eslint.json',
                tsconfigRootDir: import.meta.dirname,
            },
        },
        rules: sharedRules,
    },
    {
        files: ['packages/edrion/src/**/*.ts'],
        languageOptions: {
            parserOptions: {
                project: './packages/edrion/tsconfig.json',
                tsconfigRootDir: import.meta.dirname,
            },
        },
        rules: sharedRules,
    },
);
