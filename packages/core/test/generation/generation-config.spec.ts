import { describe, expect, it } from 'vitest';
import { GenerationError, normalizeGenerationConfig } from '@core';

describe('Generation config', (): void => {
    it('should normalize defaults', (): void => {
        const config = normalizeGenerationConfig();
        expect(config.maxNewTokens).toBe(32);
        expect(config.temperature).toBe(1);
    });

    it('should reject invalid maxNewTokens', (): void => {
        expect((): void => {
            normalizeGenerationConfig({ maxNewTokens: -1 });
        }).toThrow(GenerationError);
    });
});
