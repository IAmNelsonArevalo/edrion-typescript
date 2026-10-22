import { describe, expect, it } from 'vitest';
import { Edrion, type NormalizedEdrionOptions } from '@core';

describe('Edrion', (): void => {
    it('Should create an Edrion runtime', (): void => {
        const edrion: Edrion = new Edrion();

        expect(edrion.isInitialized()).toBe(false);
    });

    it('Should initialize the runtime', async (): Promise<void> => {
        const edrion: Edrion = new Edrion();

        await edrion.initialize();

        expect(edrion.isInitialized()).toBe(true);
    });

    it('Should shutdown the runtime', async (): Promise<void> => {
        const edrion: Edrion = new Edrion();

        await edrion.initialize();
        await edrion.shutdown();

        expect(edrion.isInitialized()).toBe(false);
    });

    it('Should normalize configuration', (): void => {
        const edrion: Edrion = new Edrion({
            observability: {
                enabled: true,
            },
            runtime: {
                metadata: {
                    application: 'test',
                },
            },
        });

        const options: NormalizedEdrionOptions = edrion.getOptions();

        expect(options.observability.enabled).toBe(true);
        expect(options.runtime.metadata['application']).toBe('test');
    });
});
