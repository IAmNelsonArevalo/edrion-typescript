import { describe, expect, it } from 'vitest';
import {
    EDRION_VERSION,
    Edrion,
    getEdrionCapabilities,
    getEdrionReleaseInfo,
    type EdrionCapabilities,
    type EdrionHealthReport,
    type EdrionReleaseInfo,
} from '@core';

describe('Production Foundation', (): void => {
    it('should expose stable release information', (): void => {
        const release: EdrionReleaseInfo = getEdrionReleaseInfo();

        expect(EDRION_VERSION).toBe('1.2.0');
        expect(release.version).toBe('1.2.0');
        expect(release.channel).toBe('stable');
        expect(release.apiStability).toBe('stable');
    });

    it('should expose the stable capability set', (): void => {
        const capabilities: EdrionCapabilities = getEdrionCapabilities();

        expect(capabilities.capabilities).toContain('tensor');
        expect(capabilities.capabilities).toContain('autograd');
        expect(capabilities.capabilities).toContain('transformer');
        expect(capabilities.capabilities).toContain('training');
        expect(capabilities.capabilities).toContain('numerical-safety');
    });

    it('should report not-ready before initialization', (): void => {
        const edrion: Edrion = new Edrion();
        const health: EdrionHealthReport = edrion.getHealthReport();

        expect(health.health).toBe('not-ready');
        expect(health.ready).toBe(false);
    });

    it('should report ready after initialization', async (): Promise<void> => {
        const edrion: Edrion = new Edrion();

        await edrion.initialize();

        const health: EdrionHealthReport = edrion.getHealthReport();

        expect(health.health).toBe('ready');
        expect(health.ready).toBe(true);
        expect(health.backend?.name).toBe('cpu');

        await edrion.shutdown();
    });
});
