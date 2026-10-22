import { Inject, Injectable, type OnModuleDestroy, type OnModuleInit } from '@nestjs/common';
import { Edrion, type EdrionRuntimeInfo, type EdrionRuntimeMetricsSnapshot, type EdrionRuntimeStatus } from '@core';
import { EDRION_INSTANCE } from '@nest/constants/edrion.constants';

/**
 * NestJS facade for the Edrion runtime.
 */
@Injectable()
export class EdrionService implements OnModuleInit, OnModuleDestroy {
    constructor(
        @Inject(EDRION_INSTANCE)
        private readonly runtime: Edrion,
    ) {}

    /**
     * Initializes Edrion when NestJS starts this module.
     */
    async onModuleInit(): Promise<void> {
        await this.runtime.initialize();
    }

    /**
     * Gracefully releases Edrion resources
     * when NestJS destroys this module.
     */
    async onModuleDestroy(): Promise<void> {
        await this.runtime.shutdown();
    }

    /**
     * Returns the framework-independent Edrion runtime.
     */
    getRuntime(): Edrion {
        return this.runtime;
    }

    /**
     * Returns Edrion runtime information.
     */
    getRuntimeInfo(): EdrionRuntimeInfo {
        return this.runtime.getRuntimeInfo();
    }

    /**
     * Returns current runtime status.
     */
    getStatus(): EdrionRuntimeStatus {
        return this.runtime.getStatus();
    }

    /**
     * Indicates whether Edrion is ready.
     */
    isReady(): boolean {
        return this.runtime.isReady();
    }

    /**
     * Returns runtime metrics.
     */
    getMetrics(): EdrionRuntimeMetricsSnapshot {
        return this.runtime.getMetrics();
    }
}
