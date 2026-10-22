import { Inject, Injectable, type OnModuleDestroy, type OnModuleInit } from '@nestjs/common';
import {
    Edrion,
    type EdrionExecutionContext,
    type EdrionRuntimeInfo,
    type EdrionRuntimeMetricsSnapshot,
    type EdrionRuntimeStatus,
    type NormalizedEdrionOptions,
} from '@edrion/core';
import { EDRION_INSTANCE } from '@nest/constants/edrion.constants';

/** Official NestJS service exposing the Edrion runtime. */
@Injectable()
export class EdrionService implements OnModuleInit, OnModuleDestroy {
    public constructor(
        @Inject(EDRION_INSTANCE)
        private readonly edrion: Edrion,
    ) {}

    /** Initializes Edrion with the NestJS lifecycle. */
    public async onModuleInit(): Promise<void> {
        await this.edrion.initialize();
    }

    /** Shuts Edrion down with the NestJS lifecycle. */
    public async onModuleDestroy(): Promise<void> {
        await this.edrion.shutdown();
    }

    /** Returns the underlying runtime. */
    public getRuntime(): Edrion {
        return this.edrion;
    }

    /** Returns normalized runtime configuration. */
    public getOptions(): NormalizedEdrionOptions {
        return this.edrion.getOptions();
    }

    /** Returns the runtime execution context. */
    public getExecutionContext(): EdrionExecutionContext {
        return this.edrion.getExecutionContext();
    }

    /** Returns current lifecycle status. */
    public getStatus(): EdrionRuntimeStatus {
        return this.edrion.getStatus();
    }

    /** Returns runtime lifecycle metrics. */
    public getMetrics(): EdrionRuntimeMetricsSnapshot {
        return this.edrion.getMetrics();
    }

    /** Returns complete runtime diagnostics. */
    public getRuntimeInfo(): EdrionRuntimeInfo {
        return this.edrion.getRuntimeInfo();
    }

    /** Returns the current Edrion version. */
    public getVersion(): string {
        return this.edrion.getVersion();
    }

    /** Compatibility API retained from FEATURE-000. */
    public isInitialized(): boolean {
        return this.edrion.isInitialized();
    }

    /** Indicates whether Edrion is ready to execute work. */
    public isReady(): boolean {
        return this.edrion.isReady();
    }
}
