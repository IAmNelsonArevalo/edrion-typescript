/** One measured performance sample. */
export interface PerformanceSample {
    readonly name: string;
    readonly durationMs: number;
    readonly units?: number;
    readonly unitsPerSecond?: number;
}

/** Lightweight profiler usable in browser and Node-like runtimes. */
export class PerformanceProfiler {
    public measure<T>(
        name: string,
        operation: () => T,
        units?: number,
    ): { readonly result: T; readonly sample: PerformanceSample } {
        const start: number = performance.now();
        const result: T = operation();
        const durationMs: number = Math.max(0, performance.now() - start);
        return {
            result,
            sample: {
                name,
                durationMs,
                ...(units === undefined
                    ? {}
                    : {
                          units,
                          unitsPerSecond: durationMs === 0 ? Number.POSITIVE_INFINITY : (units * 1000) / durationMs,
                      }),
            },
        };
    }

    public async measureAsync<T>(
        name: string,
        operation: () => Promise<T>,
        units?: number,
    ): Promise<{ readonly result: T; readonly sample: PerformanceSample }> {
        const start: number = performance.now();
        const result: T = await operation();
        const durationMs: number = Math.max(0, performance.now() - start);
        return {
            result,
            sample: {
                name,
                durationMs,
                ...(units === undefined
                    ? {}
                    : {
                          units,
                          unitsPerSecond: durationMs === 0 ? Number.POSITIVE_INFINITY : (units * 1000) / durationMs,
                      }),
            },
        };
    }
}
