/** Deterministic pseudo-random source used by the data pipeline. */
export interface SeededRandom {
    next(): number;
}

/**
 * Creates a compact deterministic PRNG suitable for reproducible shuffling.
 * This is not intended for cryptographic use.
 */
export function createSeededRandom(seed: number): SeededRandom {
    let state: number = normalizeSeed(seed);

    return {
        next(): number {
            state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
            return state / 4294967296;
        },
    };
}

function normalizeSeed(seed: number): number {
    if (!Number.isInteger(seed) || !Number.isFinite(seed)) {
        return 1;
    }

    return seed >>> 0;
}
