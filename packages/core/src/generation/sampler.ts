export type RandomSource = () => number;

export interface Sampler {
    sample(logits: readonly number[]): number;
}

export function defaultRandomSource(): number {
    return Math.random();
}
