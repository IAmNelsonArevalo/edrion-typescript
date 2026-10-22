import { GenerationError } from '@core/generation/generation.errors';

export function assertLogits(logits: readonly number[]): void {
    if (logits.length === 0) {
        throw new GenerationError('Sampler requires at least one logit.');
    }
    for (let index: number = 0; index < logits.length; index += 1) {
        const value: number | undefined = logits[index];
        if (value === undefined || !Number.isFinite(value)) {
            throw new GenerationError(`Logit at index ${index} must be finite.`);
        }
    }
}

export function softmaxProbabilities(logits: readonly number[], temperature: number): number[] {
    assertLogits(logits);
    if (!Number.isFinite(temperature) || temperature <= 0) {
        throw new GenerationError(`Temperature must be finite and greater than zero. Received ${temperature}.`);
    }
    let maximum: number = Number.NEGATIVE_INFINITY;
    for (let index: number = 0; index < logits.length; index += 1) {
        const value: number | undefined = logits[index];
        if (value === undefined) {
            throw new GenerationError(`Logit at index ${index} does not exist.`);
        }
        maximum = Math.max(maximum, value / temperature);
    }
    const probabilities: number[] = new Array<number>(logits.length);
    let total: number = 0;
    for (let index: number = 0; index < logits.length; index += 1) {
        const value: number | undefined = logits[index];
        if (value === undefined) {
            throw new GenerationError(`Logit at index ${index} does not exist.`);
        }
        const exponential: number = Math.exp(value / temperature - maximum);
        probabilities[index] = exponential;
        total += exponential;
    }
    if (!Number.isFinite(total) || total <= 0) {
        throw new GenerationError('Unable to normalize sampling probabilities.');
    }
    for (let index: number = 0; index < probabilities.length; index += 1) {
        const value: number | undefined = probabilities[index];
        if (value === undefined) {
            throw new GenerationError(`Probability at index ${index} does not exist.`);
        }
        probabilities[index] = value / total;
    }
    return probabilities;
}

export function sampleFromProbabilities(probabilities: readonly number[], randomValue: number): number {
    if (!Number.isFinite(randomValue) || randomValue < 0 || randomValue >= 1) {
        throw new GenerationError(`Random source must produce values in [0, 1). Received ${randomValue}.`);
    }
    let cumulative: number = 0;
    for (let index: number = 0; index < probabilities.length; index += 1) {
        const probability: number | undefined = probabilities[index];
        if (probability === undefined) {
            throw new GenerationError(`Probability at index ${index} does not exist.`);
        }
        cumulative += probability;
        if (randomValue < cumulative) {
            return index;
        }
    }
    return probabilities.length - 1;
}
