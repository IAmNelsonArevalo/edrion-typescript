import { GenerationError } from '@core/generation/generation.errors';
import { defaultRandomSource, type RandomSource, type Sampler } from '@core/generation/sampler';
import { assertLogits, sampleFromProbabilities, softmaxProbabilities } from '@core/generation/sampler.utils';

interface IndexedLogit {
    readonly index: number;
    readonly value: number;
}

export class TopKSampler implements Sampler {
    private readonly topK: number;
    private readonly temperature: number;
    private readonly randomSource: RandomSource;

    public constructor(topK: number, temperature: number = 1, randomSource: RandomSource = defaultRandomSource) {
        if (!Number.isInteger(topK) || topK <= 0) {
            throw new GenerationError(`topK must be a positive integer. Received ${topK}.`);
        }
        if (!Number.isFinite(temperature) || temperature <= 0) {
            throw new GenerationError(`Temperature must be finite and greater than zero. Received ${temperature}.`);
        }
        this.topK = topK;
        this.temperature = temperature;
        this.randomSource = randomSource;
    }

    public sample(logits: readonly number[]): number {
        assertLogits(logits);
        const indexed: IndexedLogit[] = logits.map((value: number, index: number): IndexedLogit => ({ index, value }));
        indexed.sort((left: IndexedLogit, right: IndexedLogit): number => right.value - left.value);
        const selected: IndexedLogit[] = indexed.slice(0, Math.min(this.topK, indexed.length));
        const selectedLogits: number[] = selected.map((entry: IndexedLogit): number => entry.value);
        const probabilities: readonly number[] = softmaxProbabilities(selectedLogits, this.temperature);
        const selectedIndex: number = sampleFromProbabilities(probabilities, this.randomSource());
        const entry: IndexedLogit | undefined = selected[selectedIndex];
        if (entry === undefined) {
            throw new GenerationError(`Top-k sample index ${selectedIndex} does not exist.`);
        }
        return entry.index;
    }
}
