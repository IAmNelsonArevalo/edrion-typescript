import { GenerationError } from '@core/generation/generation.errors';
import { defaultRandomSource, type RandomSource, type Sampler } from '@core/generation/sampler';
import { sampleFromProbabilities, softmaxProbabilities } from '@core/generation/sampler.utils';

export class TemperatureSampler implements Sampler {
    private readonly temperature: number;
    private readonly randomSource: RandomSource;

    public constructor(temperature: number = 1, randomSource: RandomSource = defaultRandomSource) {
        if (!Number.isFinite(temperature) || temperature <= 0) {
            throw new GenerationError(`Temperature must be finite and greater than zero. Received ${temperature}.`);
        }
        this.temperature = temperature;
        this.randomSource = randomSource;
    }

    public sample(logits: readonly number[]): number {
        const probabilities: readonly number[] = softmaxProbabilities(logits, this.temperature);
        return sampleFromProbabilities(probabilities, this.randomSource());
    }
}
