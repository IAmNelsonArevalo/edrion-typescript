import type { Sampler } from '@core/generation/sampler';
import { assertLogits } from '@core/generation/sampler.utils';
import { GenerationError } from '@core/generation/generation.errors';

export class GreedySampler implements Sampler {
    public sample(logits: readonly number[]): number {
        assertLogits(logits);
        let bestIndex: number = 0;
        let bestValue: number | undefined = logits[0];
        if (bestValue === undefined) {
            throw new GenerationError('Unable to resolve first logit.');
        }
        for (let index: number = 1; index < logits.length; index += 1) {
            const value: number | undefined = logits[index];
            if (value === undefined) {
                throw new GenerationError(`Logit at index ${index} does not exist.`);
            }
            if (value > bestValue) {
                bestValue = value;
                bestIndex = index;
            }
        }
        return bestIndex;
    }
}
