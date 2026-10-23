import { describe, expect, it } from 'vitest';
import { Linear } from '@core/nn';
import { SGD } from '@core/optim';
import { LossScaler, PrecisionPolicy } from '@core/precision';
import { Tensor, quantizeBFloat16, quantizeFloat16 } from '@core/tensor';

describe('mixed precision', () => {
    it('supports float16 and bfloat16 logical dtypes', () => {
        const fp16: Tensor = new Tensor([1.0001, 2.0001], [2], { dtype: 'float16' });
        const bf16: Tensor = new Tensor([1.0001, 2.0001], [2], { dtype: 'bfloat16' });
        expect(fp16.dtype).toBe('float16');
        expect(bf16.dtype).toBe('bfloat16');
        expect(fp16.get(0)).toBe(quantizeFloat16(1.0001));
        expect(bf16.get(0)).toBe(quantizeBFloat16(1.0001));
    });

    it('casts while preserving autograd', () => {
        const input: Tensor = new Tensor([1, 2], [2], { requiresGrad: true });
        const loss: Tensor = input.cast('float16').cast('float32').sum();
        loss.backward();
        expect(input.grad?.toArray()).toEqual([1, 1]);
    });

    it('emulates reduced activation precision without breaking float32 models', () => {
        const policy: PrecisionPolicy = new PrecisionPolicy({ computeDType: 'float16' });
        const input: Tensor = new Tensor([1.0001, 2.0001], [2]);
        const output: Tensor = policy.castInput(input);
        expect(output.dtype).toBe('float32');
        expect(output.get(0)).toBe(quantizeFloat16(1.0001));
    });

    it('scales and unscales gradients', () => {
        const model: Linear = new Linear(1, 1);
        const optimizer: SGD = new SGD(model.parameters(), { learningRate: 0.1 });
        const scaler: LossScaler = new LossScaler({ initialScale: 128, growthInterval: 2 });
        const input: Tensor = new Tensor([2], [1]);
        const loss: Tensor = scaler.scaleLoss(model.forward(input).sum());
        loss.backward();
        const foundNonFinite: boolean = scaler.unscale(optimizer.parameters);
        expect(foundNonFinite).toBe(false);
        expect(scaler.scale).toBe(128);
    });
});
