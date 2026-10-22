import { describe, expect, it } from 'vitest';

import { Linear, Module, Parameter, Tensor, type NamedParameter } from '@core';

class CompositeModule extends Module {
    public readonly linear: Linear;
    public readonly offset: Parameter;

    public constructor() {
        super();

        this.linear = this.registerModule('linear', new Linear(2, 2));

        this.offset = this.registerParameter('offset', new Parameter([0, 0], [2]));
    }

    public override forward(input: Tensor): Tensor {
        return this.linear.forward(input).add(this.offset);
    }
}

describe('Module', (): void => {
    it('should expose recursive parameters', (): void => {
        const module: CompositeModule = new CompositeModule();
        const parameters: readonly Parameter[] = module.parameters();

        expect(parameters).toHaveLength(3);
    });

    it('should expose named parameters', (): void => {
        const module: CompositeModule = new CompositeModule();
        const parameters: readonly NamedParameter[] = module.namedParameters();

        expect(parameters.map((entry: NamedParameter): string => entry.name)).toEqual([
            'offset',
            'linear.weight',
            'linear.bias',
        ]);
    });

    it('should propagate train and eval modes', (): void => {
        const module: CompositeModule = new CompositeModule();

        module.eval();

        expect(module.training).toBe(false);
        expect(module.linear.training).toBe(false);

        module.train();

        expect(module.training).toBe(true);
        expect(module.linear.training).toBe(true);
    });

    it('should clear parameter gradients recursively', (): void => {
        const module: CompositeModule = new CompositeModule();
        const input: Tensor = new Tensor([1, 2], [2], {
            requiresGrad: true,
        });

        module.forward(input).sum().backward();

        expect(module.offset.grad).toBeDefined();
        expect(module.linear.weight.grad).toBeDefined();

        module.zeroGrad();

        expect(module.offset.grad).toBeUndefined();
        expect(module.linear.weight.grad).toBeUndefined();
    });
});
