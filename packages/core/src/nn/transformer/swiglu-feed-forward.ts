import { SiLU } from '@core/nn/activations/silu';
import { Linear } from '@core/nn/layers/linear';
import { Module } from '@core/nn/module';
import { TransformerConfigurationError } from '@core/nn/transformer/transformer.errors';
import { Tensor } from '@core/tensor/tensor';

export interface SwiGLUFeedForwardOptions {
    readonly bias?: boolean;
    readonly backend?: string;
}

/** Modern gated feed-forward block: down(SiLU(gate(x)) * up(x)). */
export class SwiGLUFeedForward extends Module {
    public readonly gateProjection: Linear;
    public readonly upProjection: Linear;
    public readonly downProjection: Linear;
    public readonly activation: SiLU;

    public constructor(embeddingDimension: number, hiddenDimension: number, options: SwiGLUFeedForwardOptions = {}) {
        super();
        if (!Number.isInteger(embeddingDimension) || embeddingDimension <= 0 || !Number.isInteger(hiddenDimension) || hiddenDimension <= 0) {
            throw new TransformerConfigurationError('SwiGLU dimensions must be positive integers.');
        }
        const linearOptions = { bias: options.bias ?? false, ...(options.backend === undefined ? {} : { backend: options.backend }) };
        this.gateProjection = this.registerModule('gateProjection', new Linear(embeddingDimension, hiddenDimension, linearOptions));
        this.upProjection = this.registerModule('upProjection', new Linear(embeddingDimension, hiddenDimension, linearOptions));
        this.activation = this.registerModule('activation', new SiLU());
        this.downProjection = this.registerModule('downProjection', new Linear(hiddenDimension, embeddingDimension, linearOptions));
    }

    public override forward(input: Tensor): Tensor {
        const gate: Tensor = this.activation.forward(this.gateProjection.forward(input));
        const up: Tensor = this.upProjection.forward(input);
        return this.downProjection.forward(gate.mul(up));
    }
}
