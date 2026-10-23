import { TextGenerator, type GenerationResult } from '@core/generation';
import type { GenerationConfig } from '@core/generation/generation-config';
import { CrossEntropyLoss } from '@core/losses';
import { Module } from '@core/nn/module';
import { TransformerModel } from '@core/nn/transformer/transformer-model';
import type { LanguageModelConfig } from '@core/language-model/language-model-config';
import type { Tokenizer } from '@core/tokenizer';
import type { Tensor } from '@core/tensor';

/** GPT-style causal decoder language model built entirely on Edrion primitives. */
export class CausalLanguageModel extends Module {
    public readonly transformer: TransformerModel;
    private readonly crossEntropy: CrossEntropyLoss = new CrossEntropyLoss();

    public constructor(config: LanguageModelConfig) {
        super();
        this.transformer = this.registerModule(
            'transformer',
            new TransformerModel({ ...config, architecture: 'modern' }),
        );
    }

    public override forward(tokens: Tensor): Tensor {
        return this.transformer.forward(tokens);
    }

    /** Computes next-token cross entropy for pre-shifted targets. */
    public loss(inputTokens: Tensor, targetTokens: Tensor): Tensor {
        return this.crossEntropy.forward(this.forward(inputTokens), targetTokens);
    }

    public generate(tokenizer: Tokenizer, prompt: string, config: GenerationConfig = {}): GenerationResult {
        return new TextGenerator(this.transformer, tokenizer).generate(prompt, config);
    }
}
