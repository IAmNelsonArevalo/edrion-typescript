import type { BasicTokenizerState } from '@core/tokenizer/basic-tokenizer';
import type { ModelState } from '@core/serialization/model-state';
import type { TrainingState } from '@core/training/training-state';

/** Current checkpoint document format version. */
export const CHECKPOINT_FORMAT_VERSION: number = 1;

/** JSON-safe Edrion training checkpoint. */
export interface Checkpoint {
    readonly formatVersion: number;
    readonly frameworkVersion: string;
    readonly createdAt: string;
    readonly model: ModelState;
    readonly training: TrainingState;
    readonly tokenizer?: BasicTokenizerState;
    readonly metadata?: Readonly<Record<string, string | number | boolean>>;
}

/** Input used to construct a checkpoint. */
export interface CreateCheckpointOptions {
    readonly frameworkVersion: string;
    readonly model: ModelState;
    readonly training: TrainingState;
    readonly tokenizer?: BasicTokenizerState;
    readonly metadata?: Readonly<Record<string, string | number | boolean>>;
}

/** Creates an immutable checkpoint document. */
export function createCheckpoint(options: CreateCheckpointOptions): Checkpoint {
    const checkpoint: Checkpoint = {
        formatVersion: CHECKPOINT_FORMAT_VERSION,
        frameworkVersion: options.frameworkVersion,
        createdAt: new Date().toISOString(),
        model: options.model,
        training: {
            ...options.training,
        },
        ...(options.tokenizer === undefined
            ? {}
            : {
                  tokenizer: options.tokenizer,
              }),
        ...(options.metadata === undefined
            ? {}
            : {
                  metadata: {
                      ...options.metadata,
                  },
              }),
    };

    return checkpoint;
}
