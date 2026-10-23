import type { Module } from '@core/nn/module';
import {
    captureModelState,
    restoreModelState,
    type ModelState,
    type SerializedParameter,
} from '@core/serialization/model-state';
import { ShardedCheckpointError } from '@core/large-model/large-model.errors';

export const SHARDED_MODEL_FORMAT_VERSION: number = 1;

export interface ModelShard {
    readonly index: number;
    readonly approximateBytes: number;
    readonly parameters: readonly SerializedParameter[];
}

export interface ShardedModelManifest {
    readonly formatVersion: number;
    readonly parameterCount: number;
    readonly shardCount: number;
    readonly approximateBytes: number;
}

export interface ShardedModelState {
    readonly manifest: ShardedModelManifest;
    readonly shards: readonly ModelShard[];
}

/** Splits a model state into deterministic parameter shards. */
export function createShardedModelState(model: Module, maxShardBytes: number): ShardedModelState {
    if (!Number.isInteger(maxShardBytes) || maxShardBytes <= 0) {
        throw new ShardedCheckpointError(`maxShardBytes must be a positive integer. Received ${maxShardBytes}.`);
    }
    const state: ModelState = captureModelState(model);
    const shards: ModelShard[] = [];
    let current: SerializedParameter[] = [];
    let currentBytes: number = 0;
    let totalBytes: number = 0;

    const flush = (): void => {
        if (current.length === 0) return;
        shards.push({ index: shards.length, approximateBytes: currentBytes, parameters: current });
        current = [];
        currentBytes = 0;
    };

    for (let index: number = 0; index < state.parameters.length; index += 1) {
        const parameter: SerializedParameter | undefined = state.parameters[index];
        if (parameter === undefined) {
            throw new ShardedCheckpointError(`Missing serialized parameter at index ${index}.`);
        }
        const bytes: number = parameter.tensor.data.length * 4;
        if (current.length > 0 && currentBytes + bytes > maxShardBytes) flush();
        current.push(parameter);
        currentBytes += bytes;
        totalBytes += bytes;
        if (currentBytes >= maxShardBytes) flush();
    }
    flush();

    return {
        manifest: {
            formatVersion: SHARDED_MODEL_FORMAT_VERSION,
            parameterCount: state.parameters.length,
            shardCount: shards.length,
            approximateBytes: totalBytes,
        },
        shards,
    };
}

/** Restores model parameters from a validated sharded state. */
export function restoreShardedModelState(model: Module, state: ShardedModelState): void {
    if (state.manifest.formatVersion !== SHARDED_MODEL_FORMAT_VERSION) {
        throw new ShardedCheckpointError(`Unsupported sharded model format ${state.manifest.formatVersion}.`);
    }
    if (state.shards.length !== state.manifest.shardCount) {
        throw new ShardedCheckpointError('Shard count does not match manifest.');
    }
    const parameters: SerializedParameter[] = [];
    const names: Set<string> = new Set<string>();
    for (let shardIndex: number = 0; shardIndex < state.shards.length; shardIndex += 1) {
        const shard: ModelShard | undefined = state.shards[shardIndex];
        if (shard === undefined || shard.index !== shardIndex) {
            throw new ShardedCheckpointError(`Invalid shard ordering at index ${shardIndex}.`);
        }
        for (let parameterIndex: number = 0; parameterIndex < shard.parameters.length; parameterIndex += 1) {
            const parameter: SerializedParameter | undefined = shard.parameters[parameterIndex];
            if (parameter === undefined) throw new ShardedCheckpointError('Missing parameter inside shard.');
            if (names.has(parameter.name)) throw new ShardedCheckpointError(`Duplicate parameter "${parameter.name}".`);
            names.add(parameter.name);
            parameters.push(parameter);
        }
    }
    if (parameters.length !== state.manifest.parameterCount) {
        throw new ShardedCheckpointError('Parameter count does not match manifest.');
    }
    restoreModelState(model, { parameters });
}
