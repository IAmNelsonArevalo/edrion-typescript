import type { Module } from '@core/nn/module';
import type { Parameter } from '@core/nn/parameter';
import { ModelStateError } from '@core/serialization/serialization.errors';
import { TensorSerializer, type SerializedTensor } from '@core/serialization/tensor-serializer';
import { tensorShapesEqual } from '@core/tensor/shape';

/** One named serialized model parameter. */
export interface SerializedParameter {
    readonly name: string;
    readonly tensor: SerializedTensor;
}

/** JSON-safe model state. */
export interface ModelState {
    readonly parameters: readonly SerializedParameter[];
}

/** Captures all trainable parameters from a Module hierarchy. */
export function captureModelState(model: Module): ModelState {
    const namedParameters = model.namedParameters();
    const parameters: SerializedParameter[] = [];

    for (let index: number = 0; index < namedParameters.length; index += 1) {
        const entry = namedParameters[index];

        if (entry === undefined) {
            throw new ModelStateError(`Missing model parameter registration at index ${index}.`);
        }

        parameters.push({
            name: entry.name,
            tensor: TensorSerializer.serialize(entry.parameter),
        });
    }

    return {
        parameters,
    };
}

/** Restores serialized parameter values into an existing Module hierarchy. */
export function restoreModelState(model: Module, state: ModelState): void {
    const currentParameters = model.namedParameters();
    const currentByName: Map<string, Parameter> = new Map<string, Parameter>();

    for (let index: number = 0; index < currentParameters.length; index += 1) {
        const entry = currentParameters[index];

        if (entry === undefined) {
            throw new ModelStateError(`Missing current model parameter at index ${index}.`);
        }

        currentByName.set(entry.name, entry.parameter);
    }

    if (state.parameters.length !== currentByName.size) {
        throw new ModelStateError(
            `Serialized model contains ${state.parameters.length} parameters but current model expects ${currentByName.size}.`,
        );
    }

    for (let index: number = 0; index < state.parameters.length; index += 1) {
        const serialized = state.parameters[index];

        if (serialized === undefined) {
            throw new ModelStateError(`Missing serialized parameter at index ${index}.`);
        }

        const parameter: Parameter | undefined = currentByName.get(serialized.name);

        if (parameter === undefined) {
            throw new ModelStateError(`Serialized parameter "${serialized.name}" does not exist in the current model.`);
        }

        if (!tensorShapesEqual(parameter.shape, serialized.tensor.shape)) {
            throw new ModelStateError(
                `Serialized parameter "${serialized.name}" shape [${serialized.tensor.shape.join(', ')}] does not match current shape [${parameter.shape.join(', ')}].`,
            );
        }

        if (parameter.dtype !== serialized.tensor.dtype) {
            throw new ModelStateError(
                `Serialized parameter "${serialized.name}" dtype does not match the current parameter dtype.`,
            );
        }

        parameter.assign(serialized.tensor.data);
    }
}
