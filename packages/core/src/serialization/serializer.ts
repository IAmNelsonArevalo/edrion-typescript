import { CHECKPOINT_FORMAT_VERSION, type Checkpoint } from '@core/serialization/checkpoint';
import { SerializationError } from '@core/serialization/serialization.errors';

/** JSON serializer for Edrion checkpoint documents. */
export class Serializer {
    public static stringify(checkpoint: Checkpoint, spacing: number = 2): string {
        return JSON.stringify(checkpoint, null, spacing);
    }

    public static parse(serialized: string): Checkpoint {
        let value: unknown;

        try {
            value = JSON.parse(serialized) as unknown;
        } catch (error: unknown) {
            throw new SerializationError(
                error instanceof Error
                    ? `Unable to parse checkpoint JSON: ${error.message}`
                    : 'Unable to parse checkpoint JSON.',
            );
        }

        assertCheckpoint(value);
        return value;
    }
}

function assertCheckpoint(value: unknown): asserts value is Checkpoint {
    if (typeof value !== 'object' || value === null) {
        throw new SerializationError('Checkpoint must be a JSON object.');
    }

    const record: Record<string, unknown> = value as Record<string, unknown>;
    const formatVersion: unknown = record['formatVersion'];
    const frameworkVersion: unknown = record['frameworkVersion'];
    const createdAt: unknown = record['createdAt'];
    const model: unknown = record['model'];
    const training: unknown = record['training'];

    if (formatVersion !== CHECKPOINT_FORMAT_VERSION) {
        throw new SerializationError(
            `Unsupported checkpoint format version ${String(formatVersion)}. Expected ${CHECKPOINT_FORMAT_VERSION}.`,
        );
    }

    if (typeof frameworkVersion !== 'string' || frameworkVersion.length === 0) {
        throw new SerializationError('Checkpoint frameworkVersion must be a non-empty string.');
    }

    if (typeof createdAt !== 'string' || Number.isNaN(Date.parse(createdAt))) {
        throw new SerializationError('Checkpoint createdAt must be a valid ISO date string.');
    }

    if (typeof model !== 'object' || model === null) {
        throw new SerializationError('Checkpoint model state is missing.');
    }

    if (typeof training !== 'object' || training === null) {
        throw new SerializationError('Checkpoint training state is missing.');
    }
}
