/** Base error for large-model infrastructure. */
export class LargeModelError extends Error {
    public constructor(message: string) {
        super(message);
        this.name = new.target.name;
    }
}

/** Raised when sharded checkpoint data is invalid. */
export class ShardedCheckpointError extends LargeModelError {}

/** Raised when memory planning configuration is invalid. */
export class MemoryPlanningError extends LargeModelError {}
