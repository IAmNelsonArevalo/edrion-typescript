import type { Parameter } from '@core/nn/parameter';
import { Tensor } from '@core/tensor/tensor';

export interface LossScalerOptions {
    readonly initialScale?: number;
    readonly growthFactor?: number;
    readonly backoffFactor?: number;
    readonly growthInterval?: number;
}

export interface LossScaleUpdate {
    readonly foundNonFinite: boolean;
    readonly scale: number;
}

/** Dynamic loss scaler for mixed-precision training. */
export class LossScaler {
    private scaleInternal: number;
    private readonly growthFactor: number;
    private readonly backoffFactor: number;
    private readonly growthInterval: number;
    private finiteSteps: number = 0;

    public constructor(options: LossScalerOptions = {}) {
        this.scaleInternal = validatePositive(options.initialScale ?? 65536, 'initialScale');
        this.growthFactor = validateGreaterThanOne(options.growthFactor ?? 2, 'growthFactor');
        this.backoffFactor = validateFraction(options.backoffFactor ?? 0.5, 'backoffFactor');
        this.growthInterval = validatePositiveInteger(options.growthInterval ?? 2000, 'growthInterval');
    }

    public get scale(): number {
        return this.scaleInternal;
    }

    public scaleLoss(loss: Tensor): Tensor {
        return loss.mul(this.scaleInternal);
    }

    public unscale(parameters: readonly Parameter[]): boolean {
        let foundNonFinite: boolean = false;
        const inverseScale: number = 1 / this.scaleInternal;
        for (let index: number = 0; index < parameters.length; index += 1) {
            const parameter: Parameter | undefined = parameters[index];
            if (parameter === undefined) {
                throw new Error(`Missing parameter at index ${index}.`);
            }
            const gradient: Tensor | undefined = parameter.grad;
            if (gradient === undefined) {
                continue;
            }
            const values: Float32Array = gradient.toFloat32Array();
            for (let offset: number = 0; offset < values.length; offset += 1) {
                const value: number | undefined = values[offset];
                if (value === undefined || !Number.isFinite(value)) {
                    foundNonFinite = true;
                    break;
                }
            }
            if (!foundNonFinite) {
                parameter.scaleGradient(inverseScale);
            }
        }
        return foundNonFinite;
    }

    public update(foundNonFinite: boolean): LossScaleUpdate {
        if (foundNonFinite) {
            this.scaleInternal = Math.max(1, this.scaleInternal * this.backoffFactor);
            this.finiteSteps = 0;
        } else {
            this.finiteSteps += 1;
            if (this.finiteSteps >= this.growthInterval) {
                this.scaleInternal *= this.growthFactor;
                this.finiteSteps = 0;
            }
        }
        return { foundNonFinite, scale: this.scaleInternal };
    }
}

function validatePositive(value: number, label: string): number {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`${label} must be finite and > 0. Received ${value}.`);
    return value;
}
function validateGreaterThanOne(value: number, label: string): number {
    if (!Number.isFinite(value) || value <= 1) throw new Error(`${label} must be finite and > 1. Received ${value}.`);
    return value;
}
function validateFraction(value: number, label: string): number {
    if (!Number.isFinite(value) || value <= 0 || value >= 1)
        throw new Error(`${label} must be between 0 and 1. Received ${value}.`);
    return value;
}
function validatePositiveInteger(value: number, label: string): number {
    if (!Number.isInteger(value) || value <= 0)
        throw new Error(`${label} must be a positive integer. Received ${value}.`);
    return value;
}
