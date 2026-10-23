import type { DType } from './dtype';

const float32Buffer: ArrayBuffer = new ArrayBuffer(4);
const float32View: Float32Array = new Float32Array(float32Buffer);
const uint32View: Uint32Array = new Uint32Array(float32Buffer);

/** Quantizes a JavaScript number to the precision represented by an Edrion dtype. */
export function quantizeToDType(value: number, dtype: DType): number {
    if (dtype === 'float32') {
        return Math.fround(value);
    }
    if (dtype === 'bfloat16') {
        return quantizeBFloat16(value);
    }
    return quantizeFloat16(value);
}

/** Emulates IEEE-754 binary16 while keeping CPU storage accessible as Float32Array. */
export function quantizeFloat16(value: number): number {
    if (!Number.isFinite(value) || value === 0) {
        return value;
    }

    float32View[0] = value;
    const bits: number = uint32View[0] ?? 0;
    const sign: number = (bits >>> 16) & 0x8000;
    let exponent: number = ((bits >>> 23) & 0xff) - 127 + 15;
    let mantissa: number = bits & 0x7fffff;
    let half: number;

    if (exponent <= 0) {
        if (exponent < -10) {
            half = sign;
        } else {
            mantissa = (mantissa | 0x800000) >>> (1 - exponent);
            if ((mantissa & 0x1000) !== 0) {
                mantissa += 0x2000;
            }
            half = sign | (mantissa >>> 13);
        }
    } else if (exponent >= 31) {
        half = sign | 0x7c00;
    } else {
        if ((mantissa & 0x1000) !== 0) {
            mantissa += 0x2000;
            if ((mantissa & 0x800000) !== 0) {
                mantissa = 0;
                exponent += 1;
            }
        }
        half = exponent >= 31 ? sign | 0x7c00 : sign | (exponent << 10) | (mantissa >>> 13);
    }

    return decodeFloat16(half);
}

/** Emulates bfloat16 round-to-nearest-even precision. */
export function quantizeBFloat16(value: number): number {
    if (!Number.isFinite(value)) {
        return value;
    }
    float32View[0] = value;
    let bits: number = uint32View[0] ?? 0;
    const lsb: number = (bits >>> 16) & 1;
    bits = (bits + 0x7fff + lsb) & 0xffff0000;
    uint32View[0] = bits;
    return float32View[0] ?? value;
}

function decodeFloat16(bits: number): number {
    const sign: number = (bits & 0x8000) !== 0 ? -1 : 1;
    const exponent: number = (bits >>> 10) & 0x1f;
    const fraction: number = bits & 0x3ff;

    if (exponent === 0) {
        return sign * Math.pow(2, -14) * (fraction / 1024);
    }
    if (exponent === 31) {
        return fraction === 0 ? sign * Infinity : NaN;
    }
    return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}
