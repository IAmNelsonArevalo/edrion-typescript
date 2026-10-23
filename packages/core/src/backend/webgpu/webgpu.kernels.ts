export const VECTOR_ADD_WGSL: string = `
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index < arrayLength(&output)) {
        output[index] = left[index] + right[index];
    }
}`;

export const MATRIX_MULTIPLY_WGSL: string = `
struct Dimensions {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
};

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: Dimensions;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.y;
    let column = id.x;
    if (row >= dimensions.m || column >= dimensions.n) {
        return;
    }

    var sum: f32 = 0.0;
    for (var inner: u32 = 0u; inner < dimensions.k; inner = inner + 1u) {
        sum = sum + left[row * dimensions.k + inner] * right[inner * dimensions.n + column];
    }

    output[row * dimensions.n + column] = sum;
}`;
