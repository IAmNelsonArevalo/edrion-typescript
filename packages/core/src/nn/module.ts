import { ModuleRegistrationError } from '@core/nn/nn.errors';
import type { Parameter } from '@core/nn/parameter';
import type { Tensor } from '@core/tensor/tensor';

/**
 * Named trainable parameter exposed by a Module hierarchy.
 */
export interface NamedParameter {
    readonly name: string;
    readonly parameter: Parameter;
}

/**
 * Base class for all Edrion neural-network modules.
 */
export abstract class Module {
    private readonly parameterRegistry: Map<string, Parameter> = new Map<string, Parameter>();

    private readonly moduleRegistry: Map<string, Module> = new Map<string, Module>();

    private trainingMode: boolean = true;

    /**
     * Executes the module forward pass.
     */
    public abstract forward(input: Tensor): Tensor;

    /**
     * Indicates whether the module is in training mode.
     */
    public get training(): boolean {
        return this.trainingMode;
    }

    /**
     * Enables training mode recursively.
     */
    public train(): this {
        this.trainingMode = true;

        const modules: readonly Module[] = Array.from(this.moduleRegistry.values());

        for (let index: number = 0; index < modules.length; index += 1) {
            const module: Module | undefined = modules[index];

            if (module === undefined) {
                throw new ModuleRegistrationError(`Missing child module at registration index ${index}.`);
            }

            module.train();
        }

        return this;
    }

    /**
     * Enables evaluation mode recursively.
     */
    public eval(): this {
        this.trainingMode = false;

        const modules: readonly Module[] = Array.from(this.moduleRegistry.values());

        for (let index: number = 0; index < modules.length; index += 1) {
            const module: Module | undefined = modules[index];

            if (module === undefined) {
                throw new ModuleRegistrationError(`Missing child module at registration index ${index}.`);
            }

            module.eval();
        }

        return this;
    }

    /**
     * Returns every trainable parameter owned by this module hierarchy.
     */
    public parameters(): readonly Parameter[] {
        const namedParameters: readonly NamedParameter[] = this.namedParameters();

        const parameters: Parameter[] = new Array<Parameter>(namedParameters.length);

        for (let index: number = 0; index < namedParameters.length; index += 1) {
            const namedParameter: NamedParameter | undefined = namedParameters[index];

            if (namedParameter === undefined) {
                throw new ModuleRegistrationError(`Missing named parameter at index ${index}.`);
            }

            parameters[index] = namedParameter.parameter;
        }

        return parameters;
    }

    /**
     * Returns recursively named parameters using dotted module paths.
     */
    public namedParameters(prefix: string = ''): readonly NamedParameter[] {
        const result: NamedParameter[] = [];

        const parameterEntries: readonly [string, Parameter][] = Array.from(this.parameterRegistry.entries());

        for (let index: number = 0; index < parameterEntries.length; index += 1) {
            const entry: [string, Parameter] | undefined = parameterEntries[index];

            if (entry === undefined) {
                throw new ModuleRegistrationError(`Missing parameter registration at index ${index}.`);
            }

            const name: string = entry[0];
            const parameter: Parameter = entry[1];

            result.push({
                name: prefix.length === 0 ? name : `${prefix}.${name}`,
                parameter,
            });
        }

        const moduleEntries: readonly [string, Module][] = Array.from(this.moduleRegistry.entries());

        for (let index: number = 0; index < moduleEntries.length; index += 1) {
            const entry: [string, Module] | undefined = moduleEntries[index];

            if (entry === undefined) {
                throw new ModuleRegistrationError(`Missing child module registration at index ${index}.`);
            }

            const name: string = entry[0];
            const module: Module = entry[1];
            const childPrefix: string = prefix.length === 0 ? name : `${prefix}.${name}`;

            const childParameters: readonly NamedParameter[] = module.namedParameters(childPrefix);

            result.push(...childParameters);
        }

        return result;
    }

    /**
     * Clears gradients from every trainable parameter.
     */
    public zeroGrad(): void {
        const parameters: readonly Parameter[] = this.parameters();

        for (let index: number = 0; index < parameters.length; index += 1) {
            const parameter: Parameter | undefined = parameters[index];

            if (parameter === undefined) {
                throw new ModuleRegistrationError(`Missing parameter at index ${index}.`);
            }

            parameter.zeroGrad();
        }
    }

    /**
     * Registers a trainable parameter owned by this module.
     */
    protected registerParameter(name: string, parameter: Parameter): Parameter {
        this.assertRegistrationName(name);

        if (this.parameterRegistry.has(name) || this.moduleRegistry.has(name)) {
            throw new ModuleRegistrationError(`Module member "${name}" is already registered.`);
        }

        this.parameterRegistry.set(name, parameter);
        return parameter;
    }

    /**
     * Registers a child module.
     */
    protected registerModule<TModule extends Module>(name: string, module: TModule): TModule {
        this.assertRegistrationName(name);

        if (this.moduleRegistry.has(name) || this.parameterRegistry.has(name)) {
            throw new ModuleRegistrationError(`Module member "${name}" is already registered.`);
        }

        this.moduleRegistry.set(name, module);

        if (!this.trainingMode) {
            module.eval();
        }

        return module;
    }

    private assertRegistrationName(name: string): void {
        const normalizedName: string = name.trim();

        if (normalizedName.length === 0) {
            throw new ModuleRegistrationError('Module registration name cannot be empty.');
        }

        if (normalizedName.includes('.')) {
            throw new ModuleRegistrationError(`Module registration name "${normalizedName}" cannot contain dots.`);
        }
    }
}
