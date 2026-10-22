import { describe, expect, it } from "vitest";

import { Edrion } from "../../src/core/index.js";

describe("Edrion runtime", () => {
  it("should create a default runtime", () => {
    const runtime = new Edrion(),
      info = runtime.getRuntimeInfo();

    expect(info.name).toBe("edrion-typescript");
    expect(info.version).toBe("0.1.0");
    expect(info.backend).toBe("cpu");
    expect(info.observabilityEnabled).toBe(true);
  });

  it("should accept custom configuration", () => {
    const runtime = new Edrion({
      backend: "cpu",
      observability: {
        enabled: false,
      },
    });

    const options = runtime.getOptions();

    expect(options.backend).toBe("cpu");
    expect(options.observability.enabled).toBe(false);
  });

  it("should normalize empty configuration", () => {
    const runtime = new Edrion({});

    expect(runtime.getOptions()).toEqual({
      backend: "cpu",
      observability: {
        enabled: true,
      },
    });
  });
});
