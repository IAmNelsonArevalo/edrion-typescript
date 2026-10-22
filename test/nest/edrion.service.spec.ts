import { describe, expect, it } from "vitest";
import { Edrion } from "../../src/core/index.js";
import { EdrionService } from "../../src/nest/index.js";

describe("EdrionService", () => {
  it("should expose the runtime instance", () => {
    const runtime = new Edrion(),
      service = new EdrionService(runtime);

    expect(service.getRuntime()).toBe(runtime);
  });

  it("should expose runtime information", () => {
    const runtime = new Edrion({
      observability: {
        enabled: false,
      },
    });

    const service = new EdrionService(runtime),
      info = service.getRuntimeInfo();

    expect(info.name).toBe("edrion-typescript");
    expect(info.observabilityEnabled).toBe(false);
  });
});
