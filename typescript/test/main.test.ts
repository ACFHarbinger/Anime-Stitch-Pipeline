import { describe, expect, it } from "vitest";
import { greet } from "../src/main";

describe("greet", () => {
  it("returns a greeting for the given name", () => {
    expect(greet("Dev-Repo-Template")).toBe("Hello, Dev-Repo-Template!");
  });

  it("handles the default case", () => {
    expect(greet("world")).toBe("Hello, world!");
  });
});
