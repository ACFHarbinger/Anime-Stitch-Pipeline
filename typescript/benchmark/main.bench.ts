import { bench, describe } from "vitest";
import { greet } from "../src/main";

describe("greet benchmark", () => {
  bench("greet", () => {
    greet("world");
  });
});
