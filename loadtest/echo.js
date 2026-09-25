// k6 load test: replays loadtest/workload.json against the gateway at a fixed
// arrival rate. Run via loadtest/run.sh, which also produces the report.
//
// Env: RATE (requests/s, default 10), MODEL (default claude-haiku-4-5),
//      WORKLOAD (file in loadtest/, default workload.json),
//      ECHO_URL (default http://gateway:8080, i.e. on the compose network),
//      RUN_ID (default: timestamp; part of the system prompt, so every run
//      starts with a cold cache).

import http from "k6/http";
import exec from "k6/execution";
import { check } from "k6";
import { SharedArray } from "k6/data";
import { Counter, Trend } from "k6/metrics";

const workload = new SharedArray("workload", () => JSON.parse(open(`./${__ENV.WORKLOAD || "workload.json"}`)));
const BASE = __ENV.ECHO_URL || "http://gateway:8080";
const MODEL = __ENV.MODEL || "claude-haiku-4-5";
const RATE = parseInt(__ENV.RATE || "10", 10);
const RUN_ID = __ENV.RUN_ID || `${Date.now()}`;
const SYSTEM = `Answer in one short sentence. (load test ${RUN_ID})`;

// Tagged with category (exact/paraphrase/unique) and cache (hit/miss/...);
// report.py aggregates the raw points.
const latency = new Trend("echo_latency", true);
const inputTokens = new Counter("echo_input_tokens");
const outputTokens = new Counter("echo_output_tokens");

export const options = {
  scenarios: {
    mixed: {
      executor: "constant-arrival-rate",
      rate: RATE,
      timeUnit: "1s",
      duration: `${Math.ceil(workload.length / RATE)}s`,
      preAllocatedVUs: 50,
      maxVUs: 300,
    },
  },
  summaryTrendStats: ["p(50)", "p(95)", "p(99)", "max"],
};

export default function () {
  const i = exec.scenario.iterationInTest;
  if (i >= workload.length) return;
  const item = workload[i];

  const res = http.post(
    `${BASE}/v1/chat/completions`,
    JSON.stringify({
      model: MODEL,
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user", content: item.prompt },
      ],
    }),
    { headers: { "content-type": "application/json" }, timeout: "120s" },
  );

  const cache = res.headers["X-Echo-Cache"] || "error";
  const tags = { category: item.category, cache, status: String(res.status) };
  check(res, { "status is 200": (r) => r.status === 200 });
  latency.add(res.timings.duration, tags);

  if (res.status === 200) {
    // A hit carries the usage of the call that produced it: what this
    // request would have cost without the cache.
    const usage = res.json("usage") || {};
    inputTokens.add(usage.prompt_tokens || 0, tags);
    outputTokens.add(usage.completion_tokens || 0, tags);
  }
}
