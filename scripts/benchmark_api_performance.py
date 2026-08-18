# -*- coding: utf-8 -*-
"""
Phase 12: FastAPI Async Performance & Concurrency Load Benchmark
Measures throughput, latency percentiles (Mean, Median, P95, P99, Max), and cache speedup.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import statistics
import concurrent.futures
import pandas as pd
from typing import List, Dict, Any
from fastapi.testclient import TestClient
from app.main import app
from app.core.cache import get_recommendation_cache


def run_benchmark_for_endpoint(
    client: TestClient,
    method: str,
    url: str,
    payload: Dict[str, Any] = None,
    concurrency: int = 1,
    num_requests: int = 50
) -> Dict[str, Any]:
    """Runs concurrent load benchmark against a specific endpoint."""
    latencies = []
    errors = 0

    def make_request():
        t0 = time.perf_counter()
        try:
            if method == "GET":
                res = client.get(url)
            else:
                res = client.post(url, json=payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if res.status_code == 200:
                return latency_ms, True
            else:
                return latency_ms, False
        except Exception:
            return 0.0, False

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            lat, success = f.result()
            if success:
                latencies.append(lat)
            else:
                errors += 1
    total_time_s = time.perf_counter() - t_start
    throughput = len(latencies) / total_time_s if total_time_s > 0 else 0.0

    if not latencies:
        return {
            "concurrency": concurrency,
            "requests": num_requests,
            "successful": 0,
            "errors": errors,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "throughput_rps": 0.0
        }

    sorted_lats = sorted(latencies)
    p95_idx = int(0.95 * len(sorted_lats))
    p99_idx = int(0.99 * len(sorted_lats))

    return {
        "concurrency": concurrency,
        "requests": num_requests,
        "successful": len(latencies),
        "errors": errors,
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted_lats[min(p95_idx, len(sorted_lats)-1)], 2),
        "p99_ms": round(sorted_lats[min(p99_idx, len(sorted_lats)-1)], 2),
        "max_ms": round(max(latencies), 2),
        "throughput_rps": round(throughput, 1)
    }


def main():
    print("=" * 105, flush=True)
    print("PHASE 12: FASTAPI ASYNC PERFORMANCE & CONCURRENCY BENCHMARK", flush=True)
    print("=" * 105, flush=True)

    client = TestClient(app)
    cache = get_recommendation_cache()
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "phase_12")
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Warm-up
    print("\n[*] Initializing and warming up API endpoints...", flush=True)
    client.get("/ready")
    client.get("/api/v1/recommendations/popular?top_k=5")
    cache.clear()

    endpoints = [
        ("Popular (/popular)", "GET", "/api/v1/recommendations/popular?top_k=10", None),
        ("Nearby (/nearby)", "GET", "/api/v1/recommendations/nearby?latitude=12.9716&longitude=77.5946&radius_km=5.0&top_k=10", None),
        ("Hybrid User (/hybrid/{id})", "GET", "/api/v1/recommendations/hybrid/2?top_k=10&mmr_enabled=false", None),
        ("Onboarding (/onboarding)", "POST", "/api/v1/recommendations/onboarding", {
            "favorite_cuisines": ["North Indian", "Mughlai"],
            "preferred_area": "Indiranagar",
            "max_budget_for_two": 1200,
            "top_k": 10,
            "mmr_enabled": False
        })
    ]

    # SECTION 1: Cache Speedup Benchmark
    print("\n" + "-" * 105, flush=True)
    print("SECTION 1: CACHE SPEEDUP BENCHMARK (Cold vs Warm Cache)", flush=True)
    print("-" * 105, flush=True)

    cache_results = []
    for name, method, url, payload in endpoints:
        cache.clear()
        # Cold request
        t0 = time.perf_counter()
        if method == "GET":
            r_cold = client.get(url)
        else:
            r_cold = client.post(url, json=payload)
        cold_lat = (time.perf_counter() - t0) * 1000.0

        # Warm requests (5 runs)
        warm_lats = []
        for _ in range(10):
            t0 = time.perf_counter()
            if method == "GET":
                r_warm = client.get(url)
            else:
                r_warm = client.post(url, json=payload)
            warm_lats.append((time.perf_counter() - t0) * 1000.0)
        warm_lat_avg = statistics.mean(warm_lats)
        speedup = cold_lat / warm_lat_avg if warm_lat_avg > 0 else 1.0

        print(f"  {name:<30} | Cold: {cold_lat:>7.2f} ms | Warm (Avg): {warm_lat_avg:>6.2f} ms | Speedup: {speedup:>6.1f}x", flush=True)
        cache_results.append({
            "endpoint": name,
            "cold_latency_ms": round(cold_lat, 2),
            "warm_latency_ms": round(warm_lat_avg, 2),
            "speedup_factor": round(speedup, 1)
        })

    pd.DataFrame(cache_results).to_csv(os.path.join(reports_dir, "cache_benchmark.csv"), index=False)

    # SECTION 2: Concurrency & Load Benchmark
    print("\n" + "-" * 105, flush=True)
    print("SECTION 2: CONCURRENCY & THROUGHPUT LOAD BENCHMARK (C = 1, 5, 10, 25, 50)", flush=True)
    print("-" * 105, flush=True)

    concurrency_levels = [1, 5, 10, 25, 50]
    load_results = []

    for name, method, url, payload in endpoints:
        print(f"\n[*] Benchmarking {name}:", flush=True)
        print(f"  {'Concurrency':<12} | {'Reqs':<6} | {'Mean (ms)':<10} | {'Median':<8} | {'P95 (ms)':<9} | {'Max (ms)':<9} | {'Throughput (req/s)':<18}", flush=True)
        print("  " + "-" * 88, flush=True)

        for c in concurrency_levels:
            num_reqs = max(c * 4, 20)
            res = run_benchmark_for_endpoint(
                client=client,
                method=method,
                url=url,
                payload=payload,
                concurrency=c,
                num_requests=num_reqs
            )
            res["endpoint"] = name
            load_results.append(res)
            print(f"  C = {c:<8} | {res['requests']:<6} | {res['mean_ms']:<10.2f} | {res['median_ms']:<8.2f} | {res['p95_ms']:<9.2f} | {res['max_ms']:<9.2f} | {res['throughput_rps']:<18.1f}", flush=True)

    pd.DataFrame(load_results).to_csv(os.path.join(reports_dir, "concurrency_load_benchmark.csv"), index=False)

    print("\n" + "=" * 105, flush=True)
    print("PHASE 12 PERFORMANCE BENCHMARK COMPLETE — Reports saved to reports/phase_12/", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    main()

