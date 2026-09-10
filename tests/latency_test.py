import time
import requests
import statistics

URL = "http://127.0.0.1:8000/predict"

payload = {
    "season": 2,
    "yr": 1,
    "mnth": 6,
    "hr": 18,
    "holiday": 0,
    "weekday": 2,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.62,
    "atemp": 0.60,
    "hum": 0.55,
    "windspeed": 0.19
}

latencies = []

NUMBER_OF_REQUESTS = 100

for i in range(NUMBER_OF_REQUESTS):

    start = time.perf_counter()

    response = requests.post(URL, json=payload)

    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    latencies.append(latency_ms)

average = statistics.mean(latencies)

p95 = sorted(latencies)[int(0.95 * len(latencies))]

print("=" * 40)
print("Latency Test Results")
print("=" * 40)

print(f"Requests : {NUMBER_OF_REQUESTS}")
print(f"Average  : {average:.2f} ms")
print(f"P95      : {p95:.2f} ms")