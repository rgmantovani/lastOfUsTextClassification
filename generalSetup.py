import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

NUM_CORES = 8

def run_experiment(algoritmo, feature, dataset, seed):
    cmd = ["python", "main.py", algoritmo, feature, dataset, str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr

setups = []
with open("setup.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        setups.append((row["dataset"], row["feature"], row["algoritmo"], int(row["seed"])))

# Executor para escalonar os experimentos
with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
    futures = {executor.submit(run_experiment, *setup): setup for setup in setups}

    for future in as_completed(futures):
        setup = futures[future]
        try:
            output, error = future.result()
            print(f"Setup {setup} completed with output:\n{output}")
            if error:
                print(f"Setup {setup} had errors:\n{error}")
        except Exception as exc:
            print(f"Setup {setup} generated an exception: {exc}")
