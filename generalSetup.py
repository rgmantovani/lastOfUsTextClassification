import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# NUM_CORES = 8
# Sets automatically the number of cores
NUM_CORES = multiprocessing.cpu_count() - 1
print(NUM_CORES)

# -------------------------------
# Run a single job (ML experiment)
# -------------------------------

def runJob(algoritmo, feature, dataset, seed):
    cmd = ["python3", "main.py", algoritmo, feature, dataset, str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr

# -------------------------------
#  Run the complete experiment
# -------------------------------

setups = []
with open("setup.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        setups.append((row["dataset"], row["feature"], row["algorithm"], int(row["seed"])))

with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
    futures = {executor.submit(runJob, *setup): setup for setup in setups}

    for future in as_completed(futures):
        setup = futures[future]
        try:
            output, error = future.result()
            print(f"Setup {setup} completed with output:\n{output}")
            if error:
                print(f"Setup {setup} had errors:\n{error}")
        except Exception as exc:
            print(f"Setup {setup} generated an exception: {exc}")

# -------------------------------
# -------------------------------