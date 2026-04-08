import subprocess
import sys

with open('test_results_part1.txt', 'w', encoding='utf-8') as f:
    result = subprocess.run(['pytest', 'tests/', '-v', '--tb=short'], stdout=f, stderr=subprocess.STDOUT)
sys.exit(result.returncode)
