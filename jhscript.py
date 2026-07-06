import subprocess
import sys

def run(cmd):
    print(f"\n>>> RUN: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(result.stdout)
    print(result.stderr)

print("Python version:", sys.version)

# 1) pip 버전 확인
run("python -m pip --version")

# 2) pip 없으면 설치 (ensurepip)
run("python -m ensurepip --default-pip")

# 3) pip 다시 버전 확인
run("python -m pip --version")

# 4) rl-games 버전 확인
run("python -m pip show rl-games")

# 5) stable-baselines 설치 여부 확인
run("python -m pip show stable-baselines")
run("python -m pip show stable-baselines3")
