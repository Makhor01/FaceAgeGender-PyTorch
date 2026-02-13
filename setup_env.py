import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = [
    "torch==2.1.0",
    "torchvision==0.17.1",
    "numpy==1.25.2",
    "opencv-python==4.7.0.72",
    "Pillow==10.0.0"
]

for pkg in packages:
    print(f"Installing {pkg}...")
    install(pkg)

print("\nВсе зависимости установлены. Можете запускать скрипт run_camera.py")