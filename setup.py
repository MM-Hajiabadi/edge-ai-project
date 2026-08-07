from setuptools import find_packages, setup

setup(
    name="edge_ai_anomaly_detection",
    version="0.1.0",
    description="MIMII acoustic anomaly detection with RTL/Verilog co-design",
    author="MM-Hajiabadi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "librosa",
        "scikit-learn",
        "pyyaml",
        "cocotb",
        "matplotlib",
    ],
    python_requires=">=3.10",
)
