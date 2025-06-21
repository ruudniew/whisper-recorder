from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="whisper-transcriber",
    version="0.1.0",
    author="WhisperLive Team",
    description="Real-time transcription app for macOS",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "whisper-transcriber=whisper_transcriber.main:main",
        ],
    },
    package_data={
        "whisper_transcriber": ["resources/*.png"],
    },
)