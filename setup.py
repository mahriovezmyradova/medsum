from setuptools import setup, find_packages

setup(
    name="medical-asr-summarization",
    version="1.0.0",
    description="ASR and Summarization for Medical Conversations in German",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
    python_requires=">=3.8",
)