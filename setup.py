"""Setup script for DatBench evaluation library."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="datbench",
    version="1.0.0",
    author="Datology AI",
    description="Official evaluation harness for DatBench vision-language benchmarks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/datologyai/DatBench",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "datasets>=2.0.0",
        "pillow>=9.0.0",
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "anls>=0.1.0",  # For ANLS scoring (DocVQA, InfoVQA, OCR-VQA)
        "apted>=1.0.3",  # For TEDS table evaluation scoring
        "editdistance>=0.6.0",  # For string distance metrics
        "latex2sympy2>=1.9.0",  # For MathVision LaTeX evaluation
        "lxml>=4.9.0",  # For HTML table parsing
        "zss>=1.2.0",  # For tree edit distance (nTED)
    ],
)
