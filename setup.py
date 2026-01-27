from setuptools import setup, find_packages

setup(
    name="sleepfm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # Add dependencies here if needed
    description="A package for sleepfm clinical pipelines.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-repo-url",  # Replace with your repo URL if applicable
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)