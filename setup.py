from setuptools import setup, find_packages

setup(
    name="LLM-AMLTK",
    version="0.0.1",
    packages=find_packages(),
    description="Using LLM priming to warm-start automated machine learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Israel Campero Jurado and Joaquin Vanschoren",
    author_email="learsi1911@gmail.com",
    url="https://github.com/israel-cj/LLM-AMLTK.git",
    python_requires=">=3.10",
    install_requires=[
        "amltk[notebook, smac, optuna, sklearn]",
        "openai",
        "openml",
        "scikit-learn",
        "ipython",
    ],
)