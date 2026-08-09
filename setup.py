from setuptools import setup, find_packages

setup(
    name="vector-vault",
    version="7.4.9.15",
    packages=find_packages(),
    package_data={"vectorvault": ["model_catalog.json"]},
    include_package_data=True,
    author="VectorVault.io",
    author_email="john@johnrood.com",
    description="Quickly create RAG apps, Agents, and Unleash the full power of AI with Vector Vault",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/John-Rood/VectorVault",
    classifiers=[
        "License :: Other/Proprietary License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    data_files=[("", ["LICENSE"])],
    install_requires=[
        "numpy", "requests", "bs4", "google-cloud-storage", "annoy", "faiss-cpu",
        "openai", "tiktoken", "anthropic", "pymupdf", "google-genai",
    ],
)
