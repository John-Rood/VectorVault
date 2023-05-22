from setuptools import setup, find_packages

setup(
    name="vector_vault",
    version="0.2.4",
    packages=find_packages(),
    author="VectorVault.io",
    author_email="john@johnrood.com",
    description="VectorVault: Simplified vector database management in the cloud for machine learning and generative ai workflows",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/John-Rood/VectorVault",
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[
        'numpy',
        'requests',
        'bs4',
        'google-cloud-storage',
        'annoy',
        'openai',
        'tiktoken',
        # any other dependencies
    ],
)
