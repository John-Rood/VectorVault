from setuptools import setup, find_packages

setup(
    name="vector_vault",
    version="0.1.2",
    packages=find_packages(),
    author="VectorVault.io",
    author_email="john@johnrood.com",
    description="Vector Vault: Simplified vector database management and secure cloud storage for data science and machine learning workflows.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="http://github.com/John-Rood/vector_vault",
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
        'openai',
        'annoy',
        'google-cloud-storage',
        'requests',
        'bs4',
        # add any other dependencies your package needs
    ],
)
