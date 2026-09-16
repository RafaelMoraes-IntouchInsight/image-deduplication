import pathlib
import setuptools

setuptools.setup(
    name="image-deduplication",
    version="0.1.3",
    description="Finds images that have a high degree of similarity",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/farrael004/image-deduplication",
    author="Rafael Moraes",
    author_email="farrael004@gmail.com",
    license="Apache 2.0",
    project_urls={
        "Documentation": "https://github.com/farrael004/image-deduplication",
        "Source": "https://github.com/farrael004/image-deduplication",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=["opencv-python", "numpy"],
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["image-deduplication = image_deduplication.cli:main"]},
)