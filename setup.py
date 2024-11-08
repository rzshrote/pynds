#!/usr/bin/env python3
import pathlib
import setuptools

# setup.py metadata
setup_location = "." # pathlib.Path(__file__).parents[1]

# package metadata: general descriptors
pynds_name = "pynds"
pynds_version = "1.0.0"
pynds_author = "Robert Z. Shrote"
pynds_author_email = "rzshrote@gmail.com"
pynds_description = "Python package for non-dominated sorting"
with open("README.md", "r", encoding = "utf-8") as readme_file:
    pynds_description_long = readme_file.read()
    pynds_description_long_type = "text/markdown"

# package metadata: project URLs
pynds_url = "https://github.com/rzshrote/pynds"
pynds_project_url = {
    "Bug Tracker": "https://github.com/rzshrote/pynds/issues",
}

# package metadata: licensing and classifiers
pynds_license = "Apache License 2.0"
pynds_classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

# package metadata: installation requirements
pynds_requirements_python = ">=3.6"
pynds_requirements_install = [
    "numpy",
]

# package metadata: package locations
pynds_package_directory = {"" : setup_location}
pynds_packages = setuptools.find_packages(where = setup_location)

# setup the package
setuptools.setup(
    name = pynds_name,
    version = pynds_version,
    author = pynds_author,
    author_email = pynds_author_email,
    description = pynds_description,
    long_description = pynds_description_long,
    long_description_content_type = pynds_description_long_type,
    url = pynds_url,
    project_urls = pynds_project_url,
    license = pynds_license,
    classifiers = pynds_classifiers,
    package_dir = pynds_package_directory,
    packages = pynds_packages,
    python_requires = pynds_requirements_python,
    install_requires = pynds_requirements_install
)
