#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'halo_tracker' python package."""

import os
import toml

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "pyyaml",
    # halo_tracker.rsl_rl_ext subclasses rsl_rl internals of this release line.
    "rsl-rl-lib>=5.5.1,<5.6",
]

# Installation operation
setup(
    name="halo_tracker",
    packages=find_packages(include=["halo_tracker", "halo_tracker.*"]),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apple Sample Code License",
    include_package_data=True,
    python_requires=">=3.11",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)