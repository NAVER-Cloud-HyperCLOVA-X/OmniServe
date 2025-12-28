# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

from typing import List
from setuptools import find_namespace_packages, setup


def get_requires() -> List[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip()
                 for line in file_content.strip().split("\n")
                 if not line.startswith("#") and not line.startswith("-")]
        return lines


setup(
    name="audiollm-encoder",
    version="0.1",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    python_requires=">=3.10",
    author="AudioLLM Contributors",
    url="",
    install_requires=get_requires(),
)
