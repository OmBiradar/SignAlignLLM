# Integrating Multimodal Sign Language Processing into Large Language Models


<p align="center">
  <a href="https://aclanthology.org/2025.findings-acl.190.pdf">Original paper</a> | <a href="https://github.com/Merterm/signAlignLM">Original code repository</a>
</p>


This is a research project for aligning sign language (specifically German Sign Language / DGS) with large language models (LLMs).


## Setup

Download and install `uv` for project setup and dependency management. Then run

```bash
uv sync
```

to setup virtual python environment. Then run

```bash
pre-commit install
```

to activate the git pre-commit hooks (formats Python files with `black` on every commit).
