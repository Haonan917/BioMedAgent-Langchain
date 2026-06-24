# BioMedAgent-Langchain

BioMedAgent-Langchain is a LangChain-based reimplementation of **BioMedAgent**, designed to provide a cleaner, more modular, and more extensible framework for biomedical multi-agent workflows.

The current version reimplements most core agent components and workflow logic from BioMedAgent using LangChain. The only major component that has not yet been implemented is the **programming agent**. This is intentional: the original implementation of the programming agent is relatively verbose and contains many redundant workflow-specific details, so BioMedAgent-Langchain plans to reimplement this component later with a simpler and more maintainable design.

## Overview

BioMedAgent-Langchain aims to support biomedical task solving through coordinated language-model agents. The system decomposes complex biomedical analysis tasks into multiple subtasks, assigns them to specialized agents, and manages their intermediate states through a shared task status object.

Compared with the original BioMedAgent implementation, BioMedAgent-Langchain focuses on:

* A cleaner LangChain-based architecture
* More readable agent definitions
* Modular prompt and action templates
* Easier workflow customization
* Better separation between task state, agent behavior, and response handling
* Future compatibility with LangGraph-based workflow orchestration

## Current Status

BioMedAgent-Langchain is currently under active development.

Implemented components include:

* Core agent abstraction
* Task and status management
* Prompt template formatting
* Multiple biomedical reasoning agents
* Agent response parsing
* Retry and validation mechanisms
* Multi-agent workflow coordination in `demo.py`

Not yet implemented:

* Programming agent / code execution agent

The programming agent will be redesigned later rather than directly copied from the original BioMedAgent implementation.


## Installation

Clone the repository and enter the project directory:

```bash
cd BioMedAgent-Langchain
```

Create a Python 3.10 virtual environment using `uv`:

```bash
uv venv --python 3.10
```

Activate the environment:

```bash
source .venv/bin/activate
```

Initialize the project environment:

```bash
uv init
```

Then install the required dependencies according to the project configuration.

For example, if dependencies are listed in `pyproject.toml`, run:

```bash
uv sync
```

## Usage

A typical entry point is `demo.py`.

Example:

```bash
python demo.py --task test
```

The exact task name and required input files depend on the workflow you want to run.

## Roadmap

Planned improvements include:

* Implementing a redesigned programming agent
* Refactoring `demo.py` with LangGraph
* Improving token efficiency during multi-agent collaboration
* Adding clearer workflow-level state transitions
* Adding more robust error handling
* Improving documentation for each agent
* Adding reproducible examples
* Adding unit tests for core components
* Supporting more biomedical tools and workflows

