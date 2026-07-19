# Optimization of Large-Scale Distributed Vector Databases for Multi-Modal Retrieval Systems

## Project Overview
This repository serves as the complete, open-access research engine for a quantitative data analytics capstone project at Walsh College (QM640). 

The goal of this study is to model performance behaviors, build a supervised machine learning telemetry framework to predict query search latency ($99^{\text{th}}$ percentile execution time), and calculate optimal distributed shard partitioning limits for multi-modal vector search spaces.

## Core Architecture & Directory Layout
* `data/raw/`: Pre-computed embedding datasets pulled directly via HDF5 streams.
* `data/processed/`: Extracted database latency, index building, and memory usage profiling telemetry.
* `notebooks/`: Active exploratory data analysis (EDA), data parsing, and regression validation pipelines.
* `src/`: Low-level multi-threaded search configuration, index compilation, and performance instrumentation tools.

## The Performance Data Schema
The compiled analysis database captures the following runtime metric features:
1. `Index_Type` (Categorical): Approximate nearest neighbor routing logic applied (`HNSW`, `IVF-PQ`, `Flat`).
2. `Vector_Dimensions` (Integer): Spatial dimension size ($100$, $128$, or $300$).
3. `Distance_Metric` (Categorical): Proximity calculation metric used (`Cosine`, `Euclidean`, `Inner_Product`).
4. `Build_Time_Sec` (Continuous): Index compilation processing speed in seconds.
5. `Memory_Usage_MB` (Continuous): Absolute RAM footprint allocation for the running graph structure.
6. `Query_Latency_MS` (Continuous): Wall-clock performance turnaround speed per inquiry task.
7. `Recall_At_K` (Continuous): Retrieval precision accuracy benchmarked against brute-force linear baselines.

## Baseline Research Datasets
All experimental configurations utilize non-Kaggle, standardized benchmarks pulled directly from the open-source **ANN-Benchmarks framework**:
* GloVe Text Vector Embeddings ($100$-dimensional and $300$-dimensional vectors)
* SIFT Computer Vision Features ($128$-dimensional vectors)

## Academic Context
* **Course:** QM640: Data Analytics Capstone
* **Institution:** Walsh College
