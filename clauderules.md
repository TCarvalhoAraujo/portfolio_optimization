# CUDA Portfolio Optimization Project — Rules & Context

## Project Overview

This project focuses on the development of a GPU-accelerated financial portfolio optimization system using CUDA and parallel computing techniques.

The core problem addressed by the project is the computational complexity involved in large-scale financial simulations, especially Monte Carlo simulations used for stock price forecasting, risk estimation, and portfolio evaluation.

The implementation must support both CPU and GPU execution paths in order to benchmark and compare sequential versus parallel performance.

---

# Core Objectives

The project MUST:

- Simulate future stock price scenarios using Monte Carlo methods.
- Use historical market data as the probabilistic basis for simulations.
- Execute simulations on both CPU and GPU environments.
- Use CUDA to parallelize computationally expensive operations.
- Measure and compare execution performance between CPU and GPU.
- Train a Machine Learning model using simulated financial data.
- Evaluate portfolio efficiency using financial risk metrics.

---

# Required Technologies

The implementation SHOULD prioritize:

- CUDA for GPU acceleration
- Python and/or C++ for core implementation
- Parallel kernels for Monte Carlo execution
- GPU memory optimization techniques
- Benchmarking utilities for performance comparison

Optional libraries MAY include:

- Numba CUDA
- CuPy
- PyTorch
- RAPIDS
- NumPy
- Pandas
- Scikit-learn

---

# Monte Carlo Simulation Requirements

The Monte Carlo engine MUST:

- Generate multiple stochastic price trajectories.
- Use probabilistic models based on historical stock behavior.
- Support large-scale scenario generation.
- Be highly parallelizable on GPU.
- Minimize CPU bottlenecks during execution.

The GPU implementation SHOULD:

- Launch independent threads per simulation path.
- Reduce unnecessary host-device memory transfers.
- Optimize memory access patterns whenever possible.

---

# Machine Learning Requirements

The ML component SHOULD:

- Consume simulated financial data.
- Identify portfolio patterns and risk-return relationships.
- Assist in selecting efficient portfolios.
- Compare portfolio quality metrics across simulations.

Possible metrics include:

- Expected Return
- Volatility
- Sharpe Ratio
- Value at Risk (VaR)

---

# Benchmarking & Performance Rules

The project MUST compare CPU and GPU implementations using controlled benchmarks.

Performance evaluation SHOULD include:

- Execution time
- Speedup
- Scalability
- GPU utilization
- Computational efficiency

The comparison MUST clearly demonstrate:

- Advantages of parallel execution
- Limitations of GPU acceleration
- Scenarios where GPU computation is most beneficial

---

# Expected Outcomes

The final implementation is expected to demonstrate:

- Significant performance gains using GPU acceleration.
- Practical applicability of CUDA in quantitative finance.
- Scalability for large financial simulations.
- Real-world relevance for investment analysis and risk management systems.

---

# Practical Applications

The system MAY be applied to:

- Investment decision support
- Quantitative finance systems
- Risk analysis platforms
- Portfolio optimization tools
- Financial forecasting environments

---

# Team Members

- Thiago Carvalho de Araujo
- Pedro Mota Lindoso
- Miguel Procópio
- Gustavo Porto
- Rafael Angelim

---

# Additional Notes

- The project should maintain clear separation between CPU and GPU implementations for benchmarking purposes.
- Financial simulations should be reproducible and configurable.
- GPU kernels should prioritize scalability and efficient thread utilization.
- The implementation should document limitations, assumptions, and computational tradeoffs.
- The final deliverable should include practical demonstrations and performance analysis.
