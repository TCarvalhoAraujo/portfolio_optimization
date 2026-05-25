/*
mc_multi_portfolio_kernel.cu
============================
Kernel CUDA para simulação de Monte Carlo de N portfólios em paralelo.

Diferença do mc_kernel.cu
--------------------------
    mc_kernel.cu              : 1 vetor de pesos → n_sims simulações
    mc_multi_portfolio_kernel : N vetores de pesos → N × n_sims simulações

Sem este kernel, avaliar N portfólios exige N chamadas de kernel separadas
(Python loop + overhead de launch por chamada). Aqui, um único launch cobre
tudo — o gargalo passa a ser a própria computação, não a orquestração.

MAPEAMENTO DE THREADS
=====================

    Grid 2D:
        blockIdx.y                               = índice do portfólio p
        blockIdx.x * blockDim.x + threadIdx.x   = índice da simulação  s

    Thread (p, s) é responsável por simular a trajetória s do portfólio p.

Por que Grid 2D?
    Todos os threads em um block têm o mesmo blockIdx.y (mesmo portfólio).
    Isso permite carregar os pesos desse portfólio em __shared__ memory
    uma única vez por block — em vez de cada thread ler de global memory.

SHARED MEMORY POR BLOCK
=======================
    L_shared[n_assets²] — matriz de Cholesky (igual para todos os portfólios,
                          mas não pode ser compartilhada entre blocks CUDA)
    W_shared[n_assets]  — pesos do portfólio blockIdx.y

    Limite prático de n_assets:
        (n_assets² + n_assets) × 4 bytes ≤ 48 KB  →  n_assets ≤ ~109
    Para portfólios típicos (10–60 ativos) há espaço de sobra.

SEMENTE DO RNG
==============
    Cada thread recebe uma semente única:
        seed = seed_base + p * n_sims + s

    Isso é O(1) em tempo de inicialização, independente do valor da semente.
    Usar sequence offset (curand_init(seed, sequence, ...)) seria O(sequence)
    e ficaria proibitivo com milhões de threads.

SAÍDA
=====
    out_returns[p * n_sims + s] = retorno do portfólio p na simulação s
    Shape flat: n_portfolios × n_sims  (row-major, linha = portfólio)
*/

#include <curand_kernel.h>
#include <math.h>

#ifndef MAX_ASSETS
#define MAX_ASSETS 128
#endif

extern "C" __global__ void mc_gbm_batch_kernel(
    /* Parâmetros do modelo GBM ─────────────────────── */
    const float* __restrict__ drift,        // (mu - 0.5*sigma²)*dt  [n_assets]
    const float* __restrict__ diffusion,    // sigma * sqrt(dt)       [n_assets]
    const float* __restrict__ chol_lower,   // matriz L, row-major    [n_assets × n_assets]
    const float* __restrict__ weights_all,  // pesos de todos os portfólios [n_portfolios × n_assets]
    /* Dimensões ─────────────────────────────────────── */
    const int n_assets,
    const int n_steps,
    const int n_sims,
    const int n_portfolios,
    /* Aleatoriedade ────────────────────────────────── */
    const unsigned long long seed_base,
    /* Saída ────────────────────────────────────────── */
    float* __restrict__ out_returns         // retornos [n_portfolios × n_sims]
)
{
    /* ── 1. ÍNDICES ─────────────────────────────────────────────────────── */
    int p = blockIdx.y;                              // portfólio
    int s = blockIdx.x * blockDim.x + threadIdx.x;  // simulação

    if (p >= n_portfolios || s >= n_sims) return;

    /* ── 2. SHARED MEMORY ───────────────────────────────────────────────── */
    __shared__ float L_shared[MAX_ASSETS * MAX_ASSETS];
    __shared__ float W_shared[MAX_ASSETS];

    int tid = threadIdx.x;
    int bsz = blockDim.x;

    /* Cholesky — igual para todos os portfólios */
    int total_L = n_assets * n_assets;
    for (int idx = tid; idx < total_L; idx += bsz)
        L_shared[idx] = chol_lower[idx];

    /* Pesos do portfólio p */
    const float* w_p = weights_all + (long long)p * n_assets;
    for (int idx = tid; idx < n_assets; idx += bsz)
        W_shared[idx] = w_p[idx];

    __syncthreads();

    /* ── 3. RNG — semente única por (portfólio, simulação) ──────────────── */
    /*
       Usamos seed único por thread (não sequence offset) porque
       curand_init(seed, large_sequence, 0) é O(sequence) — muito lento
       com milhões de threads. Semente única é O(1).
    */
    curandState rng_state;
    curand_init(
        seed_base + (unsigned long long)p * n_sims + s,
        0ULL,
        0ULL,
        &rng_state
    );

    /* ── 4. ESTADO INICIAL DO LOG-PREÇO ─────────────────────────────────── */
    float log_price[MAX_ASSETS];
    for (int i = 0; i < n_assets; i++) log_price[i] = 0.0f;

    /* ── 5. LOOP GBM ─────────────────────────────────────────────────────── */
    float Z[MAX_ASSETS];
    float eps[MAX_ASSETS];

    for (int t = 0; t < n_steps; t++) {

        /* Choques independentes N(0,1) */
        for (int i = 0; i < n_assets; i++)
            Z[i] = curand_normal(&rng_state);

        /* Correlaciona via L: eps = L @ Z (triangular inferior) */
        for (int i = 0; i < n_assets; i++) {
            float dot = 0.0f;
            for (int j = 0; j <= i; j++)
                dot += L_shared[i * n_assets + j] * Z[j];
            eps[i] = dot;
        }

        /* GBM: log_price[i] += drift[i] + diffusion[i] * eps[i] */
        for (int i = 0; i < n_assets; i++)
            log_price[i] += drift[i] + diffusion[i] * eps[i];
    }

    /* ── 6. RETORNO DO PORTFÓLIO ─────────────────────────────────────────── */
    float portfolio_return = 0.0f;
    for (int i = 0; i < n_assets; i++) {
        float S_T = expf(log_price[i]);              // S_T / S_0
        portfolio_return += W_shared[i] * (S_T - 1.0f);
    }

    out_returns[(long long)p * n_sims + s] = portfolio_return;
}
