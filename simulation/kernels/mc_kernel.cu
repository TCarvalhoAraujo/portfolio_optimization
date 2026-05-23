/*
mc_kernel.cu
============
Kernel CUDA para simulação de Monte Carlo de portfólios via GBM.

CONCEITOS FUNDAMENTAIS PARA INICIANTES
=======================================

1. HIERARQUIA DE THREADS
   GPU organiza threads em dois níveis:
     - Grid  : conjunto de blocks  (até 2^31 blocks por dimensão)
     - Block : conjunto de threads (até 1024 threads por block)

   Cada thread tem um ID único calculado como:
     global_id = blockIdx.x * blockDim.x + threadIdx.x

   Neste kernel: cada thread = uma simulação de Monte Carlo.

2. MEMÓRIA GPU
   - Global  : grande (GBs), lenta, visível por todas as threads
   - Shared  : pequena (48 KB/block), muito rápida, visível no block
   - Registers: mínima, ultra-rápida, privada por thread

   Estratégia aqui:
     - Matriz de Cholesky (L) → __shared__  (todos do block acessam)
     - mu, sigma, drift, diff  → __constant__ seria ideal, mas usamos
                                 parâmetros passados (mais flexível)
     - Preços/retornos finais  → global (escrita única no fim)

3. GERAÇÃO DE NÚMEROS ALEATÓRIOS NA GPU (cuRAND)
   Na CPU usamos numpy.random. Na GPU usamos cuRAND:
     - curand_init()   : inicializa o estado RNG por thread
     - curand_normal() : gera um float N(0,1)

   Cada thread tem seu próprio estado RNG, garantindo independência
   entre simulações sem sincronização.

4. POR QUE SHARED MEMORY PARA CHOLESKY?
   A matriz L tem n_assets² elementos lidos n_steps vezes por thread.
   Se cada thread lesse da global memory: n_sims × n_steps × n_assets²
   leituras lentas. Carregando L em shared memory uma vez por block,
   reduzimos isso a uma leitura por elemento por block — muito mais rápido.

ALGORITMO POR THREAD (id = s)
==============================
  1. Inicializa cuRAND com seed única (seed_base + s)
  2. Carrega L em shared memory (colaborativamente com outras threads do block)
  3. Para t = 0 .. n_steps-1:
       a. Gera Z[i] ~ N(0,1) para cada ativo i
       b. Calcula eps[i] = sum_j L[i,j] * Z[j]   (produto L @ Z)
       c. log_price[i] += drift[i] + diffusion[i] * eps[i]
  4. Calcula S_T[i] = exp(log_price[i])
  5. Calcula retorno do portfólio = sum_i w[i] * (S_T[i] - 1)
  6. Escreve S_T e retorno nos arrays de saída (global memory)
*/

#include <curand_kernel.h>
#include <math.h>

/* ─────────────────────────────────────────────────────────────────────────
   CONSTANTES DE COMPILAÇÃO
   Definidas em tempo de compilação pelo Python (via SourceModule).
   MAX_ASSETS limita o tamanho da shared memory alocada estaticamente.
   ───────────────────────────────────────────────────────────────────────── */
#ifndef MAX_ASSETS
#define MAX_ASSETS 128
#endif

/* ─────────────────────────────────────────────────────────────────────────
   KERNEL PRINCIPAL
   ───────────────────────────────────────────────────────────────────────── */
__global__ void mc_gbm_kernel(
    /* Parâmetros do modelo GBM ─────────────────────────── */
    const float* __restrict__ drift,       // (mu - 0.5*sigma²)*dt  [n_assets]
    const float* __restrict__ diffusion,   // sigma * sqrt(dt)       [n_assets]
    const float* __restrict__ chol_lower,  // matriz L, row-major    [n_assets × n_assets]
    const float* __restrict__ weights,     // pesos do portfólio     [n_assets]
    /* Dimensões ─────────────────────────────────────────── */
    const int n_assets,
    const int n_steps,
    const int n_sims,
    /* Aleatoriedade ────────────────────────────────────────*/
    const unsigned long long seed_base,
    /* Saídas ───────────────────────────────────────────────*/
    float* __restrict__ out_final_prices,  // S_T/S_0  [n_sims × n_assets]
    float* __restrict__ out_returns        // retorno do portfólio [n_sims]
)
{
    /* ── 1. ID GLOBAL DA THREAD ─────────────────────────────────────────── */
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_sims) return;   // threads extras além de n_sims: sem trabalho

    /* ── 2. SHARED MEMORY — carrega Cholesky colaborativamente ─────────── */
    /*
       __shared__ aloca memória compartilhada por BLOCK (não por thread).
       Todas as threads do block colaboram para carregar L da global memory.

       Por que colaborar? Cada thread precisaria de n_assets² leituras.
       Em vez disso, dividimos: cada thread carrega alguns elementos.
    */
    __shared__ float L_shared[MAX_ASSETS * MAX_ASSETS];

    int total_elements = n_assets * n_assets;
    int tid_local      = threadIdx.x;           // ID dentro do block
    int block_size     = blockDim.x;

    /* Cada thread carrega stride elements da matriz L */
    for (int idx = tid_local; idx < total_elements; idx += block_size) {
        L_shared[idx] = chol_lower[idx];
    }
    __syncthreads();   // barreira: garante que L_shared está completo antes de usar

    /* ── 3. INICIALIZA RNG DA THREAD ────────────────────────────────────── */
    /*
       curand_init(seed, sequence, offset, &state)
         seed     : semente base (igual para todas as threads)
         sequence : offset de sequência — usamos `s` para garantir
                    que cada thread gere uma sequência diferente
         offset   : deslocamento dentro da sequência (0 aqui)
    */
    curandState rng_state;
    curand_init(seed_base, (unsigned long long)s, 0ULL, &rng_state);

    /* ── 4. ESTADO INTERNO DA SIMULAÇÃO ─────────────────────────────────── */
    /*
       log_price[i] acumula ln(S_t / S_0) ao longo dos n_steps.
       Ao final: S_T / S_0 = exp(log_price[i])
    */
    float log_price[MAX_ASSETS];
    for (int i = 0; i < n_assets; i++) {
        log_price[i] = 0.0f;
    }

    /* ── 5. LOOP TEMPORAL ── o coração do GBM ───────────────────────────── */
    float Z[MAX_ASSETS];    // choques independentes N(0,1)
    float eps[MAX_ASSETS];  // choques correlacionados via Cholesky

    for (int t = 0; t < n_steps; t++) {

        /* 5a. Gera Z[i] ~ N(0,1) independentes */
        for (int i = 0; i < n_assets; i++) {
            Z[i] = curand_normal(&rng_state);
        }

        /* 5b. Correlaciona: eps = L @ Z  (multiplicação matriz-vetor)
               L é triangular inferior → só percorremos j <= i          */
        for (int i = 0; i < n_assets; i++) {
            float dot = 0.0f;
            for (int j = 0; j <= i; j++) {
                dot += L_shared[i * n_assets + j] * Z[j];
            }
            eps[i] = dot;
        }

        /* 5c. Atualiza log-preço via GBM:
               log_price[i] += drift[i] + diffusion[i] * eps[i]         */
        for (int i = 0; i < n_assets; i++) {
            log_price[i] += drift[i] + diffusion[i] * eps[i];
        }
    }

    /* ── 6. CALCULA PREÇOS E RETORNO FINAL ──────────────────────────────── */
    float portfolio_return = 0.0f;
    int base = s * n_assets;   // offset na saída flat [n_sims × n_assets]

    for (int i = 0; i < n_assets; i++) {
        float S_T = expf(log_price[i]);          // S_T / S_0
        out_final_prices[base + i] = S_T;
        portfolio_return += weights[i] * (S_T - 1.0f);
    }

    out_returns[s] = portfolio_return;
}


/* ─────────────────────────────────────────────────────────────────────────
   KERNEL DE AQUECIMENTO (warm-up)
   Executa uma operação trivial para forçar a inicialização do contexto CUDA
   antes do benchmark. Sem isso, a primeira chamada inclui overhead de setup.
   ───────────────────────────────────────────────────────────────────────── */
__global__ void warmup_kernel(float* dummy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) dummy[0] = 1.0f;
}
