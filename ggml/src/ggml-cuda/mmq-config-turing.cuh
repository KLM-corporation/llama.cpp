#pragma once
// mmq-config-turing.cuh - Config MMQ optimisée pour RTX 2080 Ti (TU102 / CC 7.5)
// Turing: 68 SM, 4352 cores, 64 cores/SM, INT8 Tensor Cores (dp4a + ldmatrix/mma.m8n8), 64KB shared/SM, 11GB VRAM
// Optimisé pour : Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_NL - les quants les plus utilisés en 2024-2026
// Ratio: on réduit I de 128->64 et occupancy 1->2/3 pour mieux remplir les 68 SM de Turing vs Ampere (108 SM)
// Et on force MMA (TURING_MMA_AVAILABLE) dès que possible car INT8 TC > DP4A sur Turing si batch > 16

static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_turing(ggml_type type, int J, bool fallback) {
    // Macro helper identique à ampere/pascal : CASE(type, nthreads, occupancy, I, J, sram, K, stream_k, fallback)
    // Turing sweet spot : nthreads=128 pour J<=32, 256 pour J>32. occupancy=2 ou 3 (vs 1 sur Ampere)
    // I=64 donne meilleure occupancy sur Turing (registres), I=128 utilisé seulement pour gros J
    // K_vram = MMQ_ITER_K (256) sauf pour très gros J où 128 améliore cache L2 (TU102 a 6MB L2)

    // --- Q4_0 : le GGUF le plus courant ---
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_0, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_0, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_0, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_0, 128, 3,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_0, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_0, 256, 3,  64,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_0, 256, 2,  64,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_0, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);

    // --- Q4_1 ---
    CASE(GGML_TYPE_Q4_1, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_1, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_1, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_1, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_1, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_1, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_1, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_1, 128, 2,  64,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_1, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_1, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_1, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);

    // --- Q5_0 / Q5_1 (similaire Q4) ---
    CASE(GGML_TYPE_Q5_0, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_0, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_0, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_0, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_0, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_0, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q5_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);

    CASE(GGML_TYPE_Q5_1, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_1, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_1, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_1, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_1, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);

    // --- Q8_0 : fp32 fallback DP4A (Turing n'a pas FP16 TC ultra rapide, donc on garde DP4A pour batch petit) ---
    CASE(GGML_TYPE_Q8_0, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q8_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q8_0, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q8_0, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q8_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);

    // --- Q2_K (2-bit, très compressé, utilise beaucoup de scales -> I plus petit) ---
    CASE(GGML_TYPE_Q2_K, 128, 2,  32,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q2_K, 128, 3,  32,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q2_K, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q2_K, 256, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q2_K, 256, 2,  64, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, true);

    // --- Q3_K ---
    CASE(GGML_TYPE_Q3_K, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q3_K, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q3_K, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q3_K, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q3_K, 256, 2,  64, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);

    // --- Q4_K (LE quant le plus utilisé avec Q6_K) ---
    CASE(GGML_TYPE_Q4_K, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 128, 3,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 128, 2,  64,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);

    // --- Q5_K ---
    CASE(GGML_TYPE_Q5_K, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_K, 128, 3,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_K, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_K, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q5_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);

    // --- Q6_K (LE sweet spot qualité/perf pour 2080 Ti) ---
    CASE(GGML_TYPE_Q6_K, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q6_K, 128, 3,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q6_K, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q6_K, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q6_K, 256, 2,  64, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q6_K, 128, 2,  64,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q6_K, 128, 2,  64,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q6_K, 128, 3,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q6_K, 256, 3,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false);
    CASE(GGML_TYPE_Q6_K, 256, 2,  64, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false);

    // --- IQ2_XXS / IQ2_XS / IQ2_S (2-bit améliorés) ---
    CASE(GGML_TYPE_IQ2_XXS, 128, 2, 32,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_IQ2_XXS, 128, 3, 32,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_IQ2_XXS, 256, 3, 64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_IQ2_XS,  128, 2, 32,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_IQ2_XS,  256, 3, 64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_IQ2_S,   128, 3, 32,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_IQ2_S,   256, 2, 64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);

    // --- IQ4_NL / IQ4_XS (4-bit near-lossless, très populaire) ---
    CASE(GGML_TYPE_IQ4_NL, 128, 2, 64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_IQ4_NL, 128, 3, 64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_IQ4_NL, 256, 3, 64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_IQ4_NL, 256, 2,128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_IQ4_XS, 128, 3, 64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_IQ4_XS, 256, 3, 64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);

    // fallback
    return {type, 128, 2, 32, 32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback};
}
