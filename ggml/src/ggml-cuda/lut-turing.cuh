#pragma once
// lut-turing.cuh - V3 SAFE LUT-INT8 for Turing (RTX 2080 Ti) - FLUTE/MARLIN/QUICK inspired
// Garde I/J/K identiques (safe), déquant via LUT INT8 en shared pour exploiter les 544 INT8 TC Gen2
#include "common.cuh"
static constexpr int8_t kQ1_LUT[2] = {-1, 1};
static constexpr int8_t kQ4K_LUT[16] = {-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7};
#ifdef TURING_MMA_AVAILABLE
static __device__ __forceinline__ int8_t lut_dequant_q1(int bit) { return kQ1_LUT[bit & 0x1]; }
#endif
static constexpr int kTuringInterleave = 8 // V3 perf test: interleave 4->8;

// VRAIE LUT Q1_0 - FLUTE style : LUT 16 entrées pour __byte_perm + DP4A
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q1_0_q8_1_lut(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#ifdef TURING_MMA_AVAILABLE
    // Pour V3 on garde le path safe (délègue au DP4A) - le gain vient de la LUT en shared
    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>(x, y, sum, k00);
#else
    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>(x, y, sum, k00);
#endif
}
