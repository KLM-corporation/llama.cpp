#pragma once
// lut-turing.cuh - V3 SAFE LUT-INT8 for Turing (RTX 2080 Ti) - FLUTE/MARLIN/QUICK inspired
// Garde I/J/K identiques (safe), déquant via LUT INT8 en shared pour exploiter les 544 INT8 TC Gen2
#include "common.cuh"
static constexpr int8_t kQ1_LUT[2] = {-1, 1};
static constexpr int8_t kQ4K_LUT[16] = {-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7};
#ifdef TURING_MMA_AVAILABLE
static __device__ __forceinline__ int8_t lut_dequant_q1(int bit) { return kQ1_LUT[bit & 0x1]; }
#endif
static constexpr int kTuringInterleave = 4;
