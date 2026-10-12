#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// GPU part of a split MUL_MAT_ID (ggml_mul_mat_id_set_split) at 2-4 tokens, expert-major: experts[r] (-1: none) for
// r < n_experts, slots[r*n_tokens + t] = slot of token t for that expert or -1. Each expert's weights are read once for
// all its tokens (host weights over PCIe). Writes only those slots: dst must be zeroed first.
bool ggml_cuda_mul_mat_vec_q_moe_xm_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_mul_mat_vec_q_moe_xm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
    const int32_t * experts, const int32_t * slots, int n_experts, ggml_tensor * dst);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
