#include "arm_nnsupportfunctions.h"

arm_cmsis_nn_status arm_nn_batch_mat_mult_s8( const cmsis_nn_context *ctx,
                                            const q7_t *lhs,
                                            const q7_t *rhs,
                                            const q31_t *bias,
                                            q7_t *dst,
                                            const int32_t dst_multiplier,
                                            const int32_t dst_shift,
                                            const int32_t lhs_rows,
                                            const int32_t lhs_cols,
                                            const int32_t rhs_cols,
                                            const int32_t lhs_offset,
                                            const int32_t rhs_offset,
                                            const int32_t dst_offset,
                                            const int32_t batch,
                                            const int32_t activation_min,
                                            const int32_t activation_max)
{
    int32_t *buffer = (int32_t*) ctx->buf;
    
    const int32_t lhs_offset_15x2 = __PKHBT(lhs_offset, lhs_offset, 16);
    const int32_t rhs_offset_15x2 = __PKHBT(rhs_offset, rhs_offset, 16);
    int32_t i_batch, i_lhs_row, i_lhs_col, i_rhs_col;

    for(i_batch = 0; i_batch < batch; ++i_batch) {
        const q7_t *lhs_start = &lhs[i_batch * lhs_rows * lhs_cols];
        const q7_t *rhs_start = &rhs[i_batch * lhs_rows * rhs_cols];
        q7_t *dst_start = &dst[i_batch * lhs_rows * rhs_cols];
        const q7_t *lhs_ptr = lhs_start;
        q7_t *dst_ptr = dst_start;

        for (i_lhs_row = 0;i_lhs_row < lhs_rows; ++i_lhs_row) {
            memset(buffer, 0, sizeof(q31_t) * rhs_cols);
            const q7_t *rhs_ptr = rhs_start;

            for (i_lhs_col = 0;i_lhs_col < lhs_cols/2; ++i_lhs_col) {
                q7_t lhs_val_0 = *(lhs_ptr++);
                q7_t lhs_val_1 = *(lhs_ptr++);
                q31_t lhs_15x2 = __PKHBT(lhs_val_0, lhs_val_1, 16);
                lhs_15x2 = __QADD(lhs_15x2, lhs_offset_15x2);

                const q7_t *rhs_head_0 = rhs_ptr;
                const q7_t *rhs_head_1 = rhs_head_0 + rhs_cols;

                buffer = (int32_t*) ctx->buf;

                for (i_rhs_col = 0;i_rhs_col < rhs_cols; ++i_rhs_col) {
                    q7_t rhs_val_0 = *(rhs_head_0++);
                    q7_t rhs_val_1 = *(rhs_head_1++);
                    q31_t rhs_15x2 = __PKHBT(rhs_val_0, rhs_val_1, 16);
                    rhs_15x2 = __QADD(rhs_15x2, rhs_offset_15x2);

                    (*buffer) = __SMLAD(lhs_15x2, rhs_15x2, (*buffer));
                    buffer++;
                }
                
                rhs_ptr += (2 * rhs_cols);
            }
            
            if (lhs_cols & 1) {
                // remain one row
                q7_t lhs_val = lhs_offset + *(lhs_ptr++);
                const q7_t *rhs_head = rhs_ptr;
                buffer = (int32_t*) ctx->buf;

                for (i_rhs_col = 0;i_rhs_col < rhs_cols; ++i_rhs_col) {
                    q7_t rhs_val = rhs_offset + *(rhs_head++);
                    (*buffer) = __QADD((*buffer), (rhs_val * lhs_val));
                    buffer++;
                }
            }
            // output
            if (bias) {
                int32_t cur_bias = bias[i_rhs_col];
                for (i_rhs_col = 0;i_rhs_col < rhs_cols; ++i_rhs_col) {
                    (*buffer) = __QADD((*buffer), cur_bias);
                }
            }
            
            for (i_rhs_col = 0;i_rhs_col < rhs_cols; ++i_rhs_col) {
                q31_t requant = arm_nn_requantize((*buffer), dst_multiplier,
                        dst_shift);
                requant += dst_offset;
                requant = MAX(requant, activation_min);
                requant = MIN(requant, activation_max);
                *(dst_ptr++) = (q7_t) requant;
            }  
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNBasicMath group
 */
