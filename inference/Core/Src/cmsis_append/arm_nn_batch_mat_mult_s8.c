#include "arm_nnsupportfunctions.h"
arm_cmsis_nn_status arm_nn_vec_mat_mult_s8_rhs_offset(
                                             const q7_t *lhs,
                                             const q7_t *rhs,
                                             const q31_t *bias,
                                             int32_t *buffer,
                                             q7_t *dst,
                                             const int32_t lhs_offset,
                                             const int32_t rhs_offset,
                                             const int32_t dst_offset,
                                             const int32_t dst_multiplier,
                                             const int32_t dst_shift,
                                             const int32_t lhs_cols,
                                             const int32_t rhs_cols,
                                             const int32_t activation_min,
                                             const int32_t activation_max,
                                             const int32_t address_offset)
{
    // lhs_cols == rhs_rows
    const int32_t row_col_cnt = lhs_cols / 4;

    const int16_t lhs_offset_s16 = (int16_t)lhs_offset;
    const uint32_t lhs_offset_s16x2 = __PKHBT(lhs_offset_s16, lhs_offset_s16, 16);
    const int16_t rhs_offset_s16 = (int16_t)rhs_offset;
    const uint32_t rhs_offset_s16x2 = __PKHBT(rhs_offset_s16, rhs_offset_s16, 16);

    memset(buffer, 0, sizeof(int32_t) * rhs_cols);
    const int8_t *lhs_vec = lhs;
    
    for (int32_t i = row_col_cnt; i > 0; i--)
    {
        if (bias)
        {
            for (int j = 0; j < rhs_cols; j++) {
                buffer[j] += bias[j];
            }
        }
        const int8_t *rhs_0 = rhs;
        const int8_t *rhs_1 = rhs_0 + rhs_cols;
        const int8_t *rhs_2 = rhs_1 + rhs_cols;
        const int8_t *rhs_3 = rhs_2 + rhs_cols;
        rhs += 4 * rhs_cols;
        
        int32_t vec_0 = arm_nn_read_q7x4_ia(&lhs_vec);
        int32_t vec_1 = __SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
        vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);
        
        for (int j = 0; j < rhs_cols; j++) {
            int32_t ker_0 = __PKHBT(*(rhs_0++), *(rhs_2++), 16);
            int32_t ker_1 = __PKHBT(*(rhs_1++), *(rhs_3++), 16);
            ker_0 = __SXTAB16(rhs_offset_s16x2, ker_0);
            ker_1 = __SXTAB16(rhs_offset_s16x2, ker_1);
            
            buffer[j] = __SMLAD(ker_0, vec_0, buffer[j]);
            buffer[j] = __SMLAD(ker_1, vec_1, buffer[j]);
        }
    }
    
    int32_t row_cnt = 4 * row_col_cnt;
    if (lhs_cols - row_cnt >= 2) {
        const int8_t *rhs_0 = rhs;
        const int8_t *rhs_1 = rhs_0 + rhs_cols;
        rhs += 2 * rhs_cols;

        int32_t vec_0 = __PKHBT(*(lhs_vec++), *(lhs_vec++), 16);
        vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);
        for (int j = 0; j < rhs_cols; j++) {
            int32_t ker_0 = __PKHBT(*(rhs_0++), *(rhs_1++), 16);
            ker_0 = __SXTAB16(rhs_offset_s16x2, ker_0);
            
            buffer[j] = __SMLAD(ker_0, vec_0, buffer[j]);
        }
        row_cnt += 2;
    }

    if (lhs_cols > row_cnt) {
        int32_t lhs_val = __QADD(*(lhs_vec++), lhs_offset);

        for (int j = 0; j < rhs_cols; j++) {
            int32_t rhs_val = __QADD(*(rhs++), rhs_offset);

            buffer[j] += (lhs_val * rhs_val);
        }
        row_cnt ++;
    }

    for (int i = 0; i < rhs_cols; i++) {
        int32_t requant = arm_nn_requantize(buffer[i], dst_multiplier, dst_shift);

        // Add offset
        requant += dst_offset;
        // Clamp the result
        requant = MAX(requant, activation_min);
        requant = MIN(requant, activation_max);
        *dst = (int8_t)requant;
        dst += address_offset;
    }


    return ARM_CMSIS_NN_SUCCESS;
}


arm_cmsis_nn_status arm_nn_batch_mat_mult_s8(const cmsis_nn_context *ctx,
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
    // rhs_rows == lhs_cols
    int32_t *buffer = (int32_t*) ctx->buf;
    const int32_t rhs_size = lhs_cols * rhs_cols;

    for (int i = 0; i < batch; i++) {
        int32_t batch_cnt = lhs_rows;
        while (batch_cnt) {
            arm_nn_vec_mat_mult_s8_rhs_offset(lhs,
                                    rhs,
                                    bias,
                                    buffer,
                                    dst,
                                    lhs_offset,
                                    rhs_offset,
                                    dst_offset,
                                    dst_multiplier,
                                    dst_shift,
                                    lhs_cols, /* col_dim or accum_depth */
                                    rhs_cols, /* row_dim or output_depth */
                                    activation_min,
                                    activation_max,
                                    1L);
            lhs += lhs_cols;
            dst += rhs_cols;
            batch_cnt--;
        }
        rhs += rhs_size;
    }
    
    return ARM_CMSIS_NN_SUCCESS;
}
