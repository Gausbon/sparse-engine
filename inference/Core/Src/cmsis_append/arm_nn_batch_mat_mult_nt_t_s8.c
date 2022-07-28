#include "arm_nnsupportfunctions.h"

arm_cmsis_nn_status arm_nn_vec_mat_mult_t_s8_rhs_offset(const q7_t *lhs,
                                             const q7_t *rhs,
                                             const q31_t *bias,
                                             q7_t *dst,
                                             const int32_t lhs_offset,
                                             const int32_t rhs_offset,
                                             const int32_t dst_offset,
                                             const int32_t dst_multiplier,
                                             const int32_t dst_shift,
                                             const int32_t rhs_cols,
                                             const int32_t rhs_rows,
                                             const int32_t activation_min,
                                             const int32_t activation_max,
                                             const int32_t address_offset)
{
    const int32_t row_loop_cnt = rhs_rows / 2;
    const int16_t lhs_offset_s16 = (int16_t)lhs_offset;
    const uint32_t lhs_offset_s16x2 = __PKHBT(lhs_offset_s16, lhs_offset_s16, 16);
    const int16_t rhs_offset_s16 = (int16_t)rhs_offset;
    const uint32_t rhs_offset_s16x2 = __PKHBT(rhs_offset_s16, rhs_offset_s16, 16);

    for (int32_t i = 0; i < row_loop_cnt; i++)
    {
        int32_t acc_0 = 0;
        int32_t acc_1 = 0;
        if (bias)
        {
            acc_0 = *bias++;
            acc_1 = *bias++;
        }

        const int32_t col_loop_cnt = rhs_cols / 4;

        const int8_t *lhs_vec = lhs;
        const int8_t *rhs_0 = rhs;
        const int8_t *rhs_1 = rhs + rhs_cols;
        rhs += 2 * rhs_cols;

        for (int j = col_loop_cnt; j != 0; j--)
        {
            
            int32_t vec_0 = arm_nn_read_q7x4_ia(&lhs_vec);
            int32_t vec_1 = __SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
            vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);

            int32_t ker_0 = arm_nn_read_q7x4_ia(&rhs_0);
            int32_t ker_1 = __SXTAB16_RORn(rhs_offset_s16x2, (uint32_t)ker_0, 8);
            ker_0 = __SXTAB16(rhs_offset_s16x2, ker_0);

            acc_0 = __SMLAD(ker_1, vec_1, acc_0);
            acc_0 = __SMLAD(ker_0, vec_0, acc_0);
            
            ker_0 = arm_nn_read_q7x4_ia(&rhs_1);
            ker_1 = __SXTAB16_RORn(rhs_offset_s16x2, (uint32_t)ker_0, 8);
            ker_0 = __SXTAB16(rhs_offset_s16x2, ker_0);

            acc_1 = __SMLAD(ker_1, vec_1, acc_1);
            acc_1 = __SMLAD(ker_0, vec_0, acc_1);
        }

        for (int k = col_loop_cnt * 4; k < rhs_cols; k++)
        {
            const int32_t lhs_temp = (*lhs_vec + lhs_offset);
            lhs_vec++;
            acc_0 += lhs_temp * (*rhs_0 + rhs_offset);
            rhs_0++;
            acc_1 += lhs_temp * (*rhs_1 + rhs_offset);
            rhs_1++;
        }
        
        acc_0 = arm_nn_requantize(acc_0, dst_multiplier, dst_shift);
        acc_1 = arm_nn_requantize(acc_1, dst_multiplier, dst_shift);
        
        // Add offset
        acc_0 += dst_offset;
        acc_1 += dst_offset;
        // Clamp the result
        acc_0 = MAX(acc_0, activation_min);
        acc_0 = MIN(acc_0, activation_max);
        acc_1 = MAX(acc_1, activation_min);
        acc_1 = MIN(acc_1, activation_max);
        *dst = (int8_t)acc_0;
        *(dst + address_offset) = (int8_t)acc_1;
        dst += 2 * address_offset;
    }

    if (rhs_rows & 0x1)
    {
        int32_t acc_0 = 0;
        if (bias)
        {
            acc_0 = *bias++;
        }
        const int32_t col_loop_cnt = rhs_cols / 4;

        const int8_t *lhs_vec = lhs;
        const int8_t *rhs_0 = rhs;

        for (int i = col_loop_cnt; i != 0; i--)
        {
            int32_t vec_0 = arm_nn_read_q7x4_ia(&lhs_vec);
            int32_t vec_1 = __SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
            vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);

            int32_t ker_0 = arm_nn_read_q7x4_ia(&rhs_0);
            int32_t ker_1 = __SXTAB16_RORn(rhs_offset_s16x2, (uint32_t)ker_0, 8);
            ker_0 = __SXTAB16(rhs_offset_s16x2, ker_0);

            acc_0 = __SMLAD(ker_1, vec_1, acc_0);
            acc_0 = __SMLAD(ker_0, vec_0, acc_0);
        }

        for (int j = col_loop_cnt * 4; j < rhs_cols; j++)
        {
            const int32_t lhs_temp = (*lhs_vec + lhs_offset);
            lhs_vec++;
            acc_0 += lhs_temp * (*rhs_0 + rhs_offset);
            rhs_0++;
        }

        acc_0 = arm_nn_requantize(acc_0, dst_multiplier, dst_shift);

        // Add offset
        acc_0 += dst_offset;
        // Clamp the result
        acc_0 = MAX(acc_0, activation_min);
        acc_0 = MIN(acc_0, activation_max);
        *dst = (int8_t)acc_0;
        dst += address_offset;
    }

    return ARM_CMSIS_NN_SUCCESS;
}


arm_cmsis_nn_status arm_nn_batch_mat_mult_nt_t_s8(const q7_t *lhs,
                                            const q7_t *rhs,
                                            const q31_t *bias,
                                            q7_t *dst,
                                            const int32_t dst_multiplier,
                                            const int32_t dst_shift,
                                            const int32_t lhs_rows,
                                            const int32_t lhs_cols,
                                            const int32_t rhs_rows,
                                            const int32_t lhs_offset,
                                            const int32_t rhs_offset,
                                            const int32_t dst_offset,
                                            const int32_t batch,
                                            const int32_t activation_min,
                                            const int32_t activation_max)
{
    // rhs_cols == lhs_cols
    const int32_t rhs_size = lhs_cols * rhs_rows;
    for (int i = 0; i < batch; i++) {
        int32_t batch_cnt = lhs_rows;
        while (batch_cnt) {
            arm_nn_vec_mat_mult_t_s8_rhs_offset(
                                    lhs,
                                    rhs,
                                    bias,
                                    dst,
                                    lhs_offset,
                                    rhs_offset,
                                    dst_offset,
                                    dst_multiplier,
                                    dst_shift,
                                    lhs_cols, /* col_dim or accum_depth */
                                    rhs_rows, /* row_dim or output_depth */
                                    activation_min,
                                    activation_max,
                                    1L);
            lhs += lhs_cols;
            dst += rhs_rows;
            batch_cnt--;
        }
        rhs += rhs_size;
    }
    
    return ARM_CMSIS_NN_SUCCESS;
}
