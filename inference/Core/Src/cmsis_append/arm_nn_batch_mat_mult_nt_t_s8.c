#include "arm_nnsupportfunctions.h"

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
    int32_t i_batch = 0;
    for(; i_batch < batch; ++i_batch) {
        const q7_t *lhs_start = &lhs[i_batch * lhs_rows * lhs_cols];
        const q7_t *rhs_start = &rhs[i_batch * rhs_rows * lhs_cols];
        q7_t *dst_start = &dst[i_batch * lhs_rows * rhs_rows];

        const int32_t off0 = lhs_cols - 4;

        for (int32_t rhs_rows_idx = 0; rhs_rows_idx <= (rhs_rows - 2); rhs_rows_idx += 2)
        {
            const q7_t *lhs_ptr = &lhs_start[0];
            q7_t *dst_ptr = &dst_start[0];

            q31_t lhs_offset_contribution0 = 0;
            q31_t lhs_offset_contribution1 = 0;

            for (int32_t x = 0; x < lhs_cols; ++x)
            {
                lhs_offset_contribution0 += rhs_start[x];
                lhs_offset_contribution1 += rhs_start[x + lhs_cols];
            }

            lhs_offset_contribution0 *= lhs_offset;
            lhs_offset_contribution1 *= lhs_offset;
            if (bias)
            {
                lhs_offset_contribution0 += bias[rhs_rows_idx];
                lhs_offset_contribution1 += bias[rhs_rows_idx + 1];
            }

            int32_t lhs_rows_idx = lhs_rows >> 1;

            while (lhs_rows_idx)
            {
                const q7_t *rhs_ptr = &rhs_start[0];

                q31_t res00 = lhs_offset_contribution0;
                q31_t res01 = lhs_offset_contribution1;
                q31_t res10 = lhs_offset_contribution0;
                q31_t res11 = lhs_offset_contribution1;

                int32_t lhs_cols_idx = 0;

                q31_t val0, val1, val2, val3, val4, val5;

                for (; lhs_cols_idx <= (lhs_cols - 16); lhs_cols_idx += 16)
                {
                    val1 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
                    val2 = __SXTB16(val1);
                    val0 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
                    val3 = __SXTB16(val0);
                    val4 = arm_nn_read_q7x4((const q7_t *)&rhs_ptr[off0]);
                    val1 = __SXTB16_RORn(val1, 8);
                    val0 = __SXTB16_RORn(val0, 8);

                    // 4 x MAC res00, res01
                    res00 = __SMLAD(val3, val2, res00);
                    val5 = __SXTB16(val4);
                    res00 = __SMLAD(val0, val1, res00);
                    val4 = __SXTB16_RORn(val4, 8);
                    res01 = __SMLAD(val3, val5, res01);
                    res01 = __SMLAD(val0, val4, res01);

                    // 4 x MAC res10, res11
                    val0 = arm_nn_read_q7x4((const q7_t *)&lhs_ptr[off0]);
                    val3 = __SXTB16(val0);
                    val0 = __SXTB16_RORn(val0, 8);
                    res10 = __SMLAD(val3, val2, res10);
                    res11 = __SMLAD(val3, val5, res11);
                    res10 = __SMLAD(val0, val1, res10);
                    val1 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
                    res11 = __SMLAD(val0, val4, res11);

                    val4 = arm_nn_read_q7x4((const q7_t *)&rhs_ptr[off0]);
                    val2 = __SXTB16(val1);
                    val0 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
                    val3 = __SXTB16(val0);
                    val1 = __SXTB16_RORn(val1, 8);
                    val0 = __SXTB16_RORn(val0, 8);

                    // 4 x MAC res00, res01
                    res00 = __SMLAD(val3, val2, res00);
                    val5 = __SXTB16(val4);
                    res00 = __SMLAD(val0, val1, res00);
                    val4 = __SXTB16_RORn(val4, 8);
                    res01 = __SMLAD(val3, val5, res01);
                    res01 = __SMLAD(val0, val4, res01);

                    // 4 x MAC res10, res11
                    val0 = arm_nn_read_q7x4((const q7_t *)&lhs_ptr[off0]);
                    val3 = __SXTB16(val0);
                    val0 = __SXTB16_RORn(val0, 8);
                    res10 = __SMLAD(val3, val2, res10);
                    res11 = __SMLAD(val3, val5, res11);
                    res10 = __SMLAD(val0, val1, res10);
                    val1 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
                    res11 = __SMLAD(val0, val4, res11);

                    val4 = arm_nn_read_q7x4((const q7_t *)&rhs_ptr[off0]);
                    val2 = __SXTB16(val1);
                    val0 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
                    val3 = __SXTB16(val0);
                    val1 = __SXTB16_RORn(val1, 8);
                    val0 = __SXTB16_RORn(val0, 8);

                    // 4 x MAC res00, res01
                    res00 = __SMLAD(val3, val2, res00);
                    val5 = __SXTB16(val4);
                    res00 = __SMLAD(val0, val1, res00);
                    val4 = __SXTB16_RORn(val4, 8);
                    res01 = __SMLAD(val3, val5, res01);
                    res01 = __SMLAD(val0, val4, res01);

                    // 4 x MAC res10, res11
                    val0 = arm_nn_read_q7x4((const q7_t *)&lhs_ptr[off0]);
                    val3 = __SXTB16(val0);
                    val0 = __SXTB16_RORn(val0, 8);
                    res10 = __SMLAD(val3, val2, res10);
                    res11 = __SMLAD(val3, val5, res11);
                    res10 = __SMLAD(val0, val1, res10);
                    val1 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
                    res11 = __SMLAD(val0, val4, res11);

                    val4 = arm_nn_read_q7x4((const q7_t *)&rhs_ptr[off0]);
                    val2 = __SXTB16(val1);
                    val0 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
                    val3 = __SXTB16(val0);
                    val1 = __SXTB16_RORn(val1, 8);
                    val0 = __SXTB16_RORn(val0, 8);

                    // 4 x MAC res00, res01
                    res00 = __SMLAD(val3, val2, res00);
                    val5 = __SXTB16(val4);
                    res00 = __SMLAD(val0, val1, res00);
                    val4 = __SXTB16_RORn(val4, 8);
                    res01 = __SMLAD(val3, val5, res01);
                    res01 = __SMLAD(val0, val4, res01);

                    // 4 x MAC res10, res11
                    val0 = arm_nn_read_q7x4((const q7_t *)&lhs_ptr[off0]);
                    val3 = __SXTB16(val0);
                    val0 = __SXTB16_RORn(val0, 8);
                    res10 = __SMLAD(val3, val2, res10);
                    res11 = __SMLAD(val3, val5, res11);
                    res10 = __SMLAD(val0, val1, res10);
                    res11 = __SMLAD(val0, val4, res11);
                }

                for (; lhs_cols_idx < lhs_cols; ++lhs_cols_idx)
                {
                    q7_t rhs_value0 = rhs_ptr[0];
                    q7_t rhs_value1 = rhs_ptr[lhs_cols];
                    q7_t lhs_value = lhs_ptr[0];

                    res00 += lhs_value * rhs_value0;
                    res01 += lhs_value * rhs_value1;

                    lhs_value = lhs_ptr[lhs_cols];
                    res10 += lhs_value * rhs_value0;
                    res11 += lhs_value * rhs_value1;

                    ++rhs_ptr;
                    ++lhs_ptr;
                }

                // Quantize down
                res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);
                res01 = arm_nn_requantize(res01, dst_multiplier, dst_shift);
                res10 = arm_nn_requantize(res10, dst_multiplier, dst_shift);
                res11 = arm_nn_requantize(res11, dst_multiplier, dst_shift);

                // Add offset
                res00 += dst_offset;
                res01 += dst_offset;
                res10 += dst_offset;
                res11 += dst_offset;

                // Clamp the result
                res00 = MAX(res00, activation_min);
                res00 = MIN(res00, activation_max);
                res01 = MAX(res01, activation_min);
                res01 = MIN(res01, activation_max);
                res10 = MAX(res10, activation_min);
                res10 = MIN(res10, activation_max);
                res11 = MAX(res11, activation_min);
                res11 = MIN(res11, activation_max);

                dst_ptr[0] = (q7_t)res00;
                dst_ptr[1] = (q7_t)res01;
                dst_ptr += rhs_rows;
                dst_ptr[0] = (q7_t)res10;
                dst_ptr[1] = (q7_t)res11;
                dst_ptr += rhs_rows;

                lhs_ptr += lhs_cols;

                lhs_rows_idx--;
            }

            // Left-over rows
            if (lhs_rows % 2)
            {
                const q7_t *rhs_ptr = &rhs_start[0];

                q31_t res00 = lhs_offset_contribution0;
                q31_t res01 = lhs_offset_contribution1;

                int32_t lhs_cols_idx = 0;

                q31_t val0, val1, val2, val3, val4, val5;
                for (; lhs_cols_idx <= (lhs_cols - 16); lhs_cols_idx += 16)
                {
                    val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
                    val1 = arm_nn_read_q7x4((const q7_t *)&rhs_ptr[off0]);
                    val2 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
                    val3 = __SXTB16(val0);
                    val5 = __SXTB16(val2);
                    val4 = __SXTB16(val1);
                    val0 = __SXTB16_RORn(val0, 8);
                    val2 = __SXTB16_RORn(val2, 8);
                    val1 = __SXTB16_RORn(val1, 8);

                    // 4 x MAC res00, res01
                    res00 = __SMLAD(val5, val3, res00);
                    res00 = __SMLAD(val2, val0, res00);
                    res01 = __SMLAD(val5, val4, res01);
                    res01 = __SMLAD(val2, val1, res01);

                    val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
                    val1 = arm_nn_read_q7x4((const q7_t *)&rhs_ptr[off0]);
                    val2 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
                    val3 = __SXTB16(val0);
                    val5 = __SXTB16(val2);
                    val4 = __SXTB16(val1);
                    val0 = __SXTB16_RORn(val0, 8);
                    val2 = __SXTB16_RORn(val2, 8);
                    val1 = __SXTB16_RORn(val1, 8);

                    // 4 x MAC res00, res01
                    res00 = __SMLAD(val5, val3, res00);
                    res00 = __SMLAD(val2, val0, res00);
                    res01 = __SMLAD(val5, val4, res01);
                    res01 = __SMLAD(val2, val1, res01);

                    val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
                    val1 = arm_nn_read_q7x4((const q7_t *)&rhs_ptr[off0]);
                    val2 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
                    val3 = __SXTB16(val0);
                    val5 = __SXTB16(val2);
                    val4 = __SXTB16(val1);
                    val0 = __SXTB16_RORn(val0, 8);
                    val2 = __SXTB16_RORn(val2, 8);
                    val1 = __SXTB16_RORn(val1, 8);

                    // 4 x MAC res00, res01
                    res00 = __SMLAD(val5, val3, res00);
                    res00 = __SMLAD(val2, val0, res00);
                    res01 = __SMLAD(val5, val4, res01);
                    res01 = __SMLAD(val2, val1, res01);

                    val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
                    val1 = arm_nn_read_q7x4((const q7_t *)&rhs_ptr[off0]);
                    val2 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
                    val3 = __SXTB16(val0);
                    val5 = __SXTB16(val2);
                    val4 = __SXTB16(val1);
                    val0 = __SXTB16_RORn(val0, 8);
                    val2 = __SXTB16_RORn(val2, 8);
                    val1 = __SXTB16_RORn(val1, 8);

                    // 4 x MAC res00, res01
                    res00 = __SMLAD(val5, val3, res00);
                    res00 = __SMLAD(val2, val0, res00);
                    res01 = __SMLAD(val5, val4, res01);
                    res01 = __SMLAD(val2, val1, res01);
                }

                // Left-over accumulations
                for (; lhs_cols_idx < lhs_cols; ++lhs_cols_idx)
                {
                    q7_t rhs_value0 = rhs_ptr[0];
                    q7_t rhs_value1 = rhs_ptr[lhs_cols];
                    q7_t lhs_value = lhs_ptr[0];

                    res00 += lhs_value * rhs_value0;
                    res01 += lhs_value * rhs_value1;

                    ++rhs_ptr;
                    ++lhs_ptr;
                }

                // Quantize down
                res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);
                res01 = arm_nn_requantize(res01, dst_multiplier, dst_shift);

                // Add offset
                res00 += dst_offset;
                res01 += dst_offset;

                // Clamp the result
                res00 = MAX(res00, activation_min);
                res00 = MIN(res00, activation_max);
                res01 = MAX(res01, activation_min);
                res01 = MIN(res01, activation_max);

                dst_ptr[0] = (q7_t)res00;
                dst_ptr[1] = (q7_t)res01;
            }

            rhs += 2 * lhs_cols;
            dst += 2;
        }

        if (rhs_rows % 2)
        {
            const q7_t *lhs_ptr = &lhs_start[0];
            q7_t *dst_ptr = &dst_start[0];

            for (int32_t lhs_rows_idx = 0; lhs_rows_idx < lhs_rows; ++lhs_rows_idx)
            {
                const q7_t *rhs_ptr = &rhs_start[0];
                q31_t res00 = 0;
                if (bias)
                {
                    res00 = bias[rhs_rows - 1];
                }

                for (int32_t lhs_cols_idx = 0; lhs_cols_idx < lhs_cols; ++lhs_cols_idx)
                {
                    q31_t rhs_value = rhs_ptr[0];
                    q31_t lhs_value = lhs_ptr[0] + lhs_offset;

                    res00 += lhs_value * rhs_value;

                    ++rhs_ptr;
                    ++lhs_ptr;
                }

                // Quantize down
                res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);

                // Add offset
                res00 += dst_offset;

                // Clamp the result
                res00 = MAX(res00, activation_min);
                res00 = MIN(res00, activation_max);

                dst_ptr[0] = (q7_t)res00;
                dst_ptr += rhs_rows;
            }
        }
    }
    
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNBasicMath group
 */
