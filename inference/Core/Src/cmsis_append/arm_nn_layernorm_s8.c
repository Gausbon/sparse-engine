#include "func.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"


// ref: http://www.azillionmonkeys.com/qed/sqroot.html
static unsigned fred_sqrt(unsigned long x) {
    static const unsigned char sqq_table[] = {
        0,  16,  22,  27,  32,  35,  39,  42,  45,  48,  50,  53,  55,  57,
        59,  61,  64,  65,  67,  69,  71,  73,  75,  76,  78,  80,  81,  83,
        84,  86,  87,  89,  90,  91,  93,  94,  96,  97,  98,  99, 101, 102,
        103, 104, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118,
        119, 120, 121, 122, 123, 124, 125, 126, 128, 128, 129, 130, 131, 132,
        133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 144, 145,
        146, 147, 148, 149, 150, 150, 151, 152, 153, 154, 155, 155, 156, 157,
        158, 159, 160, 160, 161, 162, 163, 163, 164, 165, 166, 167, 167, 168,
        169, 170, 170, 171, 172, 173, 173, 174, 175, 176, 176, 177, 178, 178,
        179, 180, 181, 181, 182, 183, 183, 184, 185, 185, 186, 187, 187, 188,
        189, 189, 190, 191, 192, 192, 193, 193, 194, 195, 195, 196, 197, 197,
        198, 199, 199, 200, 201, 201, 202, 203, 203, 204, 204, 205, 206, 206,
        207, 208, 208, 209, 209, 210, 211, 211, 212, 212, 213, 214, 214, 215,
        215, 216, 217, 217, 218, 218, 219, 219, 220, 221, 221, 222, 222, 223,
        224, 224, 225, 225, 226, 226, 227, 227, 228, 229, 229, 230, 230, 231,
        231, 232, 232, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238,
        239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246,
        246, 247, 247, 248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253,
        253, 254, 254, 255
    };

    unsigned long xn;
    if (x >= 0x10000)
        if (x >= 0x1000000)
            if (x >= 0x10000000)
                if (x >= 0x40000000) {
                    if (x >= 65535UL*65535UL)
                        return 65535;
                    xn = sqq_table[x>>24] << 8;
                } else
                    xn = sqq_table[x>>22] << 7;
            else
                if (x >= 0x4000000)
                    xn = sqq_table[x>>20] << 6;
                else
                    xn = sqq_table[x>>18] << 5;
        else {
            if (x >= 0x100000)
                if (x >= 0x400000)
                    xn = sqq_table[x>>16] << 4;
                else
                    xn = sqq_table[x>>14] << 3;
            else
                if (x >= 0x40000)
                    xn = sqq_table[x>>12] << 2;
                else
                    xn = sqq_table[x>>10] << 1;

            goto nr1;
        }
    else
        if (x >= 0x100) {
            if (x >= 0x1000)
                if (x >= 0x4000)
                    xn = (sqq_table[x>>8] >> 0) + 1;
                else
                    xn = (sqq_table[x>>6] >> 1) + 1;
            else
                if (x >= 0x400)
                    xn = (sqq_table[x>>4] >> 2) + 1;
                else
                    xn = (sqq_table[x>>2] >> 3) + 1;
            goto adj;
        } else
            return sqq_table[x] >> 4;
/* Run two iterations of the standard convergence formula */
    xn = (xn + 1 + x / xn) / 2;
nr1:
    xn = (xn + 1 + x / xn) / 2;
adj:
    if (xn * xn > x) /* Correct rounding if necessary */
        xn--;
    return xn;
}


arm_status arm_nn_layernorm_s8 (const cmsis_nn_context *ctx,
                           const cmsis_nn_layernorm_params *layernorm_params,
                           const cmsis_nn_per_tensor_quant_params *quant_params,
                           const int32_t dim_b,
                           const int32_t dim_c,
                           const q7_t *weight,
                           const q31_t *bias,
                           const q7_t *input_data,
                           q7_t *output_data)
{
    const int32_t in_offset = layernorm_params->input_offset;
    const int32_t out_offset = layernorm_params->output_offset;
    const int32_t act_max = layernorm_params->activation.max;
    const int32_t act_min = layernorm_params->activation.min;
    
    int32_t mult = quant_params->multiplier;
    int32_t shift = quant_params->shift;

    int32_t i_dim_b, i_dim_c;
    int32_t double_flag;
    q31_t diff, diff_last, diff_15x2;
    q31_t avg, var_sqrt, factor, requant, var_int, avg_offset;
    int64_t var, sum;
    
    const q7_t *input_ptr = input_data;
    const q7_t *input_start = input_data;
    const int32_t offset_sum = in_offset * dim_c;
    q7_t *output_start = output_data;
    q7_t *output_ptr = output_start;

    for (i_dim_b = 0; i_dim_b < dim_b; ++i_dim_b) {
        input_ptr = input_start;
        output_ptr = output_start;
        // calculate the avg
        sum = 0;
        double_flag = 0;
        for (i_dim_c = 0; i_dim_c < dim_c; ++i_dim_c) {
            sum = sum + *(input_ptr++);
        }
        sum += offset_sum;
        avg = (sum + (dim_c / 2)) / dim_c;
        // clamp to int16
        avg = MAX(avg, -32768);
        avg = MIN(avg, 32767);
        avg_offset = avg - in_offset;
        input_ptr = input_start;

        var = 0;
        // calculate the std
        for (i_dim_c = 0; i_dim_c < dim_c; ++i_dim_c) {
            diff = __QSUB(*(input_ptr++), avg_offset);

            if (diff > 32767 || diff < -32768) {
                // the difference is out of int16 bound
                var += (diff * diff);
            } else {
                if (double_flag == 1) {
                    diff_15x2 = __PKHBT(diff, diff_last, 16);
                    var = __SMLALD(diff_15x2, diff_15x2, var);
                } else {
                    diff_last = diff;
                }
                double_flag = 1 - double_flag;
            }
        }

        if (double_flag == 1) {
            var += (diff_last * diff_last);
        }

        // var = MAX(var, -2147483648);
        var_int = MIN(var, 2147483647);
        var_int = var_int / dim_c;
        var_sqrt = (q31_t)fred_sqrt(var_int);
        if (var_sqrt == 0) {
            var_sqrt = 1;
        }
        
        factor = 128 / var_sqrt;
        input_ptr = input_start;

        // norm
        for (i_dim_c = 0; i_dim_c < dim_c; ++i_dim_c) {
            requant = __QSUB(*(input_ptr++), avg_offset);
            requant = requant * factor;
            requant = MAX(requant, -32768);
            requant = MIN(requant, 32767);
            if (weight) {
                requant *= weight[i_dim_c];
                if (bias) {
                    requant += bias[i_dim_c];
                }
            }
            requant = MAX(requant, -2147483648);
            requant = MIN(requant, 2147483647);
            requant = arm_nn_requantize(requant, mult, shift);
            requant = requant + out_offset;
            requant = MAX(requant, act_min);
            requant = MIN(requant, act_max);
            *(output_ptr++) = (q7_t) requant;
        }
        input_start += dim_c;
        output_start += dim_c;
    }
    return ARM_MATH_SUCCESS;
}
