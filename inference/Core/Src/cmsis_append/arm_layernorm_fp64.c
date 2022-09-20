#include "func.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

arm_status arm_layernorm_fp64 (const cmsis_nn_context *ctx,
                           const cmsis_nn_layernorm_params *layernorm_params,
                           const cmsis_nn_per_tensor_quant_params *quant_params,
                           const int32_t dim_b,
                           const int32_t dim_c,
                           const q7_t *weight,
                           const q31_t *bias,
                           const double *input_data,
                           double *output_data)
{
    (void) ctx;
    
    const double in_offset = (double)layernorm_params->input_offset;
    const double out_offset = (double)layernorm_params->output_offset;
    
    double mult = (double)quant_params->multiplier;
    double shift = (double)quant_params->shift;
    mult = mult / (pow(2, shift));

    int32_t i_dim_b, i_dim_c;
    double avg, var, sum, requant;
    
    const double *input_ptr = input_data;
    const double *input_start = input_data;
    double *output_start = output_data;
    double *output_ptr = output_start;

    for (i_dim_b = 0; i_dim_b < dim_b; ++i_dim_b) {
        input_ptr = input_start;
        output_ptr = output_start;
        // calculate the avg
        sum = 0;
        for (i_dim_c = 0; i_dim_c < dim_c; ++i_dim_c) {
            sum = sum + ((*input_ptr + in_offset) / mult);
            input_ptr++;
        }
        avg = sum / dim_c;
        
        input_ptr = input_start;

        var = 0;

        // calculate the std
        for (i_dim_c = 0; i_dim_c < dim_c; ++i_dim_c) {
            var += (((*input_ptr + in_offset) / mult) - avg) * (((*input_ptr + in_offset) / mult) - avg);
            input_ptr++;
        }

        var = var / dim_c;
        var = sqrt(var);
        if (var == 0) {
            var = 1;
        }
        
        input_ptr = input_start;

        // norm
        for (i_dim_c = 0; i_dim_c < dim_c; ++i_dim_c) {
            requant = (((*input_ptr + in_offset) / mult) - avg) / var;
            input_ptr++;
            if (weight) {
                requant *= weight[i_dim_c];
                if (bias) {
                    requant += bias[i_dim_c];
                }
            }
            requant = (requant * mult);
            requant = requant + out_offset;
            *(output_ptr++) = requant;
        }
        input_start += dim_c;
        output_start += dim_c;
    }
    return ARM_MATH_SUCCESS;
}
