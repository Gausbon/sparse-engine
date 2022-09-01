#include "arm_nnsupportfunctions.h"

arm_cmsis_nn_status arm_nn_transpose_bhwc_to_bwhc_q7(
                                            const int32_t dim_b, 
                                            const int32_t dim_h, 
                                            const int32_t dim_w, 
                                            const int32_t dim_c, 
                                            const q7_t *input_section, 
                                            q7_t *output_section)
{
    const q7_t *input_start = input_section;
    q7_t *output_ptr = output_section;
    for (int32_t i_b = 0; i_b < dim_b; ++i_b) {
        for (int32_t i_w = 0; i_w < dim_w; ++i_w) {
            const q7_t *input_ptr = input_start;
            for (int32_t i_h = 0; i_h < dim_h; ++i_h) {
                memcpy(output_ptr, input_ptr, sizeof(q7_t) * dim_c);
                output_ptr += dim_c;
                input_ptr += (dim_c * dim_w);
            }
            input_start += dim_c;
        }
    }
    
    return ARM_CMSIS_NN_SUCCESS;
}


