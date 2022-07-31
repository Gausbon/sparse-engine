#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"


static void scale_q31_to_q7_with_quantization (const q31_t *buffer,
                                      q7_t *target,
                                      int32_t length,
                                      const int32_t count,
                                      const int32_t input_offset,
                                      const int32_t output_offset,
                                      const int32_t multiplier,
									  const int32_t shift,
                                      const int act_min,
                                      const int act_max)
{
    const int half_count = count / 2;

    // Prevent static code issue DIVIDE_BY_ZERO.
    if (count == 0)
    {
        return;
    }

    for (int i = 0; i < length; i++)
    {
        int32_t sum = buffer[i] > 0 ? (buffer[i] + half_count) : (buffer[i] - half_count);
        sum = sum / count;
        sum += input_offset;
        sum = arm_nn_requantize(sum, multiplier, shift);
        sum += output_offset;
        sum = MAX(sum, act_min);
        sum = MIN(sum, act_max);

        target[i] = (q7_t)sum;
    }
}


arm_status arm_avgpool_s8_with_quantization (const cmsis_nn_context *ctx,
                          const cmsis_nn_pool_params *pool_params,
                          const cmsis_nn_dims *input_dims,
                          const q7_t *src,
                          const cmsis_nn_dims *filter_dims,
                          const cmsis_nn_dims *output_dims,
                          const int32_t input_offset,
                          const int32_t output_offset,
                          const int32_t multiplier,
                          const int32_t shift,
                          q7_t *dst)
{
    const int32_t input_y = input_dims->h;
    const int32_t input_x = input_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t stride_y = pool_params->stride.h;
    const int32_t stride_x = pool_params->stride.w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t pad_y = pool_params->padding.h;
    const int32_t pad_x = pool_params->padding.w;
    const int32_t act_min = pool_params->activation.min;
    const int32_t act_max = pool_params->activation.max;
    const int32_t ch_src = input_dims->c;

    if (ctx->buf == NULL && arm_avgpool_s8_get_buffer_size(output_dims->w, input_dims->c))
    {
        return ARM_MATH_ARGUMENT_ERROR;
    }
    q31_t *buffer = (q31_t *)ctx->buf;

    /* Run the following code for CPU's with DSP extension
     */
    for (int i_y = 0, idx_y = -pad_y; i_y < output_y; idx_y += stride_y, i_y++)
    {
        for (int i_x = 0, idx_x = -pad_x; i_x < output_x; idx_x += stride_x, i_x++)
        {
            /* Condition for kernel start dimension:
                      (base_idx_<x,y> + kernel_<x,y>_start) >= 0 */
            const int32_t kernel_y_start = MAX(0, -idx_y);
            const int32_t kernel_x_start = MAX(0, -idx_x);

            /* Condition for kernel end dimension:
                   (base_idx_<x,y> + kernel_<x,y>_end) < dim_src_<width,height> */
            const int32_t kernel_y_end = MIN(kernel_y, input_y - idx_y);
            const int32_t kernel_x_end = MIN(kernel_x, input_x - idx_x);

            int count = 0;

            for (int k_y = kernel_y_start; k_y < kernel_y_end; k_y++)
            {
                for (int k_x = kernel_x_start; k_x < kernel_x_end; k_x++)
                {
                    const q7_t *start = src + ch_src * (k_x + idx_x + (k_y + idx_y) * input_x);
                    if (count == 0)
                    {
                        for (int i = 0; i < ch_src; i++)
                        {
                            buffer[i] = start[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < ch_src; i++)
                        {
                            buffer[i] = __QADD(start[i], buffer[i]);
                        }
                    }
                    count++;
                }
            }

            // Prevent static code issue DIVIDE_BY_ZERO.
            if (count == 0)
            {
                return ARM_MATH_ARGUMENT_ERROR;
            }

            scale_q31_to_q7_with_quantization(buffer, dst, ch_src, count, 
					input_offset, output_offset, multiplier, shift, act_min, act_max);
            dst += ch_src;
        }
    }
	return ARM_MATH_SUCCESS;
}

arm_status arm_maxpool_s8_with_quantization (const cmsis_nn_context *ctx,
                          const cmsis_nn_pool_params *pool_params,
                          const cmsis_nn_dims *input_dims,
                          const q7_t *src,
                          const cmsis_nn_dims *filter_dims,
                          const cmsis_nn_dims *output_dims,
                          const int32_t input_offset,
                          const int32_t output_offset,
                          const int32_t multiplier,
                          const int32_t shift,
                          q7_t *dst)
{
    const int32_t input_y = input_dims->h;
    const int32_t input_x = input_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t stride_y = pool_params->stride.h;
    const int32_t stride_x = pool_params->stride.w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t pad_y = pool_params->padding.h;
    const int32_t pad_x = pool_params->padding.w;
    const int32_t act_min = pool_params->activation.min;
    const int32_t act_max = pool_params->activation.max;
    const int32_t ch_src = input_dims->c;

    if (ctx->buf == NULL && arm_avgpool_s8_get_buffer_size(output_dims->w, input_dims->c))
    {
        return ARM_MATH_ARGUMENT_ERROR;
    }
    q31_t *buffer = (q31_t *)ctx->buf;

    /* Run the following code for CPU's with DSP extension
     */
    for (int i_y = 0, idx_y = -pad_y; i_y < output_y; idx_y += stride_y, i_y++)
    {
        for (int i_x = 0, idx_x = -pad_x; i_x < output_x; idx_x += stride_x, i_x++)
        {
            /* Condition for kernel start dimension:
                      (base_idx_<x,y> + kernel_<x,y>_start) >= 0 */
            const int32_t kernel_y_start = MAX(0, -idx_y);
            const int32_t kernel_x_start = MAX(0, -idx_x);

            /* Condition for kernel end dimension:
                   (base_idx_<x,y> + kernel_<x,y>_end) < dim_src_<width,height> */
            const int32_t kernel_y_end = MIN(kernel_y, input_y - idx_y);
            const int32_t kernel_x_end = MIN(kernel_x, input_x - idx_x);

            int count = 0;

            for (int k_y = kernel_y_start; k_y < kernel_y_end; k_y++)
            {
                for (int k_x = kernel_x_start; k_x < kernel_x_end; k_x++)
                {
                    const q7_t *start = src + ch_src * (k_x + idx_x + (k_y + idx_y) * input_x);
                    if (count == 0)
                    {
                        for (int i = 0; i < ch_src; i++)
                        {
                            buffer[i] = start[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < ch_src; i++)
                        {
                            if (start[i] > buffer[i]) {
                                buffer[i] = start[i];
                            }
                        }
                    }
                    count++;
                }
            }

            for (int i = 0; i < ch_src; i++) {
                buffer[i] += input_offset;
                buffer[i] = arm_nn_requantize(buffer[i], multiplier, shift);
                buffer[i] += output_offset;
                buffer[i] = MAX(buffer[i], act_min);
                buffer[i] = MIN(buffer[i], act_max);

                dst[i] = (q7_t)buffer[i];
            }

            dst += ch_src;
        }
    }
	return ARM_MATH_SUCCESS;
}
