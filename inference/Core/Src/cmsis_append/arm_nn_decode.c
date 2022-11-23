#include "stdio.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

q31_t *arm_nn_decode_4d (q31_t *dec_buf,
                        q31_t *dec_buf_end,
                        const q7_t **filter_ptr,
                        const q7_t *end_ptr,
                        int32_t *last_pos,
                        int32_t *res_out,
                        const int32_t input_ch,
                        const int32_t kernel_x,
                        const int32_t kernel_y,
                        const int32_t block) 
{
    const int32_t remain = 2 * (block + 4);
    int res = 0;
    int32_t pos[4] = {0};
    q31_t *cur_buf_ptr = dec_buf;
    // printf("%d %d\r\n",*filter_ptr, end_ptr);
    
    memcpy(pos, last_pos, 16);
    if ((*res_out) > 0) {
        pos[0] -= (*res_out);
    }

    while (dec_buf_end - cur_buf_ptr >= remain) {
        if (*filter_ptr >= end_ptr) {
            break;
        }
        //printf("cur_buf_ptr:  %d %d\r\n",cur_buf_ptr, dec_buf_end);
        //printf("filter: %d %d\r\n",*filter_ptr, end_ptr);
        pos[0] = pos[0] + (**filter_ptr) + 128;
        ++(*filter_ptr);
        while (pos[0] >= input_ch) {
            ++pos[1]; pos[0] -= input_ch;
            while (pos[1] >= kernel_x) {
                ++pos[2]; pos[1] -= kernel_x;
                while (pos[2] >= kernel_y) {
                    ++pos[3]; pos[2] -= kernel_y;
                }
            }
        }

        if (**filter_ptr == 0) {
            ++(*filter_ptr);
            continue;
        }

        if (pos[0] + block <= input_ch) {
            res = 0;
            memcpy(cur_buf_ptr, pos, 16);
            cur_buf_ptr += 4;
            for (int i = 0; i < block; ++i) {
                (*cur_buf_ptr++) = (q31_t) **filter_ptr;
                ++(*filter_ptr);
            }
            pos[0] += (block - 1);
        } else {
            res = (pos[0] + block) - input_ch;
            pos[0] -= res;
            memcpy(cur_buf_ptr, pos, 16);
            cur_buf_ptr += 4;
            for (int i = 0; i < res; ++i) {
                (*cur_buf_ptr++) = 0;
            }
            for (int i = 0; i < block - res; ++i) {
                (*cur_buf_ptr++) = (q31_t) **filter_ptr;
                ++(*filter_ptr);
            }

            pos[0] += block;
            while (pos[0] >= input_ch) {
                ++pos[1]; pos[0] -= input_ch;
                while (pos[1] >= kernel_x) {
                    ++pos[2]; pos[1] -= kernel_x;
                    while (pos[2] >= kernel_y) {
                        ++pos[3]; pos[2] -= kernel_y;
                    }
                }
            }
            memcpy(cur_buf_ptr, pos, 16);
            cur_buf_ptr += 4;
            for (int i = 0; i < res; ++i) {
                (*cur_buf_ptr++) = (q31_t) **filter_ptr;
                ++(*filter_ptr);
            }
            for (int i = 0; i < block - res; ++i) {
                (*cur_buf_ptr++) = 0;
            }

            pos[0] += (res-1);
        }
        
    }
    
    //printf("%d\r\n",cur_buf_ptr);
    *res_out = res;
    return cur_buf_ptr;
}
