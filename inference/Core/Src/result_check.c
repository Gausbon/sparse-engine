#include "arm_nnfunctions.h"
#include "stdio.h"

int result_check_statistics(const cmsis_nn_context *ctx,
                          const q7_t *section,
                          const int32_t conv_count,
                          const int32_t conv_time,
                          const int32_t linear_count,
                          const int32_t linear_time,
                          const int32_t trans_count,
                          const int32_t trans_time,
                          const int32_t softmax_count,
                          const int32_t softmax_time,
                          const int32_t norm_count,
                          const int32_t norm_time,
                          const int32_t pool_count,
                          const int32_t pool_time,
                          const int32_t matmul_count,
                          const int32_t matmul_time,
                          const int32_t add_count,
                          const int32_t add_time,
                          const int32_t class) {

    printf("Inference result:\r\n");
    for(int i = 0; i < 10; i++){
        printf("%d ",section[i]);
    }
    printf("\r\n");
    int32_t *buffer = (int32_t*) ctx->buf;
    for(int i = 0; i < 10; i++){
        buffer[i] = i;
    }

    for (int i = 1; i < 10; i++) {
			int j = i;
			while (j) {
				if (section[i] > section[j-1]) {
					buffer[i]--;
					buffer[j - 1]++;
				}
				j--;
			}
		}

    printf("Index sort:\r\n");
    for(int i = 0; i < 10; i++){
        printf("%d ",buffer[i]);
    }
    printf("\r\nclass:%d\r\n",class);

    if (buffer[class] < 1) {
      printf("Top 1 check: True\r\n");
    } else {
      printf("Top 1 check: False\r\n");
    }

    if (buffer[class] < 5) {
      printf("Top 5 check: True\r\n");
    } else {
      printf("Top 5 check: False\r\n");
    }

    int32_t total_time = conv_time + linear_time + trans_time + softmax_time
          + norm_time + pool_time + matmul_time + add_time;
    
    printf("\r\ntotal inference time:%d\r\n\r\n", total_time);
    printf("%10s%10s%10s%10s\r\n","count","time","avg","ratio");
    printf("conv: %4d%10d%10.2f%10.2f%%\r\n",conv_count,conv_time,
        ((conv_count!=0)?((double)conv_time/conv_count):0),(double)conv_time*100/total_time);
    printf("linear: %2d%10d%10.2f%10.2f%%\r\n",linear_count,linear_time,
        ((linear_count!=0)?((double)linear_time/linear_count):0),(double)linear_time*100/total_time);
    printf("trans: %3d%10d%10.2f%10.2f%%\r\n",trans_count,trans_time,
        ((trans_count!=0)?((double)trans_time/trans_count):0),(double)trans_time*100/total_time);
    printf("softmax: %1d%10d%10.2f%10.2f%%\r\n",softmax_count,softmax_time,
        ((softmax_count!=0)?((double)softmax_time/softmax_count):0),(double)softmax_time*100/total_time);
    printf("norm: %4d%10d%10.2f%10.2f%%\r\n",norm_count,norm_time,
        ((norm_count!=0)?((double)norm_time/norm_count):0),(double)norm_time*100/total_time);
    printf("pool: %4d%10d%10.2f%10.2f%%\r\n",pool_count,pool_time,
        ((pool_count!=0)?((double)pool_time/pool_count):0),(double)pool_time*100/total_time);
    printf("matmul: %2d%10d%10.2f%10.2f%%\r\n",matmul_count,matmul_time,
        ((matmul_count!=0)?((double)matmul_time/matmul_count):0),(double)matmul_time*100/total_time);
    printf("add: %5d%10d%10.2f%10.2f%%\r\n",add_count,add_time,
        ((add_count!=0)?((double)add_time/add_count):0),(double)add_time*100/total_time);
    printf("\r\n\r\n");
    return 0;
}
