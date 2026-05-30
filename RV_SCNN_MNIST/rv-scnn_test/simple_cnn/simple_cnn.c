#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../include/custom.h"
// Base_conv with stride=1 and padding parameter

// Ceiling division utility
static inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}
int* Base_conv(int inside, int c_in, int c_out, int8_t *pInputs, int8_t *pFilters,
               int stride, int padding) 
{
    int ksize = 3; // 3x3 kernel
    int oside = (inside - ksize + 2*padding)/stride + 1; // output size
    int *poutputs = (int *)calloc(oside * oside * c_out, sizeof(int));

    int padded_size = inside + 2*padding;
    int8_t *padded = (int8_t *)calloc(padded_size * padded_size * c_in, sizeof(int8_t));
    for(int m=0; m<c_in; m++)
        for(int i=0;i<inside;i++)
            for(int j=0;j<inside;j++)
                padded[m*padded_size*padded_size + (i+padding)*padded_size + (j+padding)] =
                    pInputs[m*inside*inside + i*inside + j];

    for(int x=0;x<c_out;x++)
        for(int i=0;i<oside;i++)
            for(int j=0;j<oside;j++) {
                int sum=0;
                for(int m=0;m<c_in;m++)
                    for(int ky=0;ky<ksize;ky++)
                        for(int kx=0;kx<ksize;kx++) {
                            int in_idx = m*padded_size*padded_size + (i*stride+ky)*padded_size + (j*stride+kx);
                            int filter_idx = x + m*ksize*ksize*c_out + (ky*ksize+kx)*c_out;
                            sum += pFilters[filter_idx] * padded[in_idx];
                        }
                poutputs[x*oside*oside + i*oside + j] = sum;
            }

    free(padded);
    return poutputs;
    
}

// Maxpooling 2x2 with stride=2
int* MaxPool2x2(int *conv_out, int c_out, int conv_side) {
    int oside = (conv_side + 1) / 2; // ceil division for 3x3→2x2 case
    int *pool_out = (int *)calloc(oside * oside * c_out, sizeof(int));

    for (int ch = 0; ch < c_out; ch++) {
        for (int i = 0; i < oside; i++) {
            for (int j = 0; j < oside; j++) {
                int i0 = i * 2;
                int j0 = j * 2;
                int max = 0;

                for (int dy = 0; dy < 2 && i0 + dy < conv_side; dy++) {
                    for (int dx = 0; dx < 2 && j0 + dx < conv_side; dx++) {
                        int val = conv_out[ch*conv_side*conv_side + (i0 + dy)*conv_side + (j0 + dx)];
                        if (val > max) max = val;
                    }
                }
                pool_out[ch*oside*oside + i*oside + j] = max;
            }
        }
    }
    return pool_out;
}



int* RVSCNN_conv_mp(int inside, int c_in, int c_out, int8_t *pInputs, int8_t *pFilters, int stride, int padding) {
    int ksize = 3;
    int conv_side = (inside - ksize + 2*padding)/stride + 1; // convolution output size
    int oside = div_ceil(conv_side, 2); // pooled output size (2x2 for inside=5)
    int *poutputs = (int *)calloc(oside * oside * c_out, sizeof(int));
    if (!poutputs) { perror("calloc poutputs"); return NULL; }

    int chn_num = c_out >> 2; // groups of 4 output channels
    int tile = div_ceil(conv_side, 2); // tiles to cover 2x2 output
    int a = 0;

    // Debug: Print filter values for each group
    printf("Filter values:\n");
    for(int i=0; i<chn_num; i++) {
        printf("Group %d:\n", i);
        int8_t *group_filters = pFilters + i*ksize*ksize*c_in*4;
        for(int f=0; f<ksize*ksize*c_in*4; f++) {
            printf("%4d ", group_filters[f]);
            if((f+1) % ksize == 0) printf("\n");
            if((f+1) % (ksize*ksize) == 0) printf("\n");
        }
    }

    // Fast conv_mp
    int t1 = record();
    L_SCNN(c_in, inside, oside*oside, 0);
    for(int i = 0; i < chn_num; i++) {
        L_MODE(0, 1, 0, 1);
        L_MODE(0, 2, 0, 1);
        L_MODE(0, 3, 0, 1);
        L_MODE(0, 4, 0, 1);

        for(int m = 0; m < tile; m++) {
            for(int n = 0; n < tile; n++) {
                // Prepare 4x4 input tile with padding
                int8_t tile_input[16] = {0};
                for(int y = 0; y < 4; y++) {
                    for(int x = 0; x < 4; x++) {
                        int src_y = m*2*stride + y - padding;
                        int src_x = n*2*stride + x - padding;
                        if(src_y >= 0 && src_y < inside && src_x >= 0 && src_x < inside) {
                            tile_input[y*4 + x] = pInputs[src_y*inside + src_x];
                        }
                    }
                }

                // Debug: Print tile input
                printf("Tile (m=%d, n=%d):\n", m, n);
                for(int y=0; y<4; y++) {
                    for(int x=0; x<4; x++) {
                        printf("%4d ", tile_input[y*4 + x]);
                    }
                    printf("\n");
                }

                // Use filters for this group (9*4 = 36 weights for c_in=1)
                int8_t *group_filters = pFilters + i*ksize*ksize*c_in*4;
                SCNN4x4(group_filters, tile_input);
                POOL(2); // cnn maxpool

                // Write back results for 4 channels
                POOL_WB_INT(&a, 1);
                printf("POOL_WB_INT channel 1: %d\n", a);
                if(m < oside && n < oside) poutputs[n + m*oside + 4*i*oside*oside] = a;
                POOL_WB_INT(&a, 2);
                printf("POOL_WB_INT channel 2: %d\n", a);
                if(m < oside && n < oside) poutputs[oside*oside + n + m*oside + 4*i*oside*oside] = a;
                POOL_WB_INT(&a, 3);
                printf("POOL_WB_INT channel 3: %d\n", a);
                if(m < oside && n < oside) poutputs[2*oside*oside + n + m*oside + 4*i*oside*oside] = a;
                POOL_WB_INT(&a, 4);
                printf("POOL_WB_INT channel 4: %d\n", a);
                if(m < oside && n < oside) poutputs[3*oside*oside + n + m*oside + 4*i*oside*oside] = a;
            }
        }
    }
    int t2 = record();

    // Fast GOPS Calculation
    float GOPS = (((c_out*(conv_side)*(conv_side)*(c_in*ksize*ksize))*1.0/(t2-t1))) * 0.3; // 1 OP = 1 MAC, 300MHz
    printf("Accelerator cycle count: %d\n", t2 - t1);
    printf("GOPS: %.4f\n", GOPS);
    return poutputs;
}

// Error check
int error_check(int *cpu_out, int *accel_out, int c_out, int oside) {
    int64_t error = 0; // Use int64_t to avoid overflow
    for(int ch=0; ch<c_out; ch++)
        for(int i=0; i<oside; i++)
            for(int j=0; j<oside; j++) {
                int cpu_val = cpu_out[ch*oside*oside + i*oside + j];
                int accel_val = accel_out[ch*oside*oside + i*oside + j];
                error += llabs((int64_t)cpu_val - (int64_t)accel_val);
                printf("Error check: ch=%d, i=%d, j=%d, cpu=%d, accel=%d, diff=%lld\n", 
                       ch, i, j, cpu_val, accel_val, llabs((int64_t)cpu_val - (int64_t)accel_val));
            }
    return (int)error;
}
// Example usage
int main() {
    int inside=4, c_in=1, c_out=4;
    int stride=1, padding=0;
    
    int8_t *input = (int8_t*)malloc(inside*inside*c_in*sizeof(int8_t));
    for(int i=0;i<25;i++) input[i]=i+1;

    int8_t kernel[9] = { 1, 0, -1,  1, 0, -1,  1, 0, -1};
    int8_t *filters = (int8_t *)malloc(c_out*c_in*9*sizeof(int8_t));
    for(int i=0;i<9;i++) filters[i] = kernel[i];
    // -------- CPU convolution + MaxPool --------
    int *conv_out = Base_conv(inside, c_in, c_out, input, filters, stride, padding);

    int conv_side = inside - 3 + 1; // conv output side
    int *pool_out = MaxPool2x2(conv_out, c_out, conv_side);

    int pool_oside = div_ceil(conv_side, 2);
    int *out = Base_conv(inside, c_in, c_out, input, filters, stride, padding);
    int ksize = 3;
    int oside = (inside - ksize + 2*padding)/stride + 1;
    int *accel_out = RVSCNN_conv_mp(inside, c_in, c_out, input, filters, stride, padding);
    printf("inside:%d c_in:%d c_out:%d\n",inside,c_in,c_out);
    
    for(int ch=0; ch<c_out; ch++) {
        printf("Channel %d:\n", ch);
        for(int i=0; i<oside; i++) {
            for(int j=0; j<oside; j++) {
                printf("%4d ", out[ch*oside*oside + i*oside + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\nMaxPool2x2 output:\n");
    for (int ch = 0; ch < c_out; ch++) {
        printf("Channel %d:\n", ch);
        int oside = (conv_side + 1) / 2;  // must match the pooling function
        for (int i = 0; i < oside; i++) {
            for (int j = 0; j < oside; j++) {
                printf("%4d ", pool_out[ch * oside * oside + i * oside + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("Accelerator RVSCNN_conv_mp output:\n");
    for(int ch=0; ch<c_out; ch++) {
        printf("Channel %d:\n", ch);
        for(int i=0; i<pool_oside; i++) {
            for(int j=0; j<pool_oside; j++) {
                printf("%4d ", accel_out[ch*pool_oside*pool_oside + i*pool_oside + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    // -------- Compare CPU vs Accelerator --------
    int error = error_check(pool_out, accel_out, c_out, pool_oside);
    printf("Total error (CPU vs Accelerator): %d\n", error);
    free(input);
    free(filters);
    free(conv_out);
    free(pool_out);
    
    return 0;
}
