#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../include/custom.h"

// -----------------------------------------------------------------------------
// Software reference for single conv + pool (like Base_conv_mp)
// -----------------------------------------------------------------------------
int* Base_conv_mp(int inside, int c_in, int c_out, int8_t *pInputs, int8_t *pFilters) {
    int oside = (inside - 2); // after 3x3 conv + 2x2 pool
    int *poutputs = (int *)calloc(oside * oside * c_out, sizeof(int));

    for (int x = 0; x < c_out; x++) {                // output channels
        for (int i = 0; i < oside; i++) {
            for (int j = 0; j < oside; j++) {

                int sum = 0;
                for (int m = 0; m < c_in; m++) { 
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            sum += pFilters[x + m*9*c_out + ky*3*c_out + kx*c_out] * 
                                pInputs[m*inside*inside + (i + ky)*(inside) + (j + kx)];
                        }
                    }
                }
                poutputs[x*(inside-2)*(inside-2) + i*(inside-2) + j] = sum;

            }
        }
    }

    return poutputs;
}

// -----------------------------------------------------------------------------
// Error check function (same as example)
// -----------------------------------------------------------------------------
int error_check(int* test1, int* test2, int num){
    int error = 0;
    for(int i = 0; i < num; i++){
        error += abs(test1[i] - test2[i]);
    }
    return error;
}

int* RVSCNN_conv_mp(int inside, int c_in, int c_out, int8_t *pInputs, int8_t *pFilters) {
    int oside = inside - 2; // convolution output size (no pooling)
    int *poutputs = (int *)calloc(oside * oside * c_out, sizeof(int));

    int chn_num = c_out >> 2; 
    int a = 0;

    int t1 = record();
    L_SCNN(c_in, inside, oside * oside, 0);

    for (int i = 0; i < chn_num; i++) {
        // Configure accelerator for each of 4 output channels
        L_MODE(0, 1, 0, 1);
        L_MODE(0, 2, 0, 1);
        L_MODE(0, 3, 0, 1);
        L_MODE(0, 4, 0, 1);

        for (int m = 0; m < oside; m++) {
            for (int n = 0; n < oside; n++) {
                // Perform 3x3 convolution
                SCNN4x4(pFilters + 36 * c_in * i, pInputs + n + m * inside);

                // Read back convolution results (no pooling)
                SCNN_WB_INT(&a);
                poutputs[n + m * oside + 4 * i * oside * oside] = a;

                SCNN_WB_INT(&a);
                poutputs[oside * oside + n + m * oside + 4 * i * oside * oside] = a;

                SCNN_WB_INT(&a);
                poutputs[2 * oside * oside + n + m * oside + 4 * i * oside * oside] = a;

                SCNN_WB_INT(&a);
                poutputs[3 * oside * oside + n + m * oside + 4 * i * oside * oside] = a;
            }
        }
    }

    int t2 = record();
    printf("Accelerator cycle count (conv only): %d\n", t2 - t1);

    return poutputs;
}




// -----------------------------------------------------------------------------
// Accelerator conv + pool (RVSCNN)
// -----------------------------------------------------------------------------
// int* RVSCNN_conv_mp(int inside, int c_in, int c_out, int8_t *pInputs, int8_t *pFilters) {
//     int oside = inside - 2; // output size after 3x3 conv (no pooling)
//     int *poutputs = (int *)calloc(oside * oside * c_out, sizeof(int));

//     int chn_num = c_out >> 2; // groups of 4 channels
//     int tile = 1+(inside - 4)/2; // tile step = 1 now
//     int a = 0;

//     int t1 = record();
//     L_SCNN(c_in, inside, oside*oside, 0);

//     for (int i = 0; i < chn_num; i++) {
//         // set mode for each of the 4 channels
//         L_MODE(0, 1, 0, 1);
//         // L_MODE(0, 2, 0, 1);
//         // L_MODE(0, 3, 0, 1);
//         // L_MODE(0, 4, 0, 1);
//     for (int m = 0; m < tile; m++) {
//         for (int n = 0; n < tile; n++) {
//              SCNN4x4(pFilters + 9*c_in*i, pInputs + n + m*inside);
//                //POOL(2); // 2x2 maxpool

//                POOL_WB_INT(&a, 1);
//                poutputs[n + m*oside + 4*i*oside*oside] = a;

//                POOL_WB_INT(&a, 2);
//                poutputs[oside*oside + n + m*oside + oside*oside] = a;

//                POOL_WB_INT(&a, 3);
//                poutputs[2*oside*oside + n + m*oside + oside*oside] = a;

//                POOL_WB_INT(&a, 4);
//                poutputs[3*oside*oside + n + m*oside + oside*oside] = a;
//             // // Use the same input tile for all output channels
//             // SCNN4x4(pFilters + 36*c_in*i, pInputs + n*2 + m*2*inside); // always top-left corner

//             // POOL(2); // 2x2 maxpool

//             // POOL_WB_INT(&a, 1);
//             // for (int ch = 0; ch < c_out; ch++) {
//             //     poutputs[ch*oside*oside + n + m*oside] = a; // broadcast to all channels
//             // }
//         }
//     }
//     }
//     int t2 = record();
//     printf("Accelerator cycle count: %d\n", t2 - t1);

//     return poutputs;
// }
//         for (int m = 0; m < oside; m++) {
//             for (int n = 0; n < oside; n++) {
//                 // Perform convolution on the 3x3 tile starting at (n,m)
//                 SCNN4x4(pFilters + 9*c_in*i, pInputs + n + m*inside);
//                 printf("input",pInputs);
//                 // Read results for 4 channels (broadcast if needed)
//                 POOL_WB_INT(&a, 1);
//                 poutputs[0*oside*oside + n + m*oside] = a;

//                 POOL_WB_INT(&a, 2);
//                 poutputs[1*oside*oside + n + m*oside] = a;

//                 POOL_WB_INT(&a, 3);
//                 poutputs[2*oside*oside + n + m*oside] = a;

//                 POOL_WB_INT(&a, 4);
//                 poutputs[3*oside*oside + n + m*oside] = a;
//             }
//         }
//     }

//     int t2 = record();
//     printf("Accelerator cycle count: %d\n", t2 - t1);

//     return poutputs;
// }





// -----------------------------------------------------------------------------
// Main testbench (same structure as example)
// -----------------------------------------------------------------------------
// int main() {
//     int inside = 5;
//     int c_out =1;   // number of output channels
//     int c_in = 1;    // single channel input

//     // Allocate input and filter matrices
//     int8_t *input_matrix = (int8_t *)malloc(inside * inside * c_in * sizeof(int8_t));
//     int8_t *filter_matrix = (int8_t *)malloc(3 * 3 * c_in * c_out * sizeof(int8_t));

//     // Fill input (1..25 for 5x5)
//     for (int i = 0; i < inside * inside * c_in; i++) {
//         input_matrix[i] = i + 1;
//     }

//     // Fixed 3x3 kernel: [ [1,0,-1], [1,0,-1], [1,0,-1] ]
//     int k[9] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
//     for (int i = 0; i < 9 * c_out; i++) {
//         filter_matrix[i] = k[i % 9];
//     }

//     int oside = inside - 2; // 5-2=3
//     int num_elem = oside * oside * c_out;

//     // Run software convolution (no pooling)
//     int *base_out = Base_conv_mp(inside, c_in, c_out, input_matrix, filter_matrix);
//     int *out1 = RVSCNN_conv_mp(inside, c_in, c_out, input_matrix, filter_matrix);

//     int error = error_check(out1, base_out, num_elem);
//     printf("inside:%d c_in:%d c_out:%d\n", inside, c_in, c_out);
//     printf("error:%d\n", error);
    
//     // Print output matrices
//     printf("Convolution Output Matrices:\n");
//     for (int c = 0; c < c_out; c++) {
//         printf("Channel %d:\n", c);
//         for (int i = 0; i < oside; i++) {
//             for (int j = 0; j < oside; j++) {
//                 printf("%4d ", base_out[c * oside * oside + i * oside + j]);
//             }
//             printf("\n");
//         }
//         printf("\n");
//     }
//     //  for (int c = 0; c < c_out; c++) {
//     //     printf("Channel %d:\n", c);
//     //     for (int i = 0; i < oside; i++) {
//     //         for (int j = 0; j < oside; j++) {
//     //             printf("%4d ", out1[c * oside * oside + i * oside + j]);
//     //         }
//     //         printf("\n");
//     //     }
//     //     printf("\n");
//     // }
//      printf("Accelerator Output:\n");
//     for (int c = 0; c < c_out; c++) {
//         printf("Channel %d:\n", c);
//         for (int i = 0; i < oside; i++) {
//             for (int j = 0; j < oside; j++) {
//                 printf("%4d ", out1[c*oside*oside + i*oside + j]);
//             }
//             printf("\n");
//         }
//     }

//     free(input_matrix);
//     free(filter_matrix);
//     free(base_out);

//     return 0;
// }
int main(){
    int inside = 5;
    int c_out = 4;   // can be increased if needed
    int c_in = 1;    // single channel like Python example

    int8_t *input_matrix = (int8_t *)malloc(inside*inside*c_in*sizeof(int8_t));
    int8_t *filter_matrix = (int8_t *)malloc(3*3*c_in*c_out*sizeof(int8_t));

    // Fill input and filter same as Python example
    for (int i = 0; i < inside*inside*c_in; i++) {
        input_matrix[i] = i + 1;  // 1..25 for 5x5
    }
    int k[9] = { 1, 0, -1,  1, 0, -1,  1, 0, -1 };
    for (int i = 0; i < 9*c_out; i++) {
        filter_matrix[i] = k[i % 9];
    }

    int oside = (inside - 2)/2;
    int num_elem = oside*oside*c_out;

    // Accelerator run
    int *out1 = RVSCNN_conv_mp(inside, c_in, c_out, input_matrix, filter_matrix);

    // Software reference
    int *base_out = Base_conv_mp(inside, c_in, c_out, input_matrix, filter_matrix);

    // Error check
    int error = error_check(out1, base_out, num_elem);
    printf("inside:%d c_in:%d c_out:%d\n", inside, c_in, c_out);
    printf("error:%d\n", error);

     printf("Accelerator Convolution Output:\n");
    for (int c = 0; c < c_out; c++) {
        printf("Channel %d:\n", c);
        for (int i = 0; i < inside - 2; i++) {
            for (int j = 0; j < inside - 2; j++) {
                printf("%4d ", out1[c*(inside-2)*(inside-2) + i*(inside-2) + j]);
            }
            printf("\n");
        }
    }

    printf("Convolution Output Matrices:\n");
   for (int c = 0; c < c_out; c++) {
        printf("Channel %d:\n", c);
        for (int i = 0; i < oside; i++) {
            for (int j = 0; j < oside; j++) {
                printf("%4d ", base_out[c*oside*oside + i*oside + j]);
            }
            printf("\n");
        }
    }
    // Print output
   

    free(input_matrix);
    free(filter_matrix);
    free(out1);
    free(base_out);

    return 0;
}


