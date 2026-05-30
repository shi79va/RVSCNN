#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include "../include/custom.h"

int* Base_conv_mp(int inside, int c_in, int c_out, int8_t *pInputs, int8_t *pFilters){
    int oside = (inside - 2)/2;//after maxpool
	int *poutputs = (int *)calloc(oside*oside*c_out,sizeof(int));
    
    int num = oside;
    for(int x=0;x<c_out;x++){
        //cal the current layer(c_in) out
        for(int i=0;i<num;i++){
            for(int j=0;j<num;j++){
                int a[4] = {0,0,0,0};
                for(int m=0;m<c_in;m++){//add
                    for(int my=0;my<2;my++){
                        for(int mx=0;mx<2;mx++){
                            a[mx+my*2]+=(pFilters[x + m*9*c_out + 0*c_out]*pInputs[m*inside*inside+2*j+2*i*inside+mx+my*inside]);
                            a[mx+my*2]+=(pFilters[x + m*9*c_out + 1*c_out]*pInputs[m*inside*inside+2*j+2*i*inside+1+mx+my*inside]);
                            a[mx+my*2]+=(pFilters[x + m*9*c_out + 2*c_out]*pInputs[m*inside*inside+2*j+2*i*inside+2+mx+my*inside]);

                            a[mx+my*2]+=(pFilters[x + m*9*c_out + 3*c_out]*pInputs[m*inside*inside+2*j+2*i*inside+inside+mx+my*inside]);
                            a[mx+my*2]+=(pFilters[x + m*9*c_out + 4*c_out]*pInputs[m*inside*inside+2*j+2*i*inside+inside+1+mx+my*inside]);
                            a[mx+my*2]+=(pFilters[x + m*9*c_out + 5*c_out]*pInputs[m*inside*inside+2*j+2*i*inside+inside+2+mx+my*inside]);

                            a[mx+my*2]+=(pFilters[x + m*9*c_out + 6*c_out]*pInputs[m*inside*inside+2*j+2*i*inside+2*inside+mx+my*inside]);
                            a[mx+my*2]+=(pFilters[x + m*9*c_out + 7*c_out]*pInputs[m*inside*inside+2*j+2*i*inside+2*inside+1+mx+my*inside]);
                            a[mx+my*2]+=(pFilters[x + m*9*c_out + 8*c_out]*pInputs[m*inside*inside+2*j+2*i*inside+2*inside+2+mx+my*inside]);
                        }
                    }
                }
                int b = 0;
                if(a[0]<a[1]) b=a[1];
                else b=a[0];
                if(b<a[2]) b=a[2];
                if(b<a[3]) b=a[3];

                poutputs[x*oside*oside + j + i*num] = b;
            }
        }
    }
	
	return poutputs;
}
int* Base_conv(int inside, int c_in, int c_out, int8_t *pInputs, int8_t *pFilters) {
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

int error_check(int* test1, int* test2, int num){
    int error = 0;
    for(int i=0;i<num;i++){
        error += abs(test1[i] - test2[i]);
    }

    return error;
}

int main(){
    //////////////////////////////////////////////gemm
    int inside = 4;
    int c_out = 4;
    int c_in = 1; //16 32 64 128 256 ......


    int8_t *input_matrix = (int8_t *)malloc(inside*inside*c_in*sizeof(int8_t));
    int8_t *filter_matrix = (int8_t *)malloc(3*3*c_in*c_out*sizeof(int8_t));

    for (int i = 0; i < inside*inside*c_in; i++) {
        input_matrix[i] = i + 1;  // 1..25 for 5x5
    }


    // for(int i=0;i<3*3*c_in*c_out;i++){
    //     filter_matrix[i] = -5 + i%7;   //random input
    // }
    int k[9] = { 1, 0, -1,  1, 0, -1,  1, 2, -1 };
    for (int i = 0; i < 9*c_out; i++) {
        filter_matrix[i] = k[i % 9];
    }

    //set
    int oside = (inside - 2)/2;//after maxpool
    int *out1 = (int *)calloc(oside*oside*c_out,sizeof(int));
    int chn_num =  c_out>>2;
    int tile = 1 + (inside-4)/2;
    int a=0;

    /////////////////////////////////////////////////////////////////////////////////////////
    //fast conv_mp
    int t1 = record();
    L_SCNN(c_in,inside,oside*oside,0);
	for(int i=0;i<chn_num;i++){
        L_MODE(0,1,0,1);
        L_MODE(0,2,0,1);
        L_MODE(0,3,0,1);
        L_MODE(0,4,0,1);

        for(int m=0;m<tile;m++){
            for(int n=0;n<tile;n++){
                SCNN4x4(filter_matrix+36*c_in*i,input_matrix+n*2+m*2*inside);
                POOL(2);//cnn maxpool

                POOL_WB_INT(&a,1);
                out1[n + m*oside + 4*i*oside*oside] = a;
                POOL_WB_INT(&a,2);
                out1[oside*oside + n + m*oside + 4*i*oside*oside] = a;
                POOL_WB_INT(&a,3);
                out1[2*oside*oside + n + m*oside + 4*i*oside*oside] = a;
                POOL_WB_INT(&a,4);
                out1[3*oside*oside + n + m*oside + 4*i*oside*oside] = a;
            }
        }
    }
    int t2 = record();
    /////////////////////////////////////////////////////////////////////////////////////////

    //error check: whether the result is same with RV32IM ?
    int *base_out = Base_conv_mp(inside,c_in,c_out,input_matrix,filter_matrix);
    int error = error_check(out1, base_out, oside*oside*c_out);
    int *base_conv = Base_conv(inside, c_in, 1, input_matrix, filter_matrix);

    //Fast GOPS Calculation
    float GOPS = (((c_out*(inside-2)*(inside-2)*(c_in*3*3))*1.0/(t2-t1))) * 0.3; //1 OP = 1 MAC //300MHz

    printf("inside:%d c_in:%d c_out:%d\n",inside,c_in,c_out);
    printf("error:%d\n",error);
    printf("cycle count::%d\n",t2-t1);
    printf("GOPS:%.4f\n",GOPS);
    
    printf("--------------------:\n");
    printf("base-conv Output:\n");
    for (int c = 0; c < c_out; c++) {
        printf("Channel %d:\n", c);
        for (int i = 0; i < oside; i++) {
            for (int j = 0; j < oside; j++) {
                printf("%4d ", base_out[c*oside*oside + i*oside + j]);
            }
            printf("\n");
        }
    }
    printf("--------------------:\n");
    printf("Convolution Output Matrices:\n");
    for (int c = 0; c < 1; c++) {
        printf("Channel %d:\n", c);
        for (int i = 0; i < (inside-2); i++) {
            for (int j = 0; j < (inside-2); j++) {
                printf("%4d ", base_conv[c * (inside-2)* (inside-2) + i * (inside-2) + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    return 0;
}

//GOPS
//inside=4 c_out=4
//c_in=1024  GOPS:3.2949
//c_in=512   GOPS:3.2671
//c_in=256   GOPS:3.2130
//c_in=128   GOPS:3.1100
//c_in=64    GOPS:2.9226
//c_in=32    GOPS:2.6083
//c_in=16    GOPS:2.5246

////////////////////////////////////////////////////////////////////////correct base conv /////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../include/custom.h"
// Base_conv with stride=1 and padding parameter
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




// int* RVSCNN_conv_mp(int inside, int c_in, int c_out, int8_t *pInputs, int8_t *pFilters, int stride, int padding) {
//     int ksize = 3;
//     int oside = (inside - ksize + 2*padding)/stride + 1;  // output size
//     int *poutputs = (int *)calloc(oside * oside * c_out, sizeof(int));

//     // Number of tiles along each dimension (assuming 2x2 pooling)
//     int tile_side = 1 + (oside - 2)/2;  
//     int a = 0;

//     int chn_num = (c_out + 3)/4; // ceil division to cover all output channels in groups of 4

//     int t1 = record();
//     L_SCNN(c_in, inside, oside*oside, 0);

//     for(int i = 0; i < chn_num; i++) {
//         // Configure modes for up to 4 channels
//         for(int ch=0; ch<4; ch++)
//             if((i*4+ch) < c_out)
//                 L_MODE(0, ch+1, 0, 1);

//         // Loop over tiles
//         for(int m = 0; m < tile_side; m++) {
//             for(int n = 0; n < tile_side; n++) {
//                 int8_t tile_input[16] = {0}; // 4x4 input tile for SCNN
//                 for(int ch_in = 0; ch_in < c_in; ch_in++) {
//                     for(int y=0; y<4 && (m*2 + y) < inside; y++) {
//                         for(int x=0; x<4 && (n*2 + x) < inside; x++) {
//                             tile_input[y*4 + x] = pInputs[ch_in*inside*inside + (m*2 + y)*inside + (n*2 + x)];
//                         }
//                     }
//                 }

//                 // SCNN4x4 expects tile_input and corresponding filters
//                 SCNN4x4(pFilters + i*4*c_in*ksize*ksize, tile_input);

//                 // 2x2 MaxPool
//                 POOL(2);

//                 // Write back for up to 4 channels
//                 for(int ch=0; ch<4; ch++) {
//                     int ch_idx = i*4 + ch;
//                     if(ch_idx >= c_out) continue;

//                     POOL_WB_INT(&a, ch+1);
//                     int out_base = ch_idx * oside * oside;
//                     int out_i = m;
//                     int out_j = n;
//                     poutputs[out_base + out_i*tile_side + out_j] = a;
//                 }
//             }
//         }
//     }

//     int t2 = record();
//     printf("Accelerator cycle count: %d\n", t2 - t1);

//     return poutputs;
// }


// -------------------- Error check --------------------
int error_check(int *cpu_out, int *accel_out, int c_out, int oside) {
    int error = 0;
    for(int ch=0; ch<c_out; ch++)
        for(int i=0; i<oside; i++)
            for(int j=0; j<oside; j++)
                error += abs(cpu_out[ch*oside*oside + i*oside + j] - accel_out[ch*oside*oside + i*oside + j]);
    return error;
}
// Example usage
int main() {
    int inside=5, c_in=1, c_out=1;
    int stride=1, padding=0;
    
    int8_t *input = (int8_t*)malloc(inside*inside*c_in*sizeof(int8_t));
    for(int i=0;i<25;i++) input[i]=i+1;

    int8_t kernel[9] = { 1, 1, 1,  1, 1, 1,  1, 1, 1};
    int8_t *filters = (int8_t *)malloc(c_out*c_in*9*sizeof(int8_t));
    for(int i=0;i<9;i++) filters[i] = kernel[i];
    // -------- CPU convolution + MaxPool --------
    int *conv_out = Base_conv(inside, c_in, c_out, input, filters, stride, padding);

    int conv_side = inside - 3 + 1; // conv output side
    int *pool_out = MaxPool2x2(conv_out, c_out, conv_side);

    int *out = Base_conv(inside, c_in, c_out, input, filters, stride, padding);
    int ksize = 3;
    int oside = (inside - ksize + 2*padding)/stride + 1;

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
    
    // -------- Compare CPU vs Accelerator --------
    // int error = error_check(pool_out, accel_out, c_out, oside);
    // printf("Total error (CPU vs Accelerator): %d\n", error);
    free(input);
    free(filters);
    free(conv_out);
    free(pool_out);
    
    return 0;
}
