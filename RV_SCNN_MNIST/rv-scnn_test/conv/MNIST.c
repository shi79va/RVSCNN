
// //inside more than 4  , padding =1 , c_out more than 4  => correct
// //RELU is correct but increases the number of cycles
// ///The problem of Bias is solved!! ,,, Note::the value of bias should be negetive , because the hardware subtracts it internally!!

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../include/custom.h"

// -----------------------------------------------------------------------------
//
//----------Software reference for single conv + pool (like Base_conv_mp)-------
//
// -----------------------------------------------------------------------------
int* Base_conv_mp(int inside, int c_in, int c_out, int8_t *pInputs, int8_t *pFilters,int stride, int padding) {
     int ksize = 3; // 3x3 kernel
    int conv_out = ((inside + 2 * padding - ksize) / stride) + 1;
    int oside = ((conv_out - 2) / 2) + 1; //first 2 is for pool_size   second 2 is for pool_stride
    //int oside = (inside - 2) / 2; // after 3x3 conv + 2x2 pool   ////original///
    printf("oside base = %d\n",oside);
    int *temp_buffer1 = (int *)calloc(oside * oside * c_out, sizeof(int));

    for (int x = 0; x < c_out; x++) {               
        for (int i = 0; i < oside; i++) {
            for (int j = 0; j < oside; j++) {
                int a[4] = {0, 0, 0, 0};
                for (int m = 0; m < c_in; m++) {    
                    for (int my = 0; my < 2; my++) {
                        for (int mx = 0; mx < 2; mx++) {
                            a[mx + my*2] += (pFilters[x + m*9*c_out + 0*c_out] * pInputs[m*inside*inside + 2*j + 2*i*inside + mx + my*inside]);
                            a[mx + my*2] += (pFilters[x + m*9*c_out + 1*c_out] * pInputs[m*inside*inside + 2*j + 2*i*inside + 1 + mx + my*inside]);
                            a[mx + my*2] += (pFilters[x + m*9*c_out + 2*c_out] * pInputs[m*inside*inside + 2*j + 2*i*inside + 2 + mx + my*inside]);

                            a[mx + my*2] += (pFilters[x + m*9*c_out + 3*c_out] * pInputs[m*inside*inside + 2*j + 2*i*inside + inside + mx + my*inside]);
                            a[mx + my*2] += (pFilters[x + m*9*c_out + 4*c_out] * pInputs[m*inside*inside + 2*j + 2*i*inside + inside + 1 + mx + my*inside]);
                            a[mx + my*2] += (pFilters[x + m*9*c_out + 5*c_out] * pInputs[m*inside*inside + 2*j + 2*i*inside + inside + 2 + mx + my*inside]);

                            a[mx + my*2] += (pFilters[x + m*9*c_out + 6*c_out] * pInputs[m*inside*inside + 2*j + 2*i*inside + 2*inside + mx + my*inside]);
                            a[mx + my*2] += (pFilters[x + m*9*c_out + 7*c_out] * pInputs[m*inside*inside + 2*j + 2*i*inside + 2*inside + 1 + mx + my*inside]);
                            a[mx + my*2] += (pFilters[x + m*9*c_out + 8*c_out] * pInputs[m*inside*inside + 2*j + 2*i*inside + 2*inside + 2 + mx + my*inside]);
                        }
                    }
                }

                int b = 0;
                if (a[0] < a[1]) b = a[1]; else b = a[0];
                if (b < a[2]) b = a[2];
                if (b < a[3]) b = a[3];

                temp_buffer1[x*oside*oside + j + i*oside] = b;
            }
        }
    }

    return temp_buffer1;
}
 int* Base_conv(int inside, int c_in, int c_out,
               int8_t *pInputs, int8_t *pFilters,
               int stride, int padding)
{
    int ksize = 3;
    int oside = ((inside + 2 * padding - ksize) / stride) + 1;

    int *poutputss = (int *)calloc(oside * oside * c_out, sizeof(int));

    for (int x = 0; x < c_out; x++) {               
        for (int i = 0; i < oside; i++) {
            for (int j = 0; j < oside; j++) {

                int sum = 0;

                for (int m = 0; m < c_in; m++) {
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {

                            // ----- FIXED PADDING HANDLING -----
                            int in_y = i * stride + ky - padding;
                            int in_x = j * stride + kx - padding;

                            int8_t inval = 0;  // default padded value

                            if (in_y >= 0 && in_y < inside &&
                                in_x >= 0 && in_x < inside) {
                                inval = pInputs[m*inside*inside +
                                               in_y*inside + in_x];
                            }

                            // original filter indexing preserved
                            int fidx = x + m*9*c_out + ky*3*c_out + kx*c_out;

                            sum += inval * pFilters[fidx];
                        }
                    }
                }

                
                poutputss[x*oside*oside + i*oside + j] = sum;
            }
        }
    }

    return poutputss;
}


// -----------------------------------------------------------------------------
// Error check function (same as example)
// -----------------------------------------------------------------------------
// int error_check(int* test1, int* test2, int num){
//     int error = 0;
//     for(int i = 0; i < num; i++){
//         error += abs(test1[i] - test2[i]);
//     }
//     return error;
// }

// -----------------------------------------------------------------------------
// Requanize() function
// -----------------------------------------------------------------------------

static inline uint8_t requantize( int x,float requant_mult,int z_y,int qmin,int qmax) 
{
    float yf = x * requant_mult;

    // round-to-nearest
    int y = (int)(yf + (yf >= 0 ? 0.5f : -0.5f));

    y += z_y;

    if (y < qmin) y = qmin;
    if (y > qmax) y = qmax;

    return (uint8_t)y;
}
// --------------------------------------------------------------------------------------
//
// ---------------------------Accelerator conv + RELU +  pool (RVSCNN)-------------------
//
// --------------------------------------------------------------------------------------

 int8_t* RVSCNN_conv_mp(int inside, int c_in, int c_out, int8_t *pInputs, int8_t *pFilters, int stride, int padding, int *conv_bias , const float *requant_mult, int output_zero_point) {
    int ksize = 3; // 3x3 kernel
    int padded_size = inside + 2*padding;
    int conv_out = ((padded_size + 0 - ksize) / stride) + 1; // conv output size
    int pool_size = 2, pool_stride = 2;
    int oside = ((conv_out - pool_size) / pool_stride) + 1; // pooled output size
   
    // final accelerator output (after pooling)
    uint8_t *poutputs = (int8_t *)calloc((size_t)oside * oside * c_out, sizeof(int8_t));
    if (!poutputs) { fprintf(stderr, "OOM poutputs\n"); exit(1); }

    // Optional writeback buffer (not required if accelerator returns via POOL_WB_INT)
    // keep for potential use / debugging
    int wb_elems = conv_out * conv_out;

    // Allocate padded input and copy original into center
    int8_t* padded_inputs = (int8_t*)calloc((size_t)c_in * padded_size * padded_size, sizeof(int8_t));
    if (!padded_inputs) { fprintf(stderr,"OOM padded_inputs\n"); exit(1); }

    for (int m = 0; m < c_in; m++) {
        for (int y = 0; y < inside; y++) {
            for (int x = 0; x < inside; x++) {
                padded_inputs[m * padded_size * padded_size + (y + padding) * padded_size + (x + padding)] =
                    pInputs[m * inside * inside + y * inside + x];
            }
        }
    }

    // number of 4-channel groups (ceil)
    int chn_num = (c_out + 3) / 4;

    // number of tiles along each axis (we step tiles by pool_stride=2)
    int tile = oside; // equals 1 + (padded_size-4)/2 for our config

    printf("tile: %d\n", tile);
    int t1 = record();

    // Let accelerator know dimensions (use padded_size so internal memory/layouts align)
    L_SCNN(c_in, padded_size, conv_out * conv_out, 0);

    for (int g = 0; g < chn_num; g++) {
        // set mode for each of the 4 hardware channels 
        L_MODE(0,1, conv_bias[g*4+0], 1);
        L_MODE(0,2, conv_bias[g*4+1], 1);
        L_MODE(0,3, conv_bias[g*4+2], 1);
        L_MODE(0,4, conv_bias[g*4+3], 1);
        // pack up to 4 output-channel filters into the contiguous buffer accelerator expects
        int packed_len = 9 * 4 * c_in;
        int8_t *packed_filt = (int8_t*)malloc((size_t)packed_len * sizeof(int8_t));
        if (!packed_filt) { fprintf(stderr, "OOM packed_filt\n"); exit(1); }

        // Fill packed_filt. If the global out_ch >= c_out, fill zeros for that channel.
        for (int m_in = 0; m_in < c_in; m_in++) {
            for (int kidx = 0; kidx < 9; kidx++) {
                for (int ch4 = 0; ch4 < 4; ch4++) {
                    int out_ch = g*4 + ch4; // global output channel index
                    int8_t val = 0;
                    if (out_ch < c_out) {
                        val = pFilters[(m_in*9 + kidx)*c_out + out_ch];
                    } else {
                        val = 0; // padding for missing channels
                    }
                    packed_filt[(m_in*9 + kidx)*4 + ch4] = val;
                }
            }
        }

        // For each tile (tile rows m, tile cols n), call accelerator on the 4x4 patch starting at (m*2, n*2)
        for (int m = 0; m < tile; m++) {
            for (int n = 0; n < tile; n++) {
                // input pointer: start of 4x4 patch at row = m*pool_stride, col = n*pool_stride
                // linear offset in padded_inputs (channel interleaving / accelerator expectations assumed same as before)
                int row_offset = m * pool_stride;
                int col_offset = n * pool_stride;
                int8_t *inptr = padded_inputs + (row_offset * padded_size + col_offset); // assumes inner-most dimension is width

                // Call accelerator for this 4-channel group and this tile's input patch
                SCNN4x4(packed_filt, inptr);
                ///-----RELU-----///
                for (int ch4 = 0; ch4 < c_out; ch4++) {
                     RELU(NULL, 16);
                }
                // After SCNN4x4, perform pooling via accelerator's POOL/POOL_WB_INT interface.
                // Request max-pool of 2x2
                POOL(2);

                // Retrieve pooled outputs for up to 4 channels in this group
                int pooled_val;
                for (int ch4 = 0; ch4 < 4; ch4++) {
                    int global_ch = g*4 + ch4;
                    if (global_ch >= c_out) {
                        // nothing to write for non-existent channel
                        // but we still need to call POOL_WB_INT only for valid indices the accelerator expects.
                        // accelerator's POOL_WB_INT takes index 1..4 — only call when channel exists.
                        break;
                    }
                    POOL_WB_INT(&pooled_val, ch4 + 1);
                    //-----REQUANTIZE()-----//
                    uint8_t qval = requantize(
                        pooled_val,
                        requant_mult[global_ch],  // (s_x * s_w) / s_y
                        output_zero_point,
                        0, 255
                    );
                    // write pooled value into final output at [global_ch, m, n]
                    poutputs[global_ch * oside * oside + m * oside + n] = qval;
                }
            }
        }

        free(packed_filt);
    }

    int t2 = record();
    printf("Accelerator cycle count: %d\n", t2 - t1);

    // cleanup
    free(padded_inputs);

    return poutputs;
}



//-----------------------------------------------------------------------------
//
// --------------------------------------FLATTEN-------------------------------
//
// -----------------------------------------------------------------------------
uint8_t* flatten_int8(
    const int8_t *input,
    int channels,
    int height,
    int width
) {
    int size = channels * height * width;
    uint8_t *output = malloc(size * sizeof(uint8_t));
    if (!output) {
        fprintf(stderr, "OOM in flatten_int8\n");
        exit(1);
    }

    memcpy(output, input, size * sizeof(uint8_t));
    return output;
}

//-----------------------------------------------------------------------------//
//
// -----------------------------FULLY CONNECTED--------------------------------//
//
// ----------------------------------------------------------------------------//
// Fully connected layer
// Inputs:
//   input: pointer to flattened input array
//   in_size: number of input neurons
//   out_size: number of output neurons
//   weights: pointer to weight matrix (row-major: out_size x in_size)
//   bias: pointer to bias array (size out_size)
// Returns: pointer to output array of size out_size (dynamically allocated)
// int8_t* linear_layer_int8(
//     const int8_t *input,
//     int in_size,
//     int out_size,
//     const int8_t *weights,
//     int *bias,
//     const float *requant_mult,
//     int output_zero_point
// ) {
//     int8_t *output = (int8_t*)malloc(out_size * sizeof(int8_t));
//     if (!output) {
//         fprintf(stderr, "OOM linear_layer_int8\n");
//         exit(1);
//     }

//     for (int o = 0; o < out_size; o++) {
//         int32_t acc = 0;

//         // MAC: INT8 × INT8 → INT32
//         for (int i = 0; i < in_size; i++) {
//             acc += (int32_t)input[i] *
//                    (int32_t)weights[o * in_size + i];
//         }

//         // bias (INT32)
//         acc += bias[o];

//         // ReLU (optional)
//         if (acc < 0) acc = 0;

//         // Requantize (float, مثل Conv)
//         float yf = acc * requant_mult[o];
//         int y = (int)(yf + 0.5f) + output_zero_point;

//         // clamp to UINT8
//         if (y < 0)   y = 0;
//         if (y > 255) y = 255;

//         output[o] = (int8_t)y;
//     }

//     return output;
// }
uint8_t* linear_layer_q(
    const uint8_t* input,
    int in_size,
    int out_size,
    int8_t* weights,
    int32_t* bias,
    float* requant_mult,      // per-output-channel
    int output_zero_point
) {
    uint8_t* output = (uint8_t*)calloc(out_size, sizeof(uint8_t));
    if (!output) {
        fprintf(stderr, "OOM linear output\n");
        exit(1);
    }
   int input_zero_point =0;
    for (int o = 0; o < out_size; o++) {
        int32_t sum = 0;

        for (int i = 0; i < in_size; i++) {
            int32_t x = (int32_t)input[i] - input_zero_point;
            int32_t w = (int32_t)weights[o * in_size + i];
            sum += x * w;
        }

        sum += bias[o]; // bias already in INT32 domain

        // Requantize
        // output[o] = requantize(
        //     (int) sum,
        //     requant_mult[o],   // = (s_x * s_w[o]) / s_y
        //     output_zero_point,
        //     0,                 // qmin for UINT8
        //     255                // qmax for UINT8
        // );
        float yf = sum * requant_mult[o];

        // round-to-nearest
        int32_t y = (int32_t)(yf + (yf >= 0 ? 0.5f : -0.5f));

        y += output_zero_point;

        if (y < 0) y = 0;
        if (y > 255) y = 255;
        if( 0<= y<=255) y=y;

        output[o] = (uint8_t)y;

        // if (output[o] < output_zero_point)  output[o] =output_zero_point;
        // if (output[o] > output_zero_point)  output[o] =output[o];
    }

    return output;
}




//-----------------------------------------------------------------------------
// load input_matrix
// -----------------------------------------------------------------------------
// #define MAX_C_OUT 32

// void load_filters(int8_t *dst, int8_t K[][MAX_C_OUT][9], int c_in, int c_out)
// {
//     int idx = 0;

//     for (int m = 0; m < c_in; m++) {          // input channel
//         for (int k = 0; k < 9; k++) {         // 3x3 kernel
//             for (int oc = 0; oc < c_out; oc++) { // output channel
//                 dst[idx++] = K[m][oc][k];
//             }
//         }
//     }
     
// }




// -----------------------------------------------------------------------------//
//
// ---------------Main testbench (same structure as example)--------------------//
//
// -----------------------------------------------------------------------------//

// #define QMIN (0)
// #define QMAX (255)
// static inline int clamp_int(int x, int min, int max)
// {
//     if (x < min) return min;
//     if (x > max) return max;
//     return x;
// }

int main() {
    int inside = 28;      // input size
    int c_in1 = 1;       // first conv input channels
    int c_out1 = 8;      // first conv output channels
    int stride1 = 1, padding1 = 1;


    //-----------------------------------------------------------------------------
    // load input_matrix
    // -----------------------------------------------------------------------------
    // Allocate input matrix
    int8_t *input_matrix = (int8_t*)malloc(inside*inside*c_in1*sizeof(int8_t));
    float scale= 0.007870171219110489;
    int zero_point =0;
    int ip[784]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,84,185,159,151,60,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,254,254,254,254,241,198,198,198,198,198,198,198,198,170,52,0,0,0,0,0,0,0,0,0,0,0,0,67,114,72,114,163,227,254,225,254,254,254,250,229,254,254,140,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,17,66,14,67,67,67,59,21,236,254,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,253,209,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,233,255,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,254,238,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,249,254,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,187,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,205,248,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,254,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,251,240,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,221,254,166,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,203,254,219,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,254,254,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,254,115,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,242,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,254,219,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,254,207,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};
    for (int i = 0; i < 784; i++) {
    // Normalize to [0,1]
        float x = ip[i] / 255.0f;

        // Quantization
        float qf = x / scale + zero_point;

        int q;
        if (qf >= 0.0f)
            q = (int)(qf + 0.5f);
        else
            q = (int)(qf - 0.5f);

        // Clamp to [0,255]
        if (q < -128)   q = -128;
        if (-128<= q <= 127)  q=q;
        if (q > 127) q = 127;

        input_matrix[i] = (uint8_t)q;

        printf("x=%d  x_norm=%.4f  q=%u\n",
            ip[i], x, input_matrix[i]);
    }

//-----------------------------------------------------------------------------//
//
// --------------------- FIRST CONVOLUTION LAYER (1→8)-------------------------//
//
// -----------------------------------------------------------------------------//

    int8_t *filter1 = (int8_t*)malloc(3*3*c_in1*c_out1*sizeof(int8_t));
    int bias1[8] = {0, 0, 0, 0,0, 0, 0, 0};
    int8_t K1[8][1][9] = {
        { {106, 111,   6,  19,  64, 117, -128, -50,  85} },  // out 0
        { { 24, 120, 127,  19,  22, -34,   90,  58,  36}},  // out 1
        { { 48,   7, -28,  65, 103, -53,  105, 127, -107} },  // out 2
        { { 60,  59, 103, 127,   4, -30,   78, -38,  -11} },  // out 3
        { { 94,  14,  98,  10,  81,  -9,  -87, -128, -81}},  // out 4
        { { 83,  41,  76, -66,  54,  86,  -67, -128, -32} },  // out 5
        { { 88, -86, -120, 80, -13, -128,  95,  91,  -1} },  // out 6
        { { -4,  80,  44,  28, 108, 127, 104,  21, -63} }   // out 7
    };
    int idx = 0;
    // for (int in = 0; in < c_in1; in++) {
        for (int i = 0; i < 9; i++) {          
            for (int out = 0; out < 8; out++) { 
                filter1[idx++] = K1[out][0][i];
            }
        }
    //}
    float requant_mult1[8] = {
        0.0021570551173471847f,
        0.0013519407901308612f,
        0.0016110192503369222f,
        0.0017258973034171930f,
        0.0024914301320889906f,
        0.0021759635816515337f,
        0.0020346772544926320f,
        0.0016554155417697570f
    };
    int output_zero_point =0;
    // First layer convolution
    int8_t *out1 = RVSCNN_conv_mp(inside, c_in1, c_out1, input_matrix, filter1, stride1, padding1, bias1,requant_mult1, output_zero_point);
    
    // Compute first layer output size after pooling
    int conv_out1 = ((inside + 2*padding1 - 3)/stride1)+1;
    int oside1 = ((conv_out1 - 2)/2)+1;
    
    printf("L1 output side = %d (expected 14)\n", oside1);
    
    printf("First Conv Output (1→8):\n");
    for(int c = 0; c < c_out1; c++) {
        printf("Channel %d:\n", c);
        for(int i = 0; i < oside1; i++){
            for(int j = 0; j < oside1; j++){
                printf("%4d ", out1[c*oside1*oside1 + i*oside1 + j]);
            }
            printf("\n");
        }
    }

    //-----------------------------------------------------------------------------//
    //
    // --------------------- SECOND CONVOLUTION LAYER (8→16)-----------------------//
    //
    // ----------------------------------------------------------------------------//
    int c_in2 = c_out1;
    int c_out2 = 16;
    int stride2 = 1, padding2 = 1;

    
    int8_t *filter2 = (int8_t*)malloc(3*3*c_in2*c_out2*sizeof(int8_t));
    int bias2[16] = { 0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0,0, 0, 0, 0 };

    int8_t K2[16][8][9] = {
        { // k = 0
            {88, 19, -7, 32, 49, 16, -66, -76, -68},
            {5, 12, 36, 0, 0, -29, 3, -68, 2},
            {24, 9, 74, -97, -88, -106, 8, -29, 9},
            {19, 27, 49, -63, 31, 25, -27, -69, -62},
            {77, 42, 97, 35, 116, 102, -43, -109, -99},
            {94, 52, 41, 69, 48, 18, -51, -118, -72},
            {86, -26, -72, -74, -128, -93, -6, 24, 35},
            {31, 21, 33, -71, 8, 5, 5, -66, -36},
        },
        { // k = 1
            {80, -31, -54, 64, 47, -41, 13, 33, -3},
            {-18, -7, -44, 24, 20, -3, -15, 19, 6},
            {29, 45, 5, -41, 23, 34, -4, -18, 4},
            {-7, 17, -54, -41, 15, -9, 1, 28, 26},
            {-71, -127, -25, -32, -76, -86, -67, -25, 22},
            {36, -82, 9, 37, 57, -44, -25, 15, 30},
            {97, 116, 51, -36, 50, 106, -57, -11, -25},
            {11, -34, -28, 9, 1, 8, 23, 8, 6},
        },
        { // k = 2
            {50, 2, -46, 19, 49, -2, -41, 38, -8},
            {-1, 14, 3, -25, -5, -31, -2, 17, -14},
            {28, 16, -9, -84, 22, 34, -20, -41, 15},
            {3, 5, 8, -15, -3, -17, -30, 16, -10},
            {-97, -116, -29, -16, -55, -86, 5, -48, -1},
            {35, -1, -40, 31, 6, -24, -8, 12, 9},
            {-9, 44, 58, -67, 58, 115, -127, -49, 11},
            {31, 11, 17, -27, -16, -31, -32, 19, 28},
        },
        { // k = 3
            {-16, 5, -13, 11, 0, -6, -20, 18, -34},
            {-40, 10, 35, -32, 9, 44, -13, 38, 5},
            {4, -5, 37, -56, 18, 29, -31, 23, 38},
            {-21, 2, 53, -11, -6, 21, -36, 12, 6},
            {3, -100, -46, -14, -127, -63, -69, -79, -21},
            {-6, -46, 2, -23, -31, -42, -10, -20, -31},
            {23, -85, -28, -32, -98, 32, -49, -28, 51},
            {-39, 18, 35, -15, 9, 42, -23, 28, 2},
        },
        { // k = 4
        {35, -24, -3, -55, -8, 17, -10, 15, 2},
        {-14, -39, 14, -34, 26, 44, 37, 25, 6},
        {-44, -75, -19, -60, 29, 21, 38, 33, -39},
        {-9, -48, 5, -83, 6, 53, 23, 60, 24},
        {108, 30, -79, -20, -18, 64, -24, 82, 37},
        {38, -49, -7, -46, -44, 24, -4, 34, -7},
        {0, -27, -105, -96, -128, -94, -16, -23, -25},
        {-6, -62, 6, -86, 60, 59, 64, 45, -38},
        },
        { // k = 5
            {25, -64, -51, -45, -58, -96, -21, 7, 54},
            {47, -50, -33, 24, -51, -26, 25, -41, 18},
            {61, -7, -34, 50, 3, 48, -22, -28, -9},
            {-2, 10, -53, 26, 4, -46, 38, 14, 15},
            {-96, -32, 2, -93, -82, -51, -4, 105, 87},
            {-39, -49, 22, -14, -82, -33, 44, 65, 66},
            {64, 16, -42, 50, 34, 127, -2, 4, 21},
            {11, -9, -46, 24, -37, -47, 21, 34, 22},
        },
        { // k = 6
            {-31, -1, 47, -30, -13, -6, 18, -63, -19},
            {30, 31, 6, 53, 23, -29, -5, -115, -12},
            {61, 28, -38, 94, 59, -93, 43, -37, -55},
            {50, 26, -6, 34, 52, -23, 23, -77, -27},
            {-92, 19, -5, -47, -41, -54, 51, -63, -50},
            {-64, 11, 17, -54, -10, 31, -4, -76, 50},
            {-39, -27, 22, -6, 93, -34, 26, 23, -67},
            {64, 57, -14, 81, -17, -22, -6, -128, -50},
        },
        { // k = 7
            {10, -44, 27, 77, 43, -78, -22, 92, 63},
            {16, 17, -11, -7, -12, -48, 11, 1, -20},
            {-9, 13, -37, -1, 63, -11, -89, -65, 6},
            {-10, -21, 4, -43, -27, -48, -25, -49, -18},
            {-68, -8, 56, 13, -109, -61, 24, 8, 22},
            {-52, -11, 56, 85, 18, -61, 59, 94, 22},
            {42, 19, -128, 12, 110, 84, -85, -31, 69},
            {-1, 0, 21, -2, -30, -15, -15, -2, 18},
        },
        { // k = 8
            {0, 38, -28, 19, 14, -37, 18, -14, -31},
            {11, 38, 11, -13, 39, -25, 9, 9, -64},
            {-17, 29, 31, -40, 16, 21, -21, 26, 0},
            {15, 26, 33, -35, 35, -11, -5, 32, -18},
            {-65, -72, 15, -99, -96, -13, -35, -13, 0},
            {2, -22, -3, -3, 2, -57, 16, 8, -33},
            {-115, -30, 25, -128, 33, 69, -50, 30, 40},
            {23, 26, 18, 1, 32, -1, -10, 17, -52},
        },
        { // k = 9
            {-34, -31, -25, 20, -2, -13, 3, 2, -28},
            {-40, 7, 21, 22, 38, 32, 27, 36, -9},
            {-47, -3, 35, -11, 49, 9, -21, 15, -36},
            {-47, 1, 26, -22, 35, 45, 16, 47, 5},
            {-59, -128, -9, -67, -41, 1, -54, -20, -24},
            {-43, -37, -5, -23, -21, -29, -23, -33, -12},
            {-19, -69, -98, -86, -79, -24, -59, 3, 6},
            {-24, 7, 41, 10, 19, 27, 8, 10, -2},
        },
        { // k = 10
            {-33, -63, -62, 37, 39, 38, 24, 0, -23},
            {-22, 3, -17, 40, 45, 11, 14, 15, 19},
            {11, 5, -15, 13, -11, -11, -26, -8, 10},
            {22, -9, -35, 4, 43, 28, 5, 0, -7},
            {-54, -128, -41, 22, 39, 71, -24, -37, 54},
            {-52, -71, -56, 4, 41, 63, 6, -5, 1},
            {-7, 3, -20, -29, -91, -54, -60, -43, -69},
            {-1, -23, 6, 23, 21, 21, 18, 23, 1},
        },
        { // k = 11
            {29, 13, 47, -46, -8, 25, -5, 29, -14},
            {29, 28, 3, -39, -19, 38, -23, 10, 31},
            {-32, -8, 23, -33, -58, 28, -11, -31, 34},
            {9, 35, 1, -29, -3, 19, 6, 1, 12},
            {20, 44, -17, -6, -50, -104, -21, -76, -60},
            {27, 33, 5, -31, 2, -9, -27, -22, -18},
            {-87, -65, -14, -45, -128, -13, -33, -78, -7},
            {24, 34, 16, -20, -13, 28, -25, 10, 24},
        },
        { // k = 12
            {11, 55, -64, -23, 53, 17, -66, 51, 35},
            {19, 30, -22, -36, 19, -2, -68, -6, -8},
            {-9, 51, 33, -83, 23, 40, -92, -55, -11},
            {-23, 9, 6, -44, 22, -11, -64, 1, -11},
            {-5, -67, -41, -2, -45, -78, -30, -32, -19},
            {20, 23, -51, 15, 36, -17, -10, 45, 27},
            {-19, 95, 95, -53, 47, 127, -21, -41, 62},
            {-4, 31, -13, -32, 11, -17, -79, -10, -9},
        },
        { // k = 13
            {-98, -87, -112, -10, 40, 23, 63, 26, 72},
            {-60, -76, -30, -5, 2, -6, 30, -13, 11},
            {-81, -66, -66, 47, 28, -1, 37, -23, 8},
            {-98, -99, -54, 3, 35, 15, 48, -6, 18},
            {-52, -128, -78, 31, -6, -66, 65, 35, 54},
            {-80, -113, -40, 21, 18, -5, 40, 21, 59},
            {-55, 6, 62, 28, 55, 60, 24, 20, -25},
            {-111, -101, -75, 15, 36, 5, 4, 28, 19},
        },
        { // k = 14
            {25, 46, 34, 22, -17, -19, 12, -14, 9},
            {23, 30, 9, 0, 9, 26, 18, 4, 0},
            {16, 5, 24, 6, -1, -17, -52, -18, 20},
            {-5, 32, -4, 18, 5, 22, 8, -28, 3},
            {-94, -76, -90, -16, -113, -90, 34, -23, -53},
            {-43, -41, 7, 22, 1, 8, 41, -25, 4},
            {-29, -63, -25, -70, -91, 23, -123, -128, -7},
            {37, 20, 8, 12, 2, 13, 1, -5, -14},
        },
        { // k = 15
            {-50, -6, -28, 86, 57, 49, 6, 8, -26},
            {27, -2, -6, 28, 26, 23, -54, -59, -50},
            {22, 22, 4, 24, -28, -37, -113, -46, -53},
            {6, -3, 14, 32, 46, 51, -30, -54, -44},
            {-116, -78, -68, 59, 127, 112, 52, 125, 89},
            {-60, -39, -70, 45, 49, 68, 24, 11, 36},
            {83, 9, -17, -55, -95, -81, 2, -48, -114},
            {-9, 18, -9, 26, 17, 57, -66, -6, -56},
        },
        };
     int idx1 = 0;
     for (int in = 0; in < 8; in++) {
        for (int i = 0; i < 9; i++) {         
            for (int out = 0; out < 16; out++) { 
                filter2[idx1++] = K2[out][in][i];
            }
        }
    }
    float requant_mult2[16] = {
      0.0013303808397372868f,
       0.001740413360527118f,
        0.001602644429314104f,
         0.0020878168241033057f,
          0.002094378551917773f,
           0.0013603047374895509f, 
           0.0013197166726330609f, 
           0.0014403455574581267f, 
           0.002010679694746616f, 
           0.002084023340658359f, 
           0.0017364663054302433f, 
           0.0021943576519578203f, 
           0.0019738493568452996f, 
           0.0015939745515910866f, 
           0.0015606144057387047f, 
           0.0015222715847976363f
    };

    // Second layer convolution
    int8_t *out2 = RVSCNN_conv_mp(oside1, c_in2, c_out2, out1, filter2, stride2, padding2, bias2,requant_mult2, output_zero_point);

    // Compute second layer output size
    int conv_out2 = ((oside1 + 2*padding2 - 3)/stride2)+1;
    int oside2 = ((conv_out2 - 2)/2)+1;
    printf("L1 output side = %d (expected 7)\n", oside2);

    printf("Second Conv Output (8→16):\n");
    for(int c = 0; c < c_out2; c++) {
        printf("Channel %d:\n", c);
        for(int i = 0; i < oside2; i++){
            for(int j = 0; j < oside2; j++){
                printf("%4d   ", out2[c*oside2*oside2 + i*oside2 + j]);
            }
            printf("\n");
        }
    }
    

    uint8_t *flat = flatten_int8(out2, c_out2, oside2, oside2);

    int flatten_size = c_out2 * oside2 * oside2; 
    printf(" Flatten size (size=%d):\n", flatten_size);
    printf("Flattened Output (size = %d):\n", c_out2*oside2*oside2);
    for (int i = 0; i < c_out2*oside2*oside2; i++) {
        printf("%4d, ", flat[i]);
    }
    printf("\n");

    //-----------------------------------------------------------------------------//
    //
    // --------------------------- FULLY CONNECTED --------------------------------//
    //                             LAYER 4 784->64
    // ----------------------------------------------------------------------------//
    
    int output_zero_pointfc1 =70;
    int fc1_in = flatten_size; 
    int fc1_out = 64;
    int32_t* fc1_bias = (int32_t*)malloc(fc1_out*sizeof(int32_t));
    int8_t* fc1_output = (int8_t*)malloc(64 * sizeof(int8_t));

    
    for (int i = 0; i < 64; i++)
        fc1_bias[i] = 0;

    int8_t FC1[784][64] = {
  {6, 6, -4, -7, 1, -12, 6, -10, 13, 3, 0, 5, -4, 27, 2, 7, -96, 4, 18, -7, -11, -18, 14, 3, 10, 8, -6, -18, -24, -1, 14, -27, -2, -11, -2, -24, -77, -16, -2, 12, 56, -17, 99, -11, 1, 4, 4, 12, 13, -3, 14, 57, -90, 10, -1, 13, -84, -86, -4, 43, 27, 12, 1, 11},
  {-12, 13, -3, 14, -26, -13, -19, -12, -2, 1, -25, -22, 25, 9, -19, -9, 46, -6, 1, 3, -12, 3, 6, -12, -10, -7, 3, 30, 4, 39, 36, 49, 3, 4, -33, -12, -52, 15, 7, 13, -19, -11, -11, 9, 38, -7, 6, -57, 6, -5, 14, -45, 103, 48, 12, -8, -20, -72, 5, -6, -35, 20, 23, 21},
  {1, 28, 23, -8, -76, -21, -20, 6, -1, 34, -8, -5, -13, 31, -20, -30, -37, -16, -34, -9, -3, -3, 6, -26, -23, -26, 5, 51, -99, 35, -27, 77, 33, 30, -44, 15, -109, -40, 28, -2, -90, 51, 61, -23, -11, -20, -7, -29, 6, 9, -23, -73, 75, 49, -18, 37, 99, -69, -3, -75, -6, 30, 27, -20},
  {11, -20, 19, -34, -53, -58, 23, 32, 13, 15, -18, 6, -47, 28, -21, 1, -43, -7, -45, -15, -44, -22, 6, 15, -2, 28, -22, 14, 16, 17, -58, -57, 50, 16, -3, 17, -60, -51, 17, 8, -3, 39, 8, -10, 5, -36, 30, 30, 26, -3, -22, -4, 81, 61, 25, 63, 26, -126, 2, 51, -84, 41, 7, 16},
  {8, -14, -18, -27, -82, -36, 31, 27, 3, -3, -11, -3, -68, -2, -20, -18, 47, 3, -52, -24, -9, -17, -9, 25, 7, 6, 5, -21, -48, -21, -20, 2, 47, 33, 2, -13, -76, -47, 32, 29, -44, -7, -39, 2, -21, 18, 38, 20, 0, 29, 16, -35, 81, 16, -33, 40, 4, -97, 8, -76, -54, -7, -27, 62},
  {-20, -12, 7, -12, 86, -3, 22, -3, 27, 7, -14, -15, -1, 22, -16, -28, 13, 2, -14, -6, -14, -6, -6, 21, 5, 21, -19, -16, 8, 4, -30, -97, 27, 10, -30, 14, 38, 2, 11, 36, 93, -17, -65, 12, 41, 3, 11, 44, -4, 14, -6, -34, -103, 4, 4, 38, -114, -126, 12, 40, -31, 13, 8, 19},
  {4, -1, 4, -18, 3, 13, 10, -30, 3, 24, -4, -7, -16, 26, -5, -30, 59, 3, -5, -4, -5, 19, -7, 13, -3, 1, 9, 25, -1, 16, 23, -70, 16, 5, -19, 10, -45, -10, 1, 6, -3, 11, 78, -7, -41, -11, -14, 7, -11, 20, 1, 2, 26, 34, 3, 25, 16, -9, 37, 24, -81, 17, 24, -4},
  {44, -12, 26, -27, -47, 26, 14, -29, 2, -5, 9, 4, -18, 27, -8, 31, -97, -20, -21, -16, -1, -46, 32, 50, -15, -22, -38, 51, -76, -8, -41, -93, 25, -21, 5, 24, -63, -54, -21, -3, -37, 80, -47, -31, -24, -43, 24, 7, -24, -7, 43, 40, -11, 4, 18, 61, -126, -24, 1, 23, -28, 64, 7, -25},
  {8, -18, -8, 10, -97, 28, -16, -31, -10, -12, 7, 5, 3, 16, 44, 55, -8, -28, 36, -22, 17, -38, 27, 112, -26, -33, -8, 51, -57, -9, 10, -91, 55, -106, -30, -67, 27, -53, -38, -5, -52, 60, 36, -41, 47, -45, 45, -18, -31, -12, 74, 72, -23, 48, 16, -23, -75, 27, 15, 13, -32, 68, -19, 14},
  {-73, -25, 15, -43, 39, 33, -6, -106, 3, -27, 18, -24, 15, 50, 66, 64, -106, 2, 63, 33, 25, -70, 10, 78, -13, -1, 0, -5, 59, -1, 35, -126, 2, -127, 46, -112, 44, -37, -43, -33, 13, -31, -69, -44, 4, -25, 47, -9, -24, -66, 76, -68, 16, 55, 47, 5, -83, 9, 29, 11, -114, 57, -5, 17},
  {-43, -58, -1, -3, -93, 25, 55, -50, 49, -14, 26, 19, -1, 45, 74, -17, -72, -35, 31, 8, 31, -14, -60, 27, -7, 36, -4, -17, -32, -9, 26, -121, 4, -105, 6, 23, -112, -41, -17, -7, -4, 4, 37, -25, 39, 18, -10, -13, -27, -44, -10, -31, 56, 67, 23, 15, -96, 1, 80, 32, 12, 25, 14, -35},
  {-22, 3, -42, -12, -67, -16, 34, 4, -14, -18, 2, 43, 67, 6, 6, -36, 2, -52, 1, -25, 5, 63, -14, 15, -1, -11, 16, -9, 19, 20, 11, 44, 15, -66, -14, 50, 27, 7, 15, 19, -65, -30, 39, 4, 87, -6, -3, -70, -24, 14, -29, -12, 89, 25, 15, 2, -83, 31, 41, 30, 72, -28, 2, -41},
  {-42, 4, -75, -13, -13, -14, -3, 24, 59, -9, -8, 1, 43, 13, 19, -53, -114, -57, -11, -16, -4, 127, 9, 15, -23, -16, -13, 49, -115, 55, 77, -72, -10, -26, -37, 42, 61, 73, 48, 38, -9, -16, 37, 41, 65, 55, -52, -60, 1, 98, -52, 25, -69, -48, -28, -90, -38, 80, 20, 82, -49, -50, -8, -32},
  {-26, -13, -44, 1, 84, -28, -10, 46, 95, -30, 19, -10, -51, 20, 50, -7, -25, 4, 14, 27, 26, 104, 9, 16, -19, 1, -3, -19, 37, -21, 77, 38, -29, 4, -15, 31, 6, 65, 38, 44, -52, -37, 60, 49, 14, 93, -48, 21, -16, 55, -38, -20, 39, -98, -55, -58, -44, -32, 11, 56, 16, -72, -10, 9},
  {32, 23, 50, -49, 67, 3, 27, -11, -4, -30, -12, 20, -21, -41, -18, -75, 69, -32, 24, -25, 7, -10, 32, 27, -5, -18, -30, 3, 70, 5, -77, -54, 8, 28, 24, -6, 6, 29, -26, -26, 0, -2, 17, 8, 42, 2, 38, 15, 27, -4, 33, -24, 86, -26, 27, -20, -106, -8, -55, -64, -55, -2, 21, 37},
  {-2, 7, 60, -73, 0, -1, -4, -29, 0, -11, 24, -9, -44, 12, 9, -32, 41, -47, 37, -2, 23, -30, -10, 4, -12, 17, -7, 2, -118, 4, -16, -21, -41, 33, 35, -27, 72, -53, -26, -16, -92, -35, -42, -56, -12, -17, -3, 15, 7, -40, -6, -45, 66, 7, 45, -23, 62, 43, -21, 17, -25, 50, 11, -1},
  {-24, -9, 10, -105, -101, -13, 16, -25, -10, -17, -4, -10, -14, 20, 4, 7, -44, -27, 20, 15, -10, -48, 0, 26, -21, 10, 18, 8, -23, 20, -10, 17, -12, 1, 9, -6, -19, -31, -25, -11, -76, -13, 82, -47, 5, 0, 18, -5, -27, -39, -38, 59, 51, 25, 33, -32, -9, 26, 18, -28, -31, 53, 25, 30},
  {-20, -33, -101, -13, 33, 1, -2, -43, 15, -19, 4, 50, 21, 43, 26, -21, -53, -40, 27, 26, 19, -19, -31, 72, -28, -7, -14, 13, -26, -9, -6, 0, 2, -92, -38, -26, -19, -19, -3, 8, 61, 2, -1, -54, -30, -32, 8, -46, -75, -1, -19, -41, -55, 62, 0, -4, -53, 11, 99, -97, 5, 47, -13, -21},
  {-23, -11, -126, 19, 14, 37, -26, 28, 34, -41, 6, 41, 40, 50, 39, -112, 59, 32, -9, -5, -6, 65, -68, 10, -7, 31, 0, 23, -32, -2, -17, -47, -13, -99, -58, 5, -55, 41, -6, 17, -50, -31, 15, -31, -46, 20, -56, -36, -60, 13, -50, -62, 3, 11, -5, 13, -4, 51, 100, -11, -79, 3, 33, -19},
  {-10, 25, -53, 33, 42, 66, -29, 43, 57, -34, -10, -9, -4, 8, 32, -37, -51, 29, -18, -17, 7, 115, 7, -35, -2, 23, -5, 93, -66, 25, 5, 32, -19, -27, -122, -22, -40, 39, 36, 47, -48, -14, -96, 3, 0, 101, -82, -35, -14, 40, -88, -87, 82, -50, -45, 3, -60, -4, 57, -96, 68, -15, 42, -22},
  {3, 7, -17, 4, -50, 14, -5, 70, 12, -49, 24, -16, -68, -23, 20, 26, -87, 52, -38, -6, 25, 33, -21, -5, -10, 12, 13, 37, 66, 1, -5, 39, -3, -9, -20, -22, -89, 43, 22, 37, -9, -7, -83, 31, -1, 41, -20, -40, -23, 22, -69, 72, -67, -48, -34, -40, 65, -110, -10, -80, -36, -27, -7, 39},
  {28, -6, 3, -113, -31, -6, -33, -6, 16, -7, 18, 30, -32, 26, 17, -60, 85, -22, 10, -66, 36, 6, 12, 7, 10, -23, -22, -55, 6, 7, -35, -74, -31, -29, 6, -23, 82, 25, 4, -39, 77, 23, -54, 10, 17, 37, -10, 20, -13, 8, 62, 42, 77, -37, 6, -21, -2, -29, -28, 17, -83, -32, -25, 17},
  {14, -35, 33, 15, -118, -49, -16, -10, 15, -31, 54, 2, -15, 30, 23, -46, 41, 25, 32, -24, 31, -3, 24, 10, -48, 19, -3, -12, -7, 3, 32, 43, -93, 7, 25, 1, 56, -2, 5, -43, 23, 32, 24, 33, -20, 55, 3, -13, -43, -12, 22, -26, -71, 42, 27, -33, -63, 60, 8, -45, 0, 24, -7, -27},
  {19, -89, 52, -12, -75, -58, -13, 22, 8, -37, 56, 33, -16, 44, 29, -5, 58, -16, 22, -26, 35, 0, -7, -7, -75, 34, 25, -35, -38, -32, 71, -87, -78, -15, 14, -3, -85, -25, 11, -46, 83, 39, -84, 36, -17, 35, 12, -30, -50, -17, -1, -81, -111, 48, 38, -21, 63, -47, 9, -73, -73, 31, 0, -13},
  {-35, -46, 57, -91, -80, -28, 20, 17, 9, -50, 31, 48, 1, 14, 46, -5, -101, 21, 19, -56, -2, 9, -27, -69, -37, 24, 38, -23, -83, -76, 65, -47, -17, 41, 43, 27, -91, 3, -12, -50, -10, 7, -94, -13, 38, 63, -7, 20, -24, -24, -24, -116, -90, -52, 12, -6, 114, -52, -24, -104, 15, -35, 29, -11},
  {-29, -15, 31, -104, -61, -13, -76, -30, 42, -51, -9, 5, -20, 45, 37, 14, 43, 94, 31, 31, 36, 12, 33, -33, -21, 22, -3, 26, -36, -22, 20, -76, -7, 19, 12, -23, -93, -43, -17, -16, -52, 3, 15, -19, 8, 29, 0, 10, 13, 11, -1, -38, 5, -31, 10, 6, -120, -79, 26, -68, -58, 0, 18, -8},
  {4, -19, 54, -107, -49, -29, -82, -42, 60, -8, -6, -33, -52, 25, 19, -6, -70, 33, 39, 65, 34, 23, 6, 4, -16, 30, -21, -4, -4, -7, -23, 56, -39, -8, 34, -25, -2, 4, 4, -11, -68, -30, -82, 65, -32, 122, 9, 24, 15, 17, -28, -27, -44, 4, 16, 6, -49, -5, -13, -84, -90, -10, 35, 6},
  {13, -45, -29, -51, 78, -101, -14, 3, 21, -4, 32, -10, -5, 1, 35, 12, 93, -4, -7, 38, 30, -16, -35, -4, -58, 57, 12, 5, -88, 13, -70, 55, -67, -49, 64, 44, -46, 22, -20, -24, -89, 32, 59, 51, -40, 66, 16, -3, 7, 2, -62, 6, -3, 21, 82, -48, 65, -56, 20, -78, 85, 23, 11, 12},
  {26, -8, -61, -87, -87, 17, -2, 15, 8, 2, -1, 42, -14, -5, 12, 10, 67, 12, -48, -28, 23, 38, -24, -34, -7, -43, -6, 18, -35, 19, 17, 37, 14, -18, -21, -3, -23, 12, 31, 17, 66, 38, 71, 28, -14, -31, 4, -17, 11, 14, 4, 25, 50, -4, -36, 27, 78, -98, -51, 21, -50, -54, -57, -2},
  {58, 10, -30, -71, 41, -8, -38, 20, 36, -8, 24, -10, 7, 1, 21, -62, -74, 6, -35, 0, 23, 10, -23, -6, -17, -24, 12, 30, -115, -36, 15, -87, 3, 15, 10, 9, 76, 9, 18, 14, 27, 4, 64, 48, -40, 34, -6, 7, 19, -1, 5, -8, 55, -37, -21, 6, -29, -23, 29, -9, -18, -1, -12, -25},
  {28, 9, 20, -50, 46, 8, 4, 54, 18, -10, 40, 15, -1, 15, 46, -128, -12, -35, -25, 16, 29, -19, -15, -52, -19, 30, 49, 16, -72, -48, 23, -89, -31, 27, 52, 32, -38, 31, -4, 1, -16, -33, 63, 40, -68, 44, -16, 35, -7, -23, -25, -14, 38, -21, 16, -22, -39, 59, 24, -91, 83, 2, 32, -22},
  {-17, 59, 49, -8, -81, 9, -12, 67, -41, 22, 28, 18, 10, 5, 31, -54, -34, -42, 27, -24, -16, -23, 42, -55, -5, -9, 70, 5, 15, -15, 48, -35, -47, 60, 72, 21, 77, 72, -25, -14, -7, -84, -47, 50, -3, 64, -4, 43, -17, -43, -58, 9, -53, -1, 49, -73, 12, 0, -24, 12, -50, -7, 47, 1},
  {-13, -4, 8, -88, -62, 20, -80, 14, 18, 6, -11, -10, -26, 55, 14, -25, 43, -60, 19, -90, 24, 4, 51, -16, -18, -27, 59, -5, -64, -2, -30, -97, -7, 0, -6, -23, 55, -10, 3, 12, 96, -33, -34, -14, -5, 14, 13, 23, -19, 16, -31, -3, 53, -21, -5, 5, -123, 22, 0, -81, -118, 33, -15, 16},
  {70, -32, -69, -47, -37, -11, -39, -27, -22, 4, -24, 11, 1, 23, -3, -31, 46, -111, 39, -97, 18, 4, 56, 71, -17, -42, 51, -15, -14, 15, 42, 82, -5, -82, -34, -87, 0, -15, 22, 33, -60, 27, -42, -18, -81, -77, 64, -43, -31, 3, 60, -12, -109, 15, -9, -1, 20, -106, -27, -27, -93, 46, -92, 26},
  {16, -4, -62, -41, 82, -27, 71, 21, -49, -18, -17, 43, 61, 4, -30, -20, -91, -76, -23, -82, -13, 10, 18, 23, -31, -36, 22, 7, -73, 53, -15, 2, -55, -83, -18, 45, -93, -14, 7, 21, 10, 49, 87, -41, -116, -108, 22, -68, -52, -23, 14, 78, 72, 36, 26, -28, 50, 42, 18, -88, 43, 50, -33, 26},
  {3, -30, 10, -22, -42, -5, -12, -5, -7, -1, -19, 33, 8, -41, -52, -54, 81, 3, 3, -4, -25, -9, 25, -1, 0, -31, -22, -11, -101, 28, -4, -36, 3, -1, -12, -29, -45, 25, 12, 6, -1, 10, 18, 22, -50, -19, 43, -27, 11, 12, -11, 51, -28, 25, -10, 17, -108, -86, -24, 75, 16, -24, -16, 29},
  {-28, 14, 6, -24, -65, -6, 4, 11, -9, -17, 23, 18, 1, -8, -26, -43, -19, -31, -19, -59, -4, -31, -26, -33, 29, 23, -19, -25, 52, 3, -8, 76, -40, -2, 11, -18, 37, 6, 5, 10, 50, -12, 14, -14, 34, -22, -22, 13, 8, -22, -36, -44, 11, 2, 11, 25, -103, 31, 2, -54, 29, -24, 13, -3},
  {-88, 15, -26, -71, -36, -22, 36, 31, 4, -47, 12, -8, 36, 3, -17, -34, -66, -29, -6, -61, -44, -11, -18, -39, 40, 51, -16, -27, -30, 15, -31, 1, -22, -23, 13, 37, 49, 31, -18, 11, -119, -43, -52, 3, 5, -9, -32, 19, -15, -54, -28, -55, 70, -4, 17, 27, -84, -11, -13, 10, -108, -21, 40, 1},
  {-93, 23, -14, -77, -53, -27, -1, -9, 6, -36, -14, -45, 46, 0, -46, -33, -109, -3, 13, -58, -36, -6, -9, -27, 40, 37, -16, -47, -54, 36, -35, 116, -15, -31, -1, 11, -120, 12, -5, 29, 37, -30, 55, 15, -63, -10, -5, 33, 13, -24, 0, -1, -111, -1, 12, 29, -18, 81, -35, 98, -60, -36, 33, 44},
  {-90, 5, -3, 15, -37, -15, -10, -25, -4, -45, -22, -32, 41, -26, -21, 3, -100, 7, 12, -52, -22, 5, 5, -9, 12, 11, -40, -41, -98, 37, -14, 114, 7, -29, -19, 34, -97, -7, -1, 23, -79, -13, 9, -8, 12, 2, 15, 2, 10, -16, 16, 49, -117, 33, 8, -5, 6, 27, 3, 85, -80, -6, 20, 45},
  {-26, -11, -9, -53, -35, -1, -8, -16, -57, -30, -4, 5, 28, -21, -18, 53, -92, -32, -10, -18, 3, -25, -7, -24, -15, -5, 29, -3, 56, 43, 27, 101, 17, -31, -8, -9, 27, -8, 24, 8, -16, 11, 20, 24, -15, -52, 45, -62, -10, -27, 42, -10, -76, 47, 18, -67, -64, -11, 20, -15, -37, 35, -27, 19},
  {-33, 2, -4, -24, -116, -10, 23, 16, -30, 12, -2, 30, 15, -10, -31, 27, 45, -43, -18, 41, -8, -40, -17, -73, -15, -8, 36, 13, 27, 21, -21, -61, 17, -17, 1, 3, 55, 14, 15, 11, 39, -9, 107, -4, 11, -50, 34, -15, -11, -55, 31, 14, 24, 20, 1, -54, -89, 104, 35, 85, -4, 37, -33, 10},
  {4, -45, -4, -91, 30, -3, 6, 23, 2, 29, -5, -10, 52, -13, 9, 19, 86, -61, -11, 6, -17, 3, -5, 25, -34, -19, -8, 7, 8, -20, -66, 45, 40, -28, 8, 33, -119, 58, 14, -7, -8, 57, 59, 2, 23, 8, 44, 4, -14, 8, -34, 49, 72, -10, -25, -24, -65, 36, -20, 102, -102, -1, -45, 3},
  {-30, -12, 9, -16, -83, -37, 20, 17, 10, 28, -19, -17, 21, -8, -8, 45, -18, -17, 6, 30, -39, -8, -12, 12, -2, 6, 4, -4, 30, -10, -22, 77, 29, -39, 16, 9, -4, 35, -1, 9, -14, -9, -33, -5, 14, 3, 35, 4, 17, 14, -13, 27, 89, 24, 4, 21, 37, -95, -13, -18, 4, 21, -16, 31},
  {-28, -5, 15, -56, -64, -47, 11, -14, 6, -2, -17, -16, -8, 0, -18, 27, 76, -9, 20, 64, -11, -7, 13, 11, 8, 46, -36, -23, 63, 2, 13, -85, -9, -28, 3, 10, 76, 37, -2, 3, 73, -39, 57, 3, -33, 31, 8, 11, 21, 16, 3, 14, 23, 23, 14, 14, -45, -109, 2, -37, 2, 22, 11, 49},
  {5, 12, 27, -27, -29, -12, -2, -12, 36, 7, -14, -6, -29, 8, -20, 7, 20, -2, 15, 56, 7, 10, 18, 20, 4, 31, -43, -25, 39, -2, 9, -39, 11, -10, -9, 7, -56, 21, 3, 9, 54, -10, 20, 1, -14, 27, 4, 14, 43, 30, 1, 7, -26, 12, -8, -8, -20, 18, -1, -6, -89, -4, 28, 23},
  {33, 23, 17, -89, -56, -8, 10, 1, -3, -17, 7, -2, 0, -18, -30, 11, 29, -14, -17, 58, -5, 4, 5, 2, -3, -15, -32, -17, 64, -3, -10, -76, 16, 4, -11, 27, -114, -3, 12, -20, -65, 33, -12, 20, 28, -4, 11, -13, -10, 10, 3, 29, -74, -3, 6, -50, -20, -71, -18, 49, 11, 11, 23, -28},
  {39, 9, 1, -20, -89, -12, 16, 5, -11, -68, 17, -11, 2, -16, -2, -4, 58, -40, -9, 49, 6, 1, -25, 4, -8, -41, -22, -14, -7, -1, -12, -80, 15, -3, -1, 23, -80, 1, 11, -19, 57, 43, -99, 14, -15, 14, 9, -4, -15, 5, -9, -54, 21, 2, -1, -105, 4, -54, 2, -42, 13, -10, 3, -3},
  {5, 12, -58, -1, -47, 16, 23, 41, -23, -12, 35, 9, 26, -40, 37, -16, -85, -45, -11, 43, 22, 13, -26, -12, -9, -24, 2, 49, -53, -15, 59, 5, -7, -9, -9, 70, -103, 28, 19, 15, -123, -2, -18, 54, 23, 18, -20, -22, -31, 18, -31, -94, -108, 18, -14, -26, -18, 113, 30, 37, 14, -12, 4, 11},
  {27, 60, 54, -38, 104, 40, -12, -9, 3, -10, 22, -26, -29, 44, -26, -14, -50, -43, -13, -18, -6, -3, 35, 8, -1, 3, -41, 49, 9, 11, -22, -61, -35, 0, -10, 41, -110, -48, -17, -5, 58, 25, 110, -39, -19, -2, -28, 21, -7, 19, 22, -78, 66, -20, 30, 47, -17, -95, -8, -54, -49, 41, 34, -23},
  {32, 44, 18, -12, -68, 49, 6, -11, -2, -41, -27, -1, 6, -41, -38, 27, -97, -58, 20, 28, -21, 16, 37, 37, 18, -23, 42, 13, -14, 14, -18, 5, 24, -3, -14, -19, -122, -5, -16, 2, 55, -62, -28, -23, 73, 18, -1, 20, 37, 5, 9, 70, -101, -12, -16, -67, -86, -100, -24, 48, 57, 3, 8, 32},
  {4, 26, 24, 7, -31, 52, -4, -12, -6, -23, -10, -21, 50, -21, -14, 75, 31, -39, -1, 45, -24, 17, 30, -28, -7, 5, -33, -34, 37, 9, -21, 25, 13, 5, -1, 30, 58, -3, -13, -8, 24, -15, -62, -16, 16, 27, -23, 15, -3, 0, 25, -65, -48, -7, 23, -37, 47, -93, 22, 23, 5, 12, 17, -14},
  {-10, -6, 9, -40, 71, 15, -14, -3, 16, -17, 8, -7, 26, -7, 6, -38, -52, -45, -4, 21, -8, 8, 9, -36, 5, 10, -16, -40, -61, -22, -27, -44, 19, -19, 4, 34, -52, 33, -9, -2, 15, -13, -19, 9, 31, 26, -4, 8, 13, -13, -1, -49, -31, 15, 19, 10, -6, 39, -10, -70, 17, 3, 18, -11},
  {-10, 10, -41, 16, -116, -36, -14, 6, -14, -20, -5, -2, 25, -17, 10, -43, -125, -3, 6, -1, -28, 13, 15, 18, 7, 12, -38, -3, -79, 27, 49, 55, -5, -14, -12, 28, -8, 28, -7, 4, 57, 45, -69, 13, -51, 0, -3, -16, 2, 4, -36, -63, 108, 59, 11, 0, -108, -39, -3, 7, 2, 30, 39, 9},
  {3, 42, -20, 10, -14, -29, 0, 6, -17, -26, 8, -39, -42, -23, 10, 2, 16, 7, 31, -18, 15, 20, 28, 32, 26, 6, -18, -7, -70, -31, 32, -75, -6, -12, 11, -13, 9, -8, 19, 13, -13, -12, -93, 1, -36, -15, -17, -10, 7, 11, -29, -16, -53, 5, -20, 12, -88, 33, -3, -36, -45, -6, 27, 16},
  {9, -22, 18, -4, -48, 28, -7, 31, 17, -24, 14, -13, -21, 2, 5, -5, -61, 57, 15, 24, 29, -3, 11, 21, -16, 7, -31, -13, 48, -17, -4, 6, -41, 16, 31, 8, -49, 57, 12, 3, 30, -37, -90, 89, 7, 51, 23, 16, -19, -14, -2, -39, -80, -24, -20, -24, -117, 70, 14, -38, -20, -19, -34, 45},
  {29, 27, 22, -29, -33, -22, -18, -31, -5, 12, -6, 9, -58, 33, -11, -28, 33, -78, 21, -21, 21, -19, 39, 12, -13, -14, -4, 18, -51, -11, -28, -2, 1, -8, -2, -81, -59, 18, -15, -14, -78, -4, 59, -13, -2, -20, 29, 3, 14, 18, 57, -16, -12, -10, 11, -5, -41, -10, -36, -82, -50, 9, -11, 40},
  {4, 8, 7, -90, -124, 34, -3, -14, 20, 12, 15, 1, -58, 31, 14, -42, -28, -36, -2, -9, 30, -15, -7, 20, 4, 17, 28, 30, -111, 0, -7, -100, -8, 14, -41, -8, -119, -63, 2, 22, -6, 7, -69, -5, 6, -13, -34, 19, 15, 4, 27, 66, 36, -27, -10, 3, -29, -33, 1, -31, 67, -6, 7, 4},
  {0, -9, 7, -79, -76, 13, -17, -15, 5, -5, 30, 4, -36, 47, 16, -8, 83, -96, 21, 18, 31, -17, 3, 27, -29, 18, 26, -28, -36, 8, 30, -16, 10, 8, 18, -18, 52, -56, -7, -27, 45, 1, -120, 21, -43, -4, 12, -3, 6, -15, 44, -106, -84, 0, 25, -11, -50, -50, 32, -109, -59, 47, 0, 8},
  {1, -5, 12, -110, -106, -5, 28, -8, -4, 16, 4, 2, 27, -11, -3, 33, 19, -52, 2, 38, 3, -13, -16, 14, 1, 21, 6, 6, -71, 3, -12, -110, 23, 22, 17, 19, 58, -5, -11, -16, -73, -28, -14, -7, -24, -13, 8, 23, -3, 2, 18, -64, -53, -1, 1, 12, -22, -88, 4, 40, -86, 4, -4, 3},
  {16, -11, 5, -53, -78, 7, 22, 7, -3, 4, 8, 5, 26, -19, -5, 21, -27, 5, 7, 8, -1, -24, 0, -7, 7, 12, 7, 20, 8, -19, 11, 43, 15, 11, 17, 3, -118, 26, -13, -18, -43, -3, -97, -12, -75, 17, 21, 8, 5, -19, -2, 67, 13, 19, 17, 24, 54, -22, -47, 21, -122, 8, -15, 0},
  {14, 9, 16, -44, -103, 0, 12, -3, -8, 13, 7, -8, -14, -25, -6, 1, 28, 33, 13, -8, -2, -15, -6, 4, 24, -9, -11, 23, -70, -38, 11, -106, 21, -3, 13, -25, 71, 5, 7, -2, 29, 8, 20, 20, -16, 9, 14, 2, 19, -2, -38, 8, -1, 16, 16, 32, -64, -90, -23, -50, -24, -1, 27, -7},
  {-12, 5, -29, 17, -102, -32, -9, -11, -22, -15, -5, -18, -64, -29, -11, 4, -19, 37, 35, 18, -3, -44, 2, 12, 24, 23, -29, -1, -45, 10, -87, 40, 18, 6, 68, -32, 7, 23, -11, -6, 51, -22, 69, -7, 11, -22, 30, 31, 35, -8, -19, -82, 65, 24, 22, 40, -27, -121, -54, -7, 11, -33, -5, 89},
  {39, 6, -27, -68, -82, -7, -102, -8, -1, 47, 14, 38, -63, 19, 16, -13, 27, -36, 24, -127, 40, 13, 26, -1, 11, -59, -6, -21, 4, 2, 22, -60, 29, -30, 12, -78, 96, 22, 15, -13, 57, -3, -75, -9, -74, 28, 33, 12, -34, 42, 67, 30, 4, -12, -27, 21, 14, -27, -66, -5, 80, -7, -37, 31},
  {22, -39, -24, -63, -36, -16, -60, -37, 23, -14, 4, 56, -63, 33, 29, -75, -47, 16, -16, -19, 49, 12, 38, 9, -14, -40, 15, 22, -56, -20, -6, 39, -22, -50, -42, -48, -62, -76, 37, 0, 67, 66, -63, -22, -34, -29, 20, -19, -26, 19, 23, -114, 43, -5, -32, 26, -64, 1, -42, -37, -30, -50, -64, -21},
  {7, -37, -37, -102, 28, -16, -87, -41, 5, -17, 9, 46, 1, 25, 1, -4, -28, 12, -1, -51, 33, 33, 19, 16, -49, -51, 13, -55, -55, -8, 9, -24, -6, -76, -37, -42, -8, -57, 34, 33, -61, 36, -114, -9, -7, -33, -9, -71, -37, 21, 73, 53, -72, -25, -25, 5, -69, 70, 29, -88, 52, -38, -70, -23},
  {4, -36, 6, 2, -3, -12, 25, 0, 40, 59, -18, 53, -2, 23, 0, 31, -17, -67, -40, -35, -20, 49, -63, 47, -31, -13, 29, -19, 82, 50, -90, -105, 70, 26, -33, 72, 1, -53, 17, 7, 84, 79, -119, -23, -8, -26, 42, 15, 5, 45, 19, -66, 78, -53, 0, -6, 31, 3, 9, -90, 67, -29, -13, 3},
  {-19, -24, -11, -77, -121, 13, 71, -3, 9, 35, -13, 55, 23, -9, 2, 9, 41, -91, -13, 14, -24, -11, -23, 8, -12, 6, 2, 12, -78, 0, -36, 80, 44, 56, 35, 38, 26, -17, -9, -13, -76, 23, 13, -17, -8, -31, 40, 24, 12, -5, 24, -98, -88, -10, 1, 20, 36, 46, -49, 5, -83, -7, -14, -10},
  {5, -18, 8, -14, -70, 37, 26, -4, 22, 41, -43, 28, 38, -32, 5, -25, 13, -17, 4, -10, -29, -21, 4, 25, 17, -27, 1, 27, 37, -19, -75, -21, 65, 19, 30, -5, -1, 13, -6, -32, 66, 10, -33, -22, -46, 14, 24, 57, 50, -11, 11, 40, 83, -11, 8, 41, -73, -56, -70, -46, 11, -17, -9, -19},
  {-15, -9, -26, 25, 29, 16, 19, -6, 48, 25, -41, -10, -12, -8, -14, 13, -113, 30, -27, -26, -26, -19, -8, 9, 20, 9, -44, -6, -56, -14, -128, 20, 77, -20, 1, 5, -79, -6, -3, -4, 105, 27, -89, -71, -37, -2, 19, 53, 38, -11, 30, -113, -115, 18, 19, 91, -34, 20, -46, -3, 55, 2, -6, 13},
  {68, -6, 30, -58, -2, -103, -34, -51, 2, 17, 40, 14, -97, 60, 1, -41, -90, 1, 75, -16, 63, 25, -33, 14, -21, 52, 17, 33, -74, 13, 104, 13, -75, -86, 8, -68, 15, -100, -3, 0, 28, -6, 6, 54, -47, -41, 16, -11, 44, 1, 19, -51, 83, 82, 34, 41, -62, 55, -14, -26, 29, 3, 28, 6},
  {41, -7, -19, -21, 41, -10, 23, 27, 13, 26, -5, 7, 36, -66, -23, -25, 12, -62, -34, -48, -32, 8, -11, -5, -39, -27, -14, -21, -8, 21, 49, 17, -8, 92, 12, 73, -52, 69, -2, 7, 32, 37, -43, 82, -5, 31, 24, 3, 7, 33, -7, -94, -39, -20, 3, 2, -26, -97, -31, -120, 82, -63, -18, 25},
  {4, -31, -25, -28, -92, 35, 1, 33, 36, 24, -4, 56, 67, -59, 39, -35, -37, 37, 16, -5, 6, 56, -15, -12, -33, 14, 13, -33, 53, -1, 28, -71, 23, 23, 27, 49, -4, 65, 9, 43, 69, -34, -5, 56, 26, 60, 13, 9, 0, 35, -37, -50, -10, -46, -15, -12, -60, -118, -34, 14, 43, -73, -39, 7},
  {29, 7, 28, -66, 32, -47, -71, 2, 10, 27, -12, -33, 6, -20, -7, -49, -91, 27, 4, -5, -14, 52, 46, 15, -20, -18, -5, -7, -75, 34, 47, -14, -13, -13, -5, 6, 83, 20, 1, -10, 50, 9, -115, 89, -26, 79, 35, -1, 56, 32, -35, -118, -56, -45, 27, -51, -97, -21, 4, -12, 2, -49, 13, 10},
  {0, 3, -27, -33, 33, 7, 6, -2, 11, -21, 3, 9, -10, 13, -13, -16, 46, -74, 4, -15, -4, 21, 23, -21, 33, 20, 24, -27, -56, 18, 76, 9, -12, -1, -11, 7, 80, 18, 1, 46, -32, -13, -83, -8, -48, 4, -31, -6, 11, 3, 3, 30, -47, -54, -28, 9, 29, -95, -6, -15, -29, -24, -7, 39},
  {-2, 32, -23, -75, 80, 15, -2, 33, 7, 4, -5, 24, 48, -14, -2, -29, 1, -42, 8, 17, -17, 30, 15, -49, 23, 40, 4, 24, -113, -4, 25, 19, -19, 23, -22, 1, 72, 36, 0, 27, -43, -51, -43, 8, -52, -4, -30, -2, -11, -6, 9, -30, 19, -52, -17, 11, -112, -77, -13, -24, -11, -54, -5, 5},
  {-19, 20, -6, -95, 71, 7, -25, 1, 12, 22, -7, 2, 7, 1, -17, -28, -108, 14, -41, -50, -31, -2, 25, -44, 32, -7, -32, -5, -99, 20, -60, -12, 2, 2, -45, 16, -48, 1, 14, 19, 43, -65, 72, -22, -24, 10, -18, 23, 12, -13, 25, 20, -43, -12, -14, 6, 5, 9, -28, -68, -7, -29, -27, 27},
  {40, 1, 33, 46, -67, -111, -45, -40, -12, -13, 42, -36, -50, 104, -69, -38, 65, -58, 51, 36, 23, -20, -34, 17, -4, 99, -7, -32, 34, 13, 88, -30, -85, 24, 31, -52, -10, 8, -14, -3, 118, -47, -44, 21, -15, 47, -12, -16, -23, -3, -12, -24, 84, 54, 48, 22, -85, 10, 50, -22, -89, 12, 66, 63},
  {-26, 49, 0, 1, -60, 23, 8, 26, 17, -23, 26, -20, -65, 68, -34, -36, -56, -61, 12, 20, -11, -47, -5, -69, 44, 108, -55, 20, 23, -24, 26, -88, -69, -29, 11, -58, 85, 11, -16, 14, -49, -20, 5, -23, -101, -13, -60, 28, 5, -57, -20, -17, -74, 8, 25, 32, -68, 12, -12, -88, 43, 30, 22, -12},
  {-9, 10, -55, -89, -112, 28, -22, 24, 8, -60, 13, 5, -35, -18, -13, 10, 73, -3, -9, -24, 9, -39, -20, -18, 45, 58, -47, -74, 22, -43, 19, -65, 7, 10, 1, -17, -119, -24, 15, 4, -95, -41, 5, -32, 11, -32, -20, 22, -18, -19, 11, 12, -97, -5, -26, -1, -14, 4, -25, -11, -85, 0, -37, 42},
  {-14, 14, 6, 7, -26, -88, -64, -17, 12, 1, -35, -27, -40, 5, -47, -43, -91, 11, 26, 60, -14, 7, 40, 28, -2, 16, -73, -34, 16, 43, 1, 2, 0, 5, -27, 13, 22, 31, 15, 32, -92, -23, 57, 29, 1, 4, 19, 7, -6, 43, 1, 13, -2, 11, -7, -31, -67, -52, 15, -77, -59, -22, 16, 56},
  {-27, 26, 43, -57, -9, -12, -34, -13, 32, 1, -5, -22, -48, 26, -28, -9, 35, -87, 32, 48, 0, 15, 26, 20, 12, 9, -41, -48, -12, 9, 59, -84, -18, -13, -23, -15, -42, 16, 1, 25, 32, -20, -48, 43, -71, 82, -10, 18, 42, 8, -23, 61, 17, 9, 17, -9, -30, -23, -1, 8, -31, -7, 52, 23},
  {5, 29, 19, -71, 40, 6, -56, -45, 30, -12, -4, -38, -28, 7, -7, -8, 83, -96, 24, 50, 1, 7, 42, 3, 9, 4, -46, -7, 50, -5, 13, 36, 7, 6, 4, -22, -83, 7, -17, -22, 92, 3, 63, 16, -39, 38, -9, 32, 29, 9, -6, -128, -53, 5, 17, -26, 82, -10, -8, 71, -20, -5, 60, 1},
  {-17, 47, 36, -7, -85, 4, -30, -14, 30, -1, -9, -12, -21, -6, 1, 3, -4, -35, -37, 45, 8, 15, 1, -27, 11, -1, -8, 25, -107, 29, -12, 35, -21, -35, -5, -2, 45, -3, 6, 2, 60, -34, -18, 14, 44, 43, -24, 19, 52, -5, -32, -18, -87, -32, -3, -21, 70, -20, 5, 42, -29, -18, 20, 1},
  {-43, 21, 30, -43, -84, -4, -20, -26, 17, 21, 50, -16, -8, 77, 24, -19, 13, -6, 22, 13, 16, -15, -13, -32, -21, 56, -3, 3, -64, -32, -34, -101, -56, -4, 21, -16, 72, 2, -4, -26, -102, 25, 77, -36, -18, 63, -38, 33, -10, -25, -13, 5, -23, 0, 38, 6, -74, 67, 25, 14, -22, 46, 42, -19},
  {-91, 29, 14, -55, -41, 28, -13, 22, 30, 7, -6, -1, -10, -1, 2, 31, -112, -20, -21, -26, -33, 5, 12, -93, 15, 1, -31, -56, 53, -34, -57, -11, 12, -31, -1, 7, 40, 15, 18, -2, 107, -21, -40, -10, 61, 17, -19, 38, 18, 0, -38, -7, -11, -20, -17, 20, -8, 6, -23, -11, -90, -1, 12, -8},
  {-117, -3, 36, 34, -3, -33, 59, 14, 4, -7, -8, 5, 5, 8, -41, 29, -113, -21, -7, -6, -19, -6, 9, -10, 4, 43, -36, -50, -97, 9, 8, -33, 0, -22, -21, 18, 5, 10, 0, 12, -40, -8, -68, -12, -20, -4, -9, -8, 7, -1, -9, 58, -89, 24, 28, -1, 16, 61, 6, 29, -108, 32, 9, 32},
  {-6, 5, 14, 11, 50, -65, 47, -5, 18, 15, -22, -31, -16, 21, -20, 8, -125, 62, 8, 54, -13, 14, 9, 7, -19, 34, -35, 16, -67, 10, -50, -77, 4, 16, -25, 27, 33, -4, 14, -21, -119, 25, 6, -74, -12, 16, -5, 23, -6, 23, -8, -78, -56, 16, 21, 0, 38, -29, 12, 37, -119, 28, 28, -34},
  {24, 2, 26, -17, 0, -24, 29, 7, -5, 23, -4, -25, -4, -4, -16, -37, -102, -6, 9, 42, -26, 27, -22, 15, -29, -15, -24, -34, 2, 13, -41, -22, 12, -5, -2, 3, 35, 22, 2, -33, 30, 53, -45, 22, -87, 4, 28, 7, -26, 15, -11, 41, -10, 4, 22, -35, -109, -7, -27, -61, -115, -3, 17, -11},
  {8, 10, -14, -37, -100, 3, 22, -2, -26, -16, 12, -2, 6, 9, -16, 27, -109, -88, 8, 29, -13, 3, -1, 13, -2, -11, -38, -29, 43, 5, -5, -79, 18, -5, -12, 12, -20, 9, -5, -18, -35, 28, -106, 31, -30, -33, 22, -10, -7, 8, 8, -77, 80, 41, 10, -65, 2, 52, 2, 78, -28, 11, 14, -7},
  {-30, 12, -19, -64, 79, -13, 0, 1, -15, -37, 21, -2, 18, -5, 9, 13, -13, -16, 4, 89, 3, -16, -9, -23, -21, 8, -10, 16, -10, 21, 20, -121, -14, -14, 30, 37, 81, 52, -10, -2, -65, 15, 62, 65, 6, 37, -18, -9, -7, -13, -16, 25, -69, 17, 12, -31, -1, -84, 47, 100, 62, 5, 11, 10},
  {12, 10, -24, 30, 49, -29, -47, 80, -72, 74, 66, 15, 8, -83, -32, -15, -48, -9, -87, -30, -32, -9, -45, -91, 27, -34, 84, -9, 77, 6, -40, -89, -49, 43, 64, 59, 49, 93, -31, -10, 63, -102, 47, -30, -39, 93, -8, -39, -16, -38, -63, 70, 64, -84, -50, -73, -27, 65, -57, 80, -36, -81, -27, 49},
  {-58, 29, 76, -12, 109, -66, -101, -26, 27, 26, -46, -66, -62, -35, -32, 2, 33, 10, 65, 70, -49, -12, -11, 8, 52, 39, -103, -67, 82, 24, -90, -10, -36, 52, 69, -44, 75, 69, 0, -5, -106, -89, -51, 47, -58, 98, -4, 107, 102, 22, -10, -21, 8, -41, 51, 15, -70, 67, -39, 74, 3, -43, 14, 44},
  {111, 103, 33, -58, 5, -63, -81, -31, -8, 127, 5, -122, -97, -19, -23, -10, -3, -18, -78, 56, 22, -6, 90, 18, -31, -29, 9, 127, -55, -11, -33, 12, -21, 11, -63, -37, -26, -104, -7, -83, -62, 109, -31, 7, -17, 11, -20, -3, 15, 17, -49, -64, -113, 51, 7, 2, 50, -14, 39, -52, -42, 35, 57, -128},
  {113, 4, -7, -42, 26, 61, 2, 26, -25, 20, 49, -12, 18, -32, 59, 74, -68, -18, -15, -65, 46, 44, -13, 0, -48, -79, 81, 84, -57, -27, 48, 24, 1, 56, 28, 17, 63, -11, 12, -43, -28, 42, -31, -49, -100, 11, 5, -25, -30, -3, -24, 40, -25, -6, -69, -16, 12, 46, 12, -116, -24, 3, -82, -74},
  {54, -5, -40, -57, -75, 14, 52, 28, -41, -36, 42, 48, -1, 4, 18, 32, 33, 37, 16, -8, 6, 0, -24, -13, -25, -24, 80, 5, -44, -64, -8, 26, -5, 28, 72, 14, -7, 51, -2, -8, -14, 16, 37, -40, 25, -55, -11, -18, -99, -21, -11, -12, -2, -10, -10, -33, 29, 11, -14, -59, 74, 1, -34, -95},
  {24, 5, -21, -36, -123, 4, 34, 6, -38, -6, 54, 5, 35, -1, 10, 33, -40, -8, -7, -3, -6, 13, -38, 8, -16, -50, 35, -34, -103, -28, -52, -126, 6, 11, 36, 16, -1, -5, -14, -66, 71, 70, -25, -37, -93, -65, 8, -8, -74, -15, -19, 69, -35, -8, 24, -70, -58, -71, -25, 70, -29, -17, -16, -49},
  {16, 20, -31, -52, 25, 7, 0, 23, -18, -15, 35, -10, -14, -29, 9, -7, -28, -57, 11, 19, 24, 32, -15, 9, 18, -34, 20, 15, -123, -3, 24, 18, 0, -6, 27, 14, -10, 32, 10, -3, -30, -24, -33, 31, -3, 58, -19, -6, -25, 13, -7, 18, 89, -13, -14, -33, -107, -78, 11, 54, 61, -32, 29, 29},
  {34, 68, 29, 21, 71, 28, -13, -4, -1, 12, 14, -19, -19, 46, -11, -5, -70, -50, -18, -36, -13, -1, 44, 1, 16, -3, -20, 60, 60, 15, 3, -43, -25, 11, -27, 28, -36, -22, -17, -7, -4, 11, 29, -36, 45, -7, -36, 6, -23, 26, 26, -67, -107, -18, 13, 31, 80, 30, 21, 37, 25, 49, 41, -20},
  {0, 55, -1, 21, -15, 76, 2, 1, 12, -25, -26, -8, 2, -24, -24, -5, 77, -38, 19, 17, -16, 29, 30, 14, 14, -30, 43, 6, 20, 5, 24, 33, 34, -18, -30, -23, 43, -5, -8, 13, -24, -45, 93, -12, 40, 11, -10, -17, 33, -1, 26, 23, 9, 9, -36, -62, -46, -77, -12, -20, -14, -11, -10, 11},
  {19, 20, 32, -8, 39, 23, -35, -13, -6, 4, -18, -34, 16, -17, -35, 68, -104, -51, -30, 21, -23, 27, 57, -20, -13, -14, -9, -28, -4, 29, -57, -52, 17, -2, -40, 27, 16, -16, 6, 0, -70, 5, -47, 12, 23, 5, -11, -11, -9, 20, 16, -67, 106, -1, 9, -15, 10, -47, 29, 44, -51, 0, 12, -26},
  {-7, 33, -5, 18, 15, 22, -30, -17, 7, -2, -18, -2, 8, -12, -14, -55, -17, -36, -24, -22, -15, -4, 35, -25, 14, -16, 1, -8, 26, 26, -47, 63, 48, -17, -26, 26, -116, 23, 21, -2, -78, -1, -23, -3, -52, 9, 8, -6, 48, 7, 22, -19, 67, 13, 0, 8, 35, -25, 1, 75, -3, -17, 10, -8},
  {1, 11, -35, -34, -32, -18, 9, 17, 0, -30, -8, -3, 7, -22, 13, -16, -109, -33, -4, -26, -13, 20, 13, 44, -8, -18, -28, -20, -58, 21, 41, 8, -11, -8, -42, 27, 13, 15, 32, 25, -110, 51, -111, 75, -39, 6, -4, -19, 0, 24, 7, -100, -16, 21, -15, -20, 78, -104, -1, 56, -59, 4, 15, 17},
  {31, -1, -37, 55, 83, -2, -6, 33, -7, 4, 8, 22, -54, -6, 26, -27, 0, 61, -8, 29, 27, 39, 8, 10, 3, 11, -7, 1, -106, -45, 21, 18, -22, -1, -36, -40, 46, -14, 48, 49, -4, 25, 27, 47, 16, 14, 14, -15, -12, 48, 19, -57, 84, -8, -36, -18, -74, -77, 36, 23, 61, -23, -23, 22},
  {0, -10, 21, 7, 86, -10, -18, -13, 12, -11, 5, -14, 35, 18, -1, 16, 17, 20, 25, 30, 4, 19, -4, 14, -24, -14, -26, -17, -13, 37, 39, -24, -7, -13, 9, 20, -5, 72, 1, -4, 107, 21, -101, 49, 66, 15, 14, -7, 9, 9, -11, -81, 70, -7, 0, -28, 13, -30, 33, 70, 79, -25, -16, 12},
  {30, 30, -15, -11, 105, -26, -24, -22, -2, 17, 15, -7, -44, 26, -13, -22, -73, -86, 24, -16, 29, 9, 32, 18, 4, -21, 9, 21, -60, -11, 7, -109, -19, -26, -26, -93, -79, 26, -2, 3, 14, -15, 36, 6, -59, -16, 15, -5, -13, 21, 69, 26, -87, 2, -2, 15, -95, -68, -23, -56, 72, 4, -26, 28},
  {-15, 29, 21, -97, -19, 35, -28, -38, 28, 12, 0, -21, -67, 17, 4, -17, -5, -49, 12, 3, 19, -7, 4, 13, 10, 18, 18, 18, -12, -6, 15, -39, -3, 2, -28, -21, 15, -91, 15, 20, 107, -19, -7, -6, -17, 13, -44, 10, 25, 4, 25, -24, 36, -28, -13, -3, -23, -38, 3, -17, -29, 4, -1, -11},
  {-10, -9, 22, -66, 32, -15, -47, -15, -26, 5, 17, -3, -54, 26, 3, 34, 9, -51, 13, 8, 38, -8, 2, 30, -34, 10, 24, -45, 24, 25, 61, 43, -20, -18, 19, -43, 18, -36, 8, -2, -28, -26, -44, 61, -11, -11, 18, -30, 16, -6, 41, 35, 39, -16, 16, 3, 55, -84, 31, -68, -121, 37, -20, 23},
  {15, 0, 23, -13, 57, -19, 10, -30, -51, 4, 15, -19, 17, -3, -31, 3, -121, -40, 10, 4, 4, -66, 0, -7, 8, 20, 3, 7, 0, 3, -10, 10, 17, 3, 38, -16, 64, -19, -10, -4, -34, -36, -56, 14, -33, -34, 28, 6, 4, -3, 40, 20, 17, 29, 7, 18, 31, 7, 1, -109, -18, 24, -22, -14},
  {-2, 0, 18, -63, -66, 12, -14, -20, 3, 9, -18, 24, 26, -10, -20, -40, -7, 1, 44, -13, -17, -42, 46, -12, 23, -7, -14, -12, -95, -2, 64, 16, 9, 28, 16, -45, -124, 48, -16, 7, 67, -5, -79, 0, 5, 18, 26, 16, 14, -28, 6, -41, -60, 42, 26, -5, -28, -49, -27, -7, -46, 28, 1, 3},
  {16, 33, 9, -74, -117, 13, 34, -18, -28, -4, -18, -24, 21, -52, -42, 8, -113, 12, 62, 12, -37, 15, 16, 21, 49, -14, -25, -13, -112, -13, 27, 10, 25, -4, 46, -50, -59, 14, -9, -18, -42, -18, 43, 29, -42, -1, 35, -15, 31, 5, 3, -93, -96, -4, 17, -4, -107, 69, -15, 79, -32, -11, 27, 2},
  {-14, 29, -10, 31, 13, 7, -19, -16, 35, -27, -15, -8, 12, -38, 5, 16, 16, 24, 41, 28, -8, 66, 47, 16, 25, -25, -22, 12, -41, 16, -1, -103, 17, 5, 27, 17, -60, 59, -1, 27, 44, -52, 53, -2, 65, 22, -5, 26, 45, 34, -2, -17, 54, -38, -4, -3, -67, 66, -40, 27, -61, -48, 10, 23},
  {47, 35, -22, -37, -126, -36, -57, -13, -4, 45, 9, 18, -35, 13, -4, 3, -12, -56, 7, -91, 16, 45, 32, -19, 12, -50, -2, 5, 44, 0, 29, -56, 0, -33, 7, -57, 68, 24, 17, -9, -65, 4, -106, -26, -50, 9, 12, 2, -16, 47, 42, -104, -20, -1, -20, 14, 65, -80, -58, -67, 23, -34, -37, 12},
  {31, 4, 11, -71, -87, -16, -14, -11, 0, -10, 13, 44, -3, -30, 4, -36, -65, 8, -36, -20, 29, 40, 18, 21, -4, -25, 23, 19, -7, -24, 16, -58, -21, -20, -36, -3, 9, -49, 41, 8, -55, 23, -10, 37, 31, 39, -2, -23, -9, 26, -28, 34, -55, -26, -50, 19, -25, -63, -21, -6, -14, -65, -47, -25},
  {11, -36, 7, -46, -55, -13, -33, 4, -11, 12, 19, 43, 3, -25, -13, 24, -110, 59, -21, -41, 20, 29, -11, -18, -48, -18, 36, -52, -53, -4, 34, -94, -21, -12, -24, -5, 67, -31, 36, 29, 60, -4, 0, 41, -23, -10, 4, -70, -17, 26, 18, -37, -2, -27, -33, -8, -26, 51, 7, 25, 22, -22, -56, -10},
  {-2, -26, 7, 41, -95, -9, 71, 11, 17, 35, -4, 41, 12, 8, 10, 23, 78, -107, -32, -4, -26, -22, -72, 42, -3, 28, 7, 6, -2, 44, -90, 73, 29, 29, 19, 56, -10, -26, -5, -24, -32, 56, 4, -1, -22, -21, 34, 27, 30, -4, 7, 44, -57, -68, 25, -19, -68, 18, -5, -76, -85, 3, -4, 9},
  {-11, -14, -20, -47, -28, 23, 49, 0, -6, 6, 10, 51, 14, -4, -7, -26, -119, -74, 0, 29, -6, -58, -30, -4, -3, 55, 6, 6, 57, -2, -18, 18, 14, 36, 67, -1, -13, -21, 0, -25, -86, -25, -51, 23, -29, -35, 31, 23, -16, -20, 23, -109, -105, 7, 20, 32, -67, -40, -18, -122, -37, 25, -17, -29},
  {48, 11, -38, -70, 9, 43, 12, 11, -13, 24, -29, 28, 24, -41, -18, -37, -83, 23, -16, -12, -33, -4, 5, -24, 24, -24, -17, 19, -88, 22, -40, -14, 43, 7, -3, 21, 10, 5, 22, -6, -118, -39, -59, -24, -91, 18, 10, 5, 8, 2, -7, -56, 92, -15, -16, 25, 22, -92, -32, -82, -105, -39, -23, 8},
  {15, -3, -5, -42, -31, 11, -16, 47, 32, -6, -13, -19, 14, -7, 8, 37, -104, -24, -11, 27, -7, 35, -19, 9, -13, 29, -7, 12, -81, 22, -32, 77, 9, 12, -3, -10, -74, 60, 8, 20, 48, -4, -74, 24, -28, 72, -10, 13, -1, 7, -20, 36, 59, -42, 10, 3, 41, 68, 2, -17, 58, -35, 16, 43},
  {64, 24, 8, -63, 6, -8, 25, -5, -3, 15, 15, 5, -20, 1, 15, -16, 22, -50, 24, -24, 26, 38, -15, -38, 16, 7, 9, 28, 105, -16, 52, -117, -7, 0, 23, 33, -20, -30, 5, 8, -119, 27, -122, 30, -59, -20, 8, 26, 17, 2, -17, -6, 68, 25, 0, 24, -86, -2, -60, -81, 13, -35, 5, -3},
  {32, -5, -27, -67, 43, 36, 20, 26, 39, 36, 3, 16, 33, -49, 2, -17, 9, -58, -45, 6, -11, 35, -28, 5, -14, -8, 5, 22, -34, -3, 21, -77, 27, 31, -18, 45, -109, 53, 7, 28, -88, 43, -68, 40, -14, 30, -2, 14, 3, 36, -12, 54, 71, -75, -8, 7, -75, -86, -31, -36, 33, -58, 7, 13},
  {5, 5, 4, -67, 33, 32, 8, 22, 49, 44, 1, 41, 9, -36, 44, -24, -90, 44, -12, -7, 18, 53, -13, 29, 1, 18, -3, -13, -115, 3, -34, -94, 25, 27, 7, 35, 79, 49, 15, 43, -35, -10, 69, 10, -24, 72, 10, 30, 35, 48, 1, -39, -44, -103, -28, -14, 21, 12, -17, 4, -74, -63, -16, -5},
  {30, -14, 27, 10, -35, -43, -29, -1, 11, 47, 0, -5, -42, 16, -7, -28, -77, -12, -7, -8, -10, 2, 4, 35, 7, -7, -12, 13, 20, 5, -17, 43, -10, 6, 4, 9, -15, -35, -9, -35, 70, 48, 71, 48, -97, 28, 11, 15, 57, 7, -19, -103, -119, -23, 35, 5, -78, 69, 17, -37, -94, -4, 22, -25},
  {-5, -4, 6, -57, -59, 1, 14, -16, 17, -20, 8, 34, -5, 12, -5, 4, -1, -77, -12, -21, 5, 19, -8, -13, 26, 21, 4, -38, 6, -7, 17, -113, 17, -18, -14, -7, -55, -22, -4, 31, 28, 9, -100, -44, -26, 0, -30, 13, 18, 2, 13, 10, -50, -48, -31, 40, 52, 56, -15, -68, -99, -18, -7, 24},
  {-7, 2, 0, 43, -52, -3, -12, -19, 17, -35, 7, 13, -19, 20, 24, 1, 41, -4, 5, 8, 9, 12, 5, -23, 25, 16, -4, 19, -2, -6, 11, 11, -1, -1, 1, 1, 69, -5, -10, 12, 40, -31, 24, -7, 4, -4, -10, 7, 13, -27, 17, 48, -104, -3, -12, 23, -25, -90, -4, -93, -121, -7, -18, 12},
  {39, 12, 14, -75, 54, 2, 29, 4, -14, 3, -20, -13, 20, -41, -28, -20, -13, -25, -33, -28, -15, 2, 24, -45, 18, -26, 2, 10, 55, 37, -32, -45, 8, -26, -41, 54, -83, 21, 27, 7, -29, -26, 44, 18, -60, 15, -6, -6, -7, 5, 21, 54, 33, 11, -5, 2, 20, -55, -4, 45, 73, 1, 15, 21},
  {33, -13, 20, -5, -91, -34, -11, -21, -4, -19, 10, -39, -39, 67, -30, -42, -55, -38, 46, 27, 11, -5, -33, -10, 6, 70, 10, -13, 20, -7, 87, -98, -14, 13, 36, -8, 78, 13, -6, 7, 21, -14, -41, 18, -66, 27, 7, 20, -14, -8, 11, 70, 42, 44, 39, 26, 31, -114, -15, 89, -1, -9, 31, 49},
  {-1, 7, -27, 26, -61, 40, 14, 22, -7, -32, -7, 0, -26, -5, -40, -24, 35, -50, 4, -2, -21, -31, 30, -75, 40, 38, -36, 1, -53, -9, 15, -101, 1, -21, 10, -52, -28, 34, 8, 15, 33, -33, 54, 23, -67, -73, 11, 6, 13, -16, 13, 52, -48, -9, -6, 15, -53, 11, -34, 41, 4, -10, -34, 27},
  {-33, 5, -38, -58, -42, 16, 9, 4, 7, -21, -9, -3, 35, -22, -14, 22, 62, 31, -1, -34, 0, 16, 20, 2, 21, 19, -26, -43, -8, 18, 0, 62, 26, 28, -34, 12, -17, 8, 16, 22, 13, 3, -96, 0, -88, -8, 5, -12, -9, 11, 0, -53, -80, -28, -27, -24, -111, 55, -15, -102, -37, -21, -51, 39},
  {-20, 13, 10, 23, -4, -62, -23, -3, -1, 21, -13, -1, -50, 34, -30, -91, 14, 12, -9, 41, -12, -19, 19, -42, 2, 28, -4, -9, 20, 13, -18, 3, -21, 42, -10, 25, -58, -34, 15, 18, 12, -21, -47, 14, -48, 9, -6, 16, -42, 6, 1, 17, -47, 6, 9, 1, -112, -83, 34, -19, -22, -4, 27, 12},
  {-22, 22, 22, -94, -4, 0, -36, 3, 25, -4, 23, 0, -21, 6, -25, 1, -127, -57, 27, 17, 13, 14, 4, -14, 12, 9, -16, -45, -63, -20, 68, 34, -18, 25, -7, 5, -19, 20, -15, 20, -122, -6, -49, 38, -7, 70, -38, 25, 9, 0, -18, 52, -61, -18, 21, -37, -63, -122, -3, -62, -64, -16, 30, 2},
  {-11, -11, -4, -85, 71, 23, -36, 0, 16, -42, -1, 5, -11, 1, 14, 14, -62, -44, 9, 44, 14, 33, 16, -3, 1, -4, -13, -14, 64, 2, 31, -83, 5, -2, 16, 19, 8, 46, -1, -6, 62, 24, 19, 17, 2, 33, -3, 10, -11, 4, -12, -56, -59, -10, -17, -51, 76, -105, 8, -38, -90, -11, 5, 22},
  {-10, 68, 42, 3, -23, -16, 13, 33, 1, 24, 7, 1, -10, -1, -7, -3, 24, -29, -57, -4, -4, 26, 19, -60, -7, -21, 3, 8, 56, 17, 31, -55, -24, -35, -66, 34, -53, 32, 23, 25, 72, -18, 1, 7, -44, -3, -44, -14, 18, 31, -37, -2, -39, -11, -7, -26, -10, -45, 43, 34, 5, -22, 33, -31},
  {-55, -18, 28, 13, 107, -2, 23, -11, -17, 2, -6, -19, -13, 40, -5, 37, -104, -9, 18, 30, -11, -18, -28, -46, -11, 64, -37, -5, 55, -4, -32, 57, -2, -12, 15, -2, -79, 21, 7, 0, -51, -7, -15, 4, -40, 29, -19, 16, -10, -7, -11, 58, -40, 15, 15, 5, -51, -37, -4, 96, -125, 18, 25, 40},
  {-65, 1, 42, 4, -91, -16, -13, 7, 40, 10, -20, -16, -24, 9, 4, -15, -32, -42, 10, -3, -22, -5, 15, -31, 4, -3, -44, -12, -106, -36, -40, -111, 17, -19, 0, -7, -125, 18, 8, 13, 76, -34, 8, -3, 8, -8, 5, 35, 49, 15, -40, -41, 17, -20, 2, 19, 51, -109, -28, -7, 13, -4, 22, 0},
  {-46, 1, 17, -5, -52, -16, 39, -14, -9, -4, -17, -11, 31, 1, -18, 64, 54, -5, 20, -2, -25, 3, 33, 5, 9, 37, -26, 4, 54, 31, -5, -9, 10, -3, -16, -6, -108, 10, 3, 16, 37, 14, 78, -26, -26, -14, -11, -1, 23, 3, -4, 49, -69, 12, 15, -24, 14, -69, -2, -20, 14, 28, -24, 53},
  {13, 11, -13, 19, 68, -45, 19, -1, 8, 9, -25, -41, -3, -7, -19, -9, -27, 45, 10, 35, -16, 18, 13, 4, -8, 12, 2, 19, -72, -13, -5, -80, 25, 9, -18, 0, -1, -19, 20, 3, -6, 8, -85, -49, -21, -26, -4, 19, -3, 17, -13, -43, 7, 5, -14, 25, -25, 77, 8, 55, 25, 12, 15, -39},
  {17, 0, -10, -44, -66, -13, 5, 7, -10, -6, 21, -21, -8, -3, -16, -31, -121, 12, 14, 13, -3, 8, -7, 12, -28, -9, 15, -26, -83, 8, 3, 40, -7, -7, 24, 6, -5, 16, 2, -10, 8, 12, 40, 22, -43, -35, 15, -27, -94, -12, -3, -17, -25, 38, 11, -46, -126, -92, -22, 43, -70, 7, -1, -33},
  {17, -1, -11, -55, 45, -3, -3, -14, -66, -42, 35, 6, 4, 10, 4, 66, -103, -66, -14, -35, 9, -16, -6, 5, -15, -43, 15, -5, -36, 12, -5, -37, -6, 21, 18, 29, -73, -15, 8, -19, 36, 15, -68, 9, -52, -62, 15, -55, -26, 0, 45, -25, -46, 48, 36, -43, -59, -98, -1, 69, -55, 25, 9, -9},
  {-20, -8, -24, -58, 20, 5, 44, 47, -31, -4, 31, 39, 55, -25, 16, 2, 32, -4, -32, 29, -20, -16, -52, -41, -24, 32, 17, 16, -57, 23, -10, 2, -35, -31, 27, 77, 31, 27, -6, -10, -57, 9, 57, 34, 2, 19, -3, -30, -27, -13, -26, 52, -49, 26, 14, 18, -112, 25, 34, 96, -79, -8, -3, 7},
  {7, -7, -16, 23, -2, -11, -22, 22, -30, 36, 29, 21, -20, -24, 4, 11, 100, -20, -11, -11, 1, -8, -10, -27, 16, -17, 39, -7, -49, -7, -1, -1, -21, 32, 21, -1, 27, 14, -14, -23, -83, -20, 6, -2, 7, 37, 8, -11, -2, -15, 7, 30, 79, -22, -13, -6, 106, 74, -30, 35, -74, -7, 6, 7},
  {1, -3, 39, -52, -26, -42, -49, -32, 31, 35, -29, -29, -27, -36, -10, 25, -103, -11, 36, 46, -37, -9, -13, 61, 27, -27, -44, -53, -22, -24, -63, 6, 22, 33, 48, -51, 17, 37, -12, -28, -10, -22, 29, 19, -38, 27, 57, 50, 43, 0, 15, 18, 19, -6, 32, 0, -25, -117, -40, -31, 10, -14, -5, 16},
  {23, 5, -10, -21, -61, 6, -2, 16, 8, 45, 11, -35, 0, -5, 47, 33, 55, 17, -2, 13, 29, 21, 2, 7, -14, 12, 11, 65, -114, -33, 1, 6, 5, 13, -5, 3, -12, -25, 13, -3, -47, 23, -61, 2, -60, 43, -14, 11, -39, -11, -10, 14, -11, -9, -36, -29, 70, -94, 24, -17, -51, 6, -29, -31},
  {60, -2, -7, -46, 85, 38, 28, 8, -9, -18, 11, 29, 8, -5, 21, 28, 55, 35, 5, -8, 14, 41, -7, 0, -3, -25, 0, 44, -79, 2, 8, 29, -16, 26, 27, 12, -64, 48, 13, -15, 41, 28, 65, -18, -39, 5, -15, -15, -8, 10, -38, -26, 14, 1, -38, -21, 60, 46, 25, -66, -99, -8, -27, -22},
  {14, -4, -17, -54, -72, 3, 46, 17, -13, -15, 12, 13, -6, -3, -4, -22, -86, 81, 16, 40, -7, 17, -5, -5, -13, -11, 2, -11, -49, -51, -4, -104, -10, 28, 43, 13, -49, 49, -9, 6, -34, 12, -106, -23, -33, -42, -8, 0, -63, -14, -30, 16, -99, 0, -2, -32, -106, -47, -1, -6, -70, 0, -2, -72},
  {8, 8, -5, -53, -117, 14, 67, 8, -23, 6, 22, -10, 43, 18, 3, 19, 18, -36, -12, 10, -20, 0, -21, 13, 16, -26, -1, -15, 50, 9, -30, -71, 20, -9, 1, 37, 57, 24, -25, -43, -40, 61, -123, -35, -79, -54, 1, 9, -18, -3, -6, -75, -18, 8, 29, -51, 9, 117, 6, -9, 38, 8, -3, -10},
  {7, -1, -32, -18, 48, -11, 16, 11, -22, -12, 18, -28, 17, -52, 8, 10, -66, -41, 38, 38, 0, 11, -10, 30, -11, -26, 9, 2, 33, 6, 36, 0, -8, -7, 56, 2, -6, 74, -14, -10, 81, -38, 85, 68, -37, 53, -7, 16, -2, 17, 4, 54, 54, -16, 6, -22, 42, -63, -24, 60, -23, -3, -2, 36},
  {18, 55, 9, 1, 14, -1, -20, 10, -5, 16, 0, 28, -17, 29, 2, -43, -65, -31, 37, -23, -5, -1, 37, 14, 35, -2, 14, 47, 79, 0, 68, 75, -46, -20, -22, -46, -108, -9, -24, 21, 119, -43, -77, -13, -96, -9, -21, -16, -39, -16, 46, -55, 18, -44, 28, 42, 9, 25, 25, -18, -28, 23, 15, 15},
  {14, 69, 11, 30, 64, 71, -16, -1, -9, -24, -29, 8, -9, -32, -26, -11, 51, -41, 45, 27, -9, 5, 44, 20, 22, -4, 37, 8, -53, 11, 41, -2, -3, -3, -2, -26, -69, -18, -8, 25, 113, -83, -84, 27, 30, 28, -27, 7, 74, -23, 14, 0, -13, -9, -25, -51, -87, 88, -29, 67, 58, -7, 4, 0},
  {13, 46, 15, -49, -100, 39, -23, -8, -4, 11, -10, -35, 9, -26, -9, 49, 59, -60, -30, -9, -8, 54, 40, -36, -4, -14, -2, -51, -96, 33, -24, -98, -7, 43, -42, 7, -25, 3, 9, 16, 3, -22, 9, 23, 12, 67, -40, -14, -7, 16, 32, 60, 97, 3, -7, -47, -65, 76, 23, -10, -74, -6, 17, -41},
  {37, 47, 26, -79, 59, 3, -62, -29, -21, 1, 17, -27, -41, -3, -27, -43, -81, 1, -33, -23, 11, 30, 54, -15, 12, -10, 3, 17, -31, 38, -12, -84, -10, -22, -25, 8, 16, -10, 20, -21, -48, 11, -7, 7, -79, 10, -11, -31, 22, 23, -12, 68, -77, 14, -11, 3, -14, 70, -16, -66, -22, 9, 37, -66},
  {27, 26, -20, -61, -91, -34, 5, 24, -26, -24, 28, -26, -28, -16, 31, 16, -76, 9, -18, 12, 24, 18, 13, 26, -12, -9, -44, 11, 15, 12, 49, -4, -61, -22, -20, 38, -96, 26, 19, 2, 2, 58, -47, 50, -77, 13, -5, -35, 27, 1, -21, -41, 54, 16, -5, -14, 24, 74, 0, 55, -51, -10, 35, -9},
  {19, -5, -36, 40, -76, -15, -2, 33, -29, -25, 5, -9, -56, -27, -1, -21, -41, 35, 20, 12, 20, 31, 23, 54, -13, -9, -16, -7, -81, -9, 14, -30, -4, -17, -12, -74, -16, 18, 38, 50, 15, -2, 23, 68, 38, -39, 35, -51, -21, 34, 6, 91, 20, 23, -26, -23, 13, 85, 13, 11, -61, -11, -33, 40},
  {-42, -30, 5, -10, -4, -5, 24, 53, 46, -68, 11, -15, -14, 43, 11, -29, -17, 114, 41, 63, 7, 54, 26, -55, -41, 42, -38, -14, -82, 40, 65, 46, -89, -22, -18, 41, -7, 80, -10, 36, -95, -16, 76, 101, 34, 101, -43, -5, 6, 10, -35, -71, 96, -57, -7, -13, -98, 47, 80, -82, 52, -54, -31, 63},
  {23, 48, -3, -20, 89, -25, -5, -28, -11, -2, -3, -8, -23, 24, -9, -63, 49, -82, 22, -16, 23, 28, 43, 19, 17, -15, -12, 7, 69, 0, 44, 57, -37, -19, -50, -63, 32, 48, -2, 30, -50, -11, -72, -4, -81, -28, 4, -33, -43, 34, 58, -77, -71, 9, -1, -5, -45, -65, -12, -10, -70, 1, -19, 57},
  {-10, 18, -4, -39, 36, 21, 25, -17, 35, -39, -7, 23, 4, -4, -8, 9, -17, -30, 20, 33, -8, 26, 4, 22, 15, 12, 19, -36, -111, -1, 4, -43, -21, 14, -17, 5, -75, -14, -1, 25, -49, -25, 54, 11, 55, 29, -31, 3, 19, -4, 23, -71, 42, -15, -8, -29, -89, 69, 9, 4, 12, -8, -14, 23},
  {-4, -11, -21, -23, 85, 30, 24, 25, -10, -29, 15, 41, 37, -8, 4, 33, -11, -58, 10, 2, -9, 47, 9, -3, -10, -5, 15, -34, -30, -12, 10, 62, 6, 37, -10, 11, 17, 19, 18, 11, -71, -25, 64, 5, -32, 1, 5, -46, -27, -9, 21, 56, -13, -15, -24, -37, -66, 75, -1, -34, -71, 8, -30, -9},
  {23, 18, -28, -41, 43, -20, -8, 44, -19, 3, 15, 26, -30, -20, 12, 9, 75, 14, -37, -29, 22, 33, 10, 3, -7, 2, 24, 19, 44, 23, 14, -8, 6, -30, -31, 17, -32, 22, 55, 20, -53, 6, 64, 43, -128, 22, -20, -62, -18, 25, 8, -49, 69, -8, -39, -18, -108, -124, 21, -64, -51, 6, -30, 14},
  {42, 27, -10, -95, 91, -30, -12, 44, -12, 10, 22, 14, -61, -25, -9, 2, -6, -5, -38, -8, 45, 16, -4, 1, 13, 23, -29, 9, 38, 4, 61, -111, -71, -2, -33, -9, 72, -8, 45, 56, 65, -15, -79, 42, -70, 44, -32, -46, -3, 23, -2, -41, -32, 16, -28, -24, 68, -87, -16, -59, -107, -37, -11, 43},
  {57, 40, -62, 56, -41, 38, -20, 20, -30, 29, 14, -20, -65, -19, -15, -26, -112, 22, -34, 36, 28, 37, -2, 11, 11, -15, -10, 26, -20, 1, 9, -28, -49, -10, -30, -86, -78, -50, 36, 43, -46, 13, -81, 74, -17, -19, 11, -58, -18, 38, 8, 60, -77, 16, -19, -39, -14, -32, 6, 5, 65, 11, -7, 41},
  {1, -31, -32, 40, 64, -37, 4, 22, -11, -23, -10, 17, -15, -16, -4, 41, 75, -10, 9, 8, 0, -24, -2, 1, -28, -9, 21, -47, -71, 45, 19, 7, -33, -41, 11, -1, -86, 52, 4, -6, -26, 24, 56, 77, 8, -16, 38, -47, 20, 6, 18, -89, -26, -25, 14, -33, 23, -39, 3, -10, 67, -27, -46, 74},
  {44, 19, -19, -10, -25, 2, 26, 0, -1, -10, 6, 19, 15, 22, 12, -61, -87, -36, 1, -45, -6, 16, 1, -16, 32, -17, -40, -15, -12, 10, -15, -120, -17, -23, -10, 8, -50, 21, -11, -4, -33, 36, -30, -19, 10, -35, -1, -5, -55, 12, 25, 16, 88, -35, 17, -24, 53, -84, -42, 45, -80, -41, -23, 27},
  {-1, 11, -38, -56, -103, 47, 9, 21, 2, -30, 2, 9, 26, -16, 12, 40, 20, -47, 9, 19, -13, 29, 29, -13, 16, -8, -12, 6, -59, 12, -31, -57, 22, 30, 13, 13, -23, 10, 0, -17, 45, 2, 59, -14, 21, 26, -15, 30, -5, -8, -1, 66, -110, -40, -22, -14, -110, 14, 3, -14, 42, -28, 3, -27},
  {-9, -5, -25, -59, 67, 50, 38, 45, 2, -5, 2, 23, 35, -23, 54, 23, 24, -28, 9, -7, 17, 35, -31, -21, 20, 5, 2, -10, -108, -54, -29, -7, 33, 29, -11, 15, -89, 31, 36, 33, -53, -25, -99, -18, -126, -5, -4, -25, -9, 21, 28, -116, -43, -51, -68, -40, -118, -47, 6, -91, -108, -25, -39, -6},
  {10, -9, -66, -41, -114, 15, 6, 4, 10, 2, 4, -2, -15, 27, 21, 21, 48, 45, -15, -42, 26, 18, 13, -13, 15, 8, 21, 34, 28, 25, 5, -54, -6, -22, -50, -36, -119, -8, 47, 31, 24, 29, 23, -14, -65, -14, -37, -34, -23, 4, 85, -75, 2, -5, -46, 0, -12, -16, 53, 8, -24, 3, -44, 10},
  {62, 55, -2, 26, 51, 52, -9, 23, -57, 44, 47, 27, -75, 14, -43, -35, -84, -54, -65, -74, 43, -36, 3, -32, 33, -41, 27, 26, -40, 7, 11, -103, -31, 32, 1, -47, 51, -37, 37, 10, -35, 50, -85, 14, -27, -66, -12, -34, 1, 14, 35, -56, -11, 30, -12, -28, 20, -5, -44, -2, 12, 28, -15, -5},
  {42, -16, -14, -53, -39, 18, -13, -30, -39, 21, 29, 14, -85, 13, -25, -22, -68, -21, -2, 6, 41, -36, 1, 45, -12, -18, 8, -18, 38, 5, 63, -56, 8, -14, 7, -45, -4, -86, 17, -12, -36, 27, -75, 7, -46, -40, 50, -52, -14, 9, 45, -83, -90, 20, -35, 12, 12, -5, -23, -38, -101, -1, -35, 15},
  {-1, -35, 22, -42, -44, -58, 0, 5, 13, -15, -18, 11, 97, 25, -8, 16, 0, 26, 64, -4, -28, 2, 32, 55, -40, 1, -25, -1, -52, 64, -23, 69, -6, -33, 14, -12, 109, 20, -36, -28, -2, 81, 83, 48, -3, -6, 46, -8, 11, 8, 26, -64, -124, 13, 60, -31, -75, -99, 29, 18, -19, 38, -4, 36},
  {58, 10, -45, -98, -63, 49, -18, -1, -16, -17, 6, 50, -35, 9, 16, -38, -85, 7, 16, -44, 17, 7, 29, -40, 30, -45, -25, -14, 16, -4, 20, 42, 14, -10, 22, -6, 49, -10, 13, -8, -73, 27, 71, -19, -40, -45, 17, 9, 5, 37, 28, -6, 13, -29, -21, 25, 5, -62, -83, 12, 59, -63, -52, 4},
  {12, 34, -9, -78, -17, 69, 5, 26, 15, -9, -18, 15, 31, -73, 2, -45, 69, -30, -4, 10, -2, 13, 22, -43, 24, -60, -25, 6, -89, -34, -16, -47, 55, 4, -2, -2, 47, 56, 16, 22, 42, -23, -103, 1, -48, -29, 3, 18, 31, 16, -22, -51, 76, -58, -46, -5, 33, -112, -41, -32, -53, -66, -18, -3},
  {-14, 49, -17, -5, -75, 9, 40, 22, -9, 23, -12, -9, 55, -61, 24, -8, 29, -28, -7, 1, -6, 37, -27, -41, 46, -2, 16, 21, 10, 29, -36, 7, 55, 35, -12, 40, -81, 40, 23, 48, -29, -51, -121, -4, -76, 20, -15, 9, 23, 24, 10, -2, -19, -58, -38, 10, 79, 72, -13, -93, 3, -54, -22, 50},
  {-10, 49, -27, 2, 37, 27, -9, -44, -14, 3, 2, -63, -4, 36, -7, 45, -24, -58, -12, 6, -11, 28, 67, 2, 25, -44, 7, 44, -44, 40, -26, -18, -29, -28, -46, -45, -93, -19, 5, -9, 79, 39, -17, -2, -72, -34, -26, -39, -23, 22, 69, -51, -127, 32, -22, 10, 35, -98, 49, -53, -50, 45, 30, -26},
  {45, -26, -22, -43, 91, 54, 3, -3, -30, 7, 39, 23, -10, -11, -8, -41, 45, -72, -17, -27, 28, -34, -32, 47, -7, -16, 16, -3, -70, -20, -12, -64, 8, -9, 33, -28, 25, -32, 16, -36, 40, 24, -29, -7, 33, -57, 37, -58, 5, 17, 28, -104, -15, 35, -26, -28, 63, -49, -5, -55, -22, 2, -47, 4},
  {60, -62, -46, -50, -57, 15, 44, -20, -16, 9, 23, 70, -20, 14, 11, 22, -89, -41, 3, -19, 39, -80, -32, 66, -26, -8, -6, -7, -30, -28, 7, -32, 74, -9, 49, -4, 66, -79, 9, -29, 70, 23, -46, 14, 32, -72, 101, -43, -6, -10, 92, -44, -75, 31, -13, 49, -40, -60, -27, -15, -43, 7, -127, 1},
  {17, 11, 23, -19, 67, -13, 15, -5, -48, 19, -6, 6, 10, -26, -47, 21, 27, -24, 30, -15, -23, -13, -2, 14, -6, -22, -11, 16, 71, 67, -8, -42, 14, -23, -6, -9, -41, 16, -2, -10, 60, 27, 71, 11, 2, -57, 47, -13, 27, 12, 16, 77, -120, 32, 32, -11, -87, 31, 16, 55, -70, 35, 17, 52},
  {8, -23, -30, -35, 73, 49, -20, 0, -5, -34, -11, 19, 12, -10, -31, -24, -51, -29, 39, 38, -18, 5, 33, -31, 12, -24, -35, -9, -24, -2, 49, -11, 31, 2, -3, 2, -80, 44, 5, 11, -88, 22, -69, 40, -15, -4, 13, -6, -6, 13, 36, -15, -16, -12, -24, 17, -71, -90, -42, -57, -50, -47, -38, 21},
  {1, -24, 1, -32, -89, 19, 8, 12, -11, 6, -13, 26, 78, -40, 16, -3, -101, -75, 27, -3, -6, -5, 11, 10, 3, -55, -20, 5, -13, -8, -34, 59, 18, -28, 14, -24, -49, 63, 4, -4, -101, 2, -92, 19, -28, -54, 39, -18, 10, -7, -44, 20, -68, -1, -5, -18, 68, -9, -21, -111, -118, -14, -40, 3},
  {-1, 1, -1, -28, -23, -24, -10, -39, -27, 7, -55, -64, 65, -64, -22, 69, -106, 4, 24, 23, -51, 35, 58, 68, -18, -49, 24, 22, -115, 69, -15, -84, 54, 17, -15, -1, 32, 44, 2, 4, -110, 13, -36, 42, 21, -9, 80, -27, 26, 32, 20, -101, -36, 12, -2, -20, 68, -73, -48, -71, -95, -8, -41, 62},
  {38, -26, -7, -33, -119, 30, 10, -1, -26, 3, 4, 4, 20, 4, -5, -4, -78, -60, -17, 2, -14, -12, 10, 14, -37, -52, -5, 32, -102, 41, -3, -104, 18, 5, -36, -2, -31, -21, 11, -10, -53, 59, 29, 11, -8, -24, 2, -52, -38, 17, 21, -48, -34, 14, -16, 7, 23, -40, 10, 14, -104, 2, -8, -14},
  {6, -16, -33, -31, -48, 36, 63, 14, -50, 27, 33, 69, -27, 17, 12, 6, 51, -30, -28, -15, 10, -70, -42, -30, 1, 47, 27, 60, 42, -9, 12, -97, -16, 8, 29, -7, 78, -53, 15, 35, -107, 0, -110, -103, -5, -62, -3, -25, -21, -28, 58, 29, -53, -6, -18, 3, 28, 42, 12, 85, -78, -43, -27, 15},
  {-2, -29, 1, 12, 84, 9, 12, -5, -5, 3, 27, 0, -37, 62, 38, 3, 59, 26, 17, -3, 38, -50, -39, -39, 34, 33, -7, 63, -53, -40, -35, -15, 14, 63, 65, -37, -18, -42, -15, -12, 49, -43, -92, -39, 15, -30, 20, 23, 10, -66, 60, 56, -91, -13, 13, 60, 64, 49, 12, 37, 56, 19, -46, 29},
  {20, 34, 38, 5, -22, -6, 3, -28, -23, 28, 9, 10, -53, 6, -26, -17, -30, -28, -17, -32, -4, 1, 21, 2, 1, 3, -3, 37, -99, 47, -37, -66, -5, 0, -49, 12, 81, -29, 16, -17, -109, 7, -44, -1, -11, -44, -20, -26, 30, 29, 4, -12, -127, 39, 6, 18, -16, 102, 20, 110, 57, 18, 46, -9},
  {-8, -12, -2, 18, 73, -44, -12, 10, 9, 5, 17, -24, 29, -2, -11, -51, -101, -32, 0, 11, -19, -8, -8, -20, -11, 9, -22, 8, -10, 5, 29, 21, -32, -1, 4, -8, 9, 65, 2, 1, -2, 12, 49, 61, 10, 0, 0, -7, -24, -2, -26, 3, 10, -3, 21, -2, -50, 24, -10, 111, 20, -13, 9, 23},
  {0, -11, 1, 42, -20, -56, 46, 15, -6, 7, -17, -23, 45, -31, -11, 40, -55, 8, 10, 19, -34, -37, -32, 26, -15, 0, 10, 15, -14, -5, -24, 55, 30, 0, 25, 17, 79, 30, -28, -17, 108, 33, 3, 10, 27, -31, 47, -7, 5, -29, -16, 25, 88, 23, 22, -10, -48, -123, -16, -13, -62, -3, -23, 18},
  {33, -3, 11, -7, 14, -12, 43, -20, -2, 18, -43, -45, 21, -29, 14, 50, -23, -6, 13, 44, -25, 24, 27, 46, -29, -30, -2, 40, 51, 21, -70, -120, 38, -2, 13, 10, -92, 24, -7, -22, 16, 21, 78, 8, -47, -8, 55, -6, 31, 29, 20, -75, -73, 15, -12, 2, -41, -85, -12, -87, -82, 4, -21, 14},
  {74, -23, -25, -101, 79, 30, 49, 0, -1, 11, 11, 67, 0, -18, 15, -6, -17, -41, -23, 18, 2, 15, 6, 21, -10, -41, 7, 49, 70, 4, -26, 47, 41, -2, -15, 0, -119, -25, 11, -6, 33, 59, -118, -8, -11, 12, 18, -26, 3, 11, 6, 29, -128, 10, -36, 17, -74, 47, 13, -56, -46, -5, 0, -43},
  {21, 15, -28, -13, -104, 59, -6, -10, -7, 41, 46, 44, -30, 30, 18, 10, -59, -41, -58, -20, 32, -40, -8, 0, 10, -22, -33, 8, 8, 17, 15, -15, -17, -8, -3, -2, 33, -90, 7, -15, -109, 45, -56, -55, -69, -25, -25, -6, -39, -6, 10, -65, 68, -21, -16, -1, 52, 7, 11, 19, -78, 0, 18, -49},
  {-23, 10, 11, -30, -33, 21, -11, -13, -10, 14, 40, 11, -53, 26, 19, -12, -84, -7, -32, -15, 64, -19, -24, 18, 2, 38, 17, 36, -33, -20, 39, 32, -2, -4, 27, -11, -20, -35, -6, -27, -45, -12, 24, 5, 58, 24, -20, 13, 9, 33, 0, -35, -23, -29, -1, 9, -70, 78, 7, 96, 27, -9, -2, -16},
  {25, 27, 15, 25, 58, -13, -5, -6, -34, 24, -2, 16, -7, -10, -32, -5, 42, -22, -30, -17, -33, -6, 5, -18, 8, 14, -9, 6, -48, 46, -31, 50, -6, 1, -23, 17, 75, -11, 13, 11, 12, -7, -21, 3, -56, -24, -5, -29, 13, 21, 2, -46, -2, 28, 28, 2, 19, 13, -1, 118, 14, 9, 19, 19},
  {-8, -33, -43, -69, -22, -17, 1, 49, 2, 47, 20, -16, 20, -5, 23, 68, 33, -32, -34, -5, -11, -2, -35, 0, -15, -16, 24, 28, -95, -34, -29, 56, 15, -1, 0, 15, -108, 18, 16, -11, 127, -12, -67, -22, -9, 11, 20, 0, -45, 4, -30, -50, 31, -27, -28, 3, 63, -28, -11, -18, 17, -11, -51, -22},
  {-4, 5, 16, 13, -10, -2, -1, 20, -21, 30, 21, -8, -24, -9, 32, -36, -17, 30, -35, 4, 10, -14, -40, -15, -6, -7, 25, 65, 66, -35, 12, 44, -1, 0, 24, -1, -100, -15, -21, -19, 52, -7, 2, -26, -36, 32, -8, 3, -17, -26, 0, -89, -18, 13, 0, -12, -51, 42, -1, 45, -66, 21, -10, -9},
  {41, 2, -7, -43, -76, 47, -10, -17, -8, 13, 20, -17, -24, 8, 42, 19, -111, -80, 42, 19, 51, -19, -8, 28, 26, 8, 29, 42, -59, -46, -26, 41, 26, 20, 55, -16, -102, -36, 16, -6, -8, -33, -7, -40, 24, 3, -7, 20, -4, -8, 31, -13, -55, 2, -19, -7, 38, -41, -3, 37, 4, 20, -26, -41},
  {32, 30, -2, -16, 10, 45, 17, 30, -5, -22, 69, 39, -9, -3, 51, -48, 21, 32, -10, 1, 27, -2, -18, -18, -7, -29, 21, -5, -25, -15, 37, -24, 18, 17, 62, 9, -62, 26, 9, -56, 24, 33, -64, 13, -114, 42, -15, 24, -8, -15, -17, -50, -8, -20, -15, -35, 89, -57, 11, -70, 5, -23, 29, -46},
  {30, 2, 14, 28, 10, 27, -3, -9, 22, 20, 77, 20, -41, 36, 41, 5, -92, -36, -53, -22, 25, -11, -22, -53, -18, 3, 8, -9, 65, -44, -49, -67, -41, 40, 39, 54, 20, -52, 1, -30, -124, 24, -88, -37, -64, 7, -23, 45, -15, -6, -11, -24, 80, -30, 41, 4, 37, -105, 4, -61, -62, 9, 51, -34},
  {-15, -60, -13, -27, -121, 22, 13, -49, 51, 19, 52, 37, -91, 78, 39, -11, -95, 20, 9, 20, 74, 41, 5, -9, -40, 11, 30, 15, 32, -13, 95, -84, -12, -30, -27, -41, -33, -79, 2, -15, -61, 38, -30, 38, -54, 3, -40, -49, -4, 51, 22, 41, -72, -25, -16, 10, -61, 23, 41, 72, -18, -23, -14, -16},
  {11, 6, -5, 3, -92, -13, 25, 10, -15, 1, 9, 6, 21, 1, 13, -10, -56, -9, -11, -2, -3, -8, -14, -10, -8, 0, 20, 2, -80, -1, -7, -128, -5, 2, 10, 4, 74, 17, -9, 1, 6, 14, -102, -9, 27, -1, 6, -4, -4, -8, -10, 89, -52, -2, 2, -12, 59, 26, 7, 26, 28, 8, 7, 4},
  {18, 21, 46, 35, -101, -52, 6, 44, 28, 10, 30, -8, 2, 26, -28, 24, 6, -30, 17, 19, 16, -19, -33, 3, -28, 49, -28, 5, -56, -8, 17, -30, -46, 31, 29, 39, 25, -11, -2, -4, 63, 6, 53, 19, -22, 17, -25, 41, -2, -1, -19, 86, -89, -15, 28, -11, 19, 73, -31, 13, -8, 25, 42, 2},
  {4, 28, 45, -34, -121, 30, -3, -25, 16, -33, -34, -30, -4, -17, -36, 31, 68, -48, -1, 42, -29, -3, 37, 55, 11, 16, -1, 52, 10, 19, -30, 48, 17, 21, -30, 1, -8, -20, -34, 12, -47, -55, 69, -21, 30, 20, 4, 31, 8, -27, 5, -58, -3, -15, 33, -1, 51, -75, -29, -17, 76, 38, 22, 31},
  {-27, 33, 0, -21, 21, 26, 27, 1, -6, -31, -28, -16, 32, -53, -7, 29, -4, -32, 5, 53, -31, 31, 0, -16, 3, 2, -5, -11, 62, 6, 20, -98, -14, 1, -32, 30, 20, 37, -6, 29, 77, -80, -4, 19, 65, 51, -22, 7, -4, -7, 21, 66, -98, 11, -14, -45, 37, -68, -11, 18, -39, -30, -10, 8},
  {14, 6, 28, 30, -114, 5, -35, -8, 30, 4, 22, -1, -4, -18, -12, -40, 54, -35, 13, 14, 27, 25, 3, -15, 0, 31, -24, -49, 11, -29, 21, -83, -3, 3, -1, -25, -72, 8, 13, -8, -107, -62, -87, 31, -6, 70, -13, 20, 14, -6, 11, -86, -83, 8, 3, 14, -14, -46, -5, 49, -9, 12, 10, -28},
  {23, 25, 5, -38, 29, -9, -35, -18, -1, 6, 11, -27, -3, -44, -3, 17, 23, -34, 6, -7, -2, 26, 33, 10, 9, -12, -16, -5, -36, -9, 46, -34, -20, -16, 2, 2, 37, 38, 2, -10, -67, 28, 36, -9, -8, 28, 7, -2, 40, 9, -34, 19, 32, 0, 4, -11, -46, -86, -27, 11, 30, -10, 34, -1},
  {21, 21, 30, 24, -117, -16, -6, 5, -6, 5, 22, -38, -37, -27, 3, 15, 39, 18, 11, -5, 16, -2, 30, 18, -2, -7, -31, 9, -49, -28, 56, -33, -28, -27, 19, -8, 46, 39, -7, -13, -102, 41, -71, 23, -3, 9, -4, 12, 30, -6, -60, -87, -28, -3, 7, 22, 55, 65, -28, -71, -93, 0, 37, -10},
  {10, -1, 6, -14, -34, -23, 1, -15, -18, -24, 18, -37, -51, -23, -8, -31, 0, 35, 22, 17, 18, -16, 10, 47, -7, 12, -34, 10, -48, -21, -39, -73, -8, 44, 46, -35, 106, 8, -13, -16, -6, -2, 3, 56, 49, -43, 34, 2, -5, 10, -26, -6, 41, 32, 18, 23, -99, 80, -24, 46, -63, -10, -19, 33},
  {30, 19, 27, -25, -24, -38, 39, -19, -31, -41, -25, 27, 7, -43, -21, -49, 31, -35, 10, 21, -11, 7, 5, 35, -5, -12, -36, 10, -63, 12, -15, -24, -15, -11, 21, 15, 30, 12, 4, 14, 68, -27, -31, 19, -13, 2, 37, -12, 11, 18, 11, -44, -36, -8, 4, -40, -49, -78, -48, -69, 18, -22, -2, 42},
  {-11, -17, 22, -71, -27, -3, 47, -17, 35, -24, -6, 0, 13, -21, 15, -10, 25, -55, 34, 30, -1, 8, -8, 24, 15, 17, -24, -41, -6, 0, -44, 5, -7, 22, 25, 10, 19, -9, -34, 1, -101, 0, 16, -17, -3, 13, -11, 45, 2, -15, 13, -5, -5, -19, 23, -30, -38, -60, -12, -88, 16, -1, 0, 12},
  {-8, 2, 40, -45, -110, 19, 46, -5, 14, -3, -10, 0, 30, -34, -7, -23, -118, -92, 10, 17, -24, 2, 22, 3, 15, -12, -19, -7, -82, 4, -54, 37, 20, 58, 25, 31, -21, 18, -25, -42, -81, 5, 50, -22, -79, 10, 14, 23, 12, -11, -27, -40, -102, 23, 11, -13, -104, -38, -39, -22, -74, 33, 23, -10},
  {-16, 21, 10, -4, -105, 11, 72, 35, 1, -23, -14, -11, 77, -49, 17, 15, 39, 12, -3, 20, -55, 10, 4, -5, 17, -12, -30, 13, -118, -28, -50, -28, 52, 7, 31, 53, 25, 71, -36, -36, 64, 47, -30, -30, -46, 2, 7, 41, 45, -24, -80, -43, -61, -17, 18, -19, -61, -30, -36, -93, -3, 17, 29, -16},
  {-20, 16, -9, -76, -115, 10, 39, 42, -7, -36, -10, -16, 37, -33, 20, 37, 26, 33, 8, 21, -9, 33, -10, 13, -2, -6, -29, 30, 56, -10, 12, -42, -17, -21, -17, 5, -45, 64, -1, 19, -16, 22, -102, 12, -87, 23, -21, -10, 0, 16, -52, -13, -4, 0, 0, -17, -103, 75, 15, -111, -122, -21, -3, 25},
  {34, -14, -18, -71, -1, 16, -69, 22, 26, 25, 14, 7, -91, 0, 16, 8, 78, 48, -35, 21, 44, 50, -2, 25, -42, 6, 13, 16, -16, -20, 3, 13, -8, -23, -47, -78, -68, -32, 57, 57, -51, -17, 33, 56, -54, 40, -5, -39, -18, 64, -21, -72, -27, -6, -63, -25, -63, -86, 24, -21, -110, -26, -41, 25},
  {-2, -18, -39, -15, -117, -2, -34, 33, 19, -1, -1, 16, -56, -5, 9, 22, 79, 50, -72, 38, 28, 26, -13, -9, -45, 11, -9, -1, -29, 29, -50, -102, -19, -47, -15, -13, 28, -14, 40, 28, -79, 17, -10, 7, -75, 45, 12, -31, 4, 36, -28, 45, 35, -30, -29, -20, -14, -45, 1, -8, 44, -38, -20, 67},
  {34, -3, 2, -70, 36, 6, 34, 7, 20, -27, 3, 45, -8, 26, 10, -58, -1, 16, -19, -13, 11, -6, -8, 5, 22, -2, -19, -37, -59, 12, -58, 56, -2, -6, -12, 25, -95, 39, -16, -17, 107, 38, -22, 3, -8, 7, 8, 30, -5, 15, 44, 26, -114, -60, 4, -57, 53, 35, -41, 23, 31, -49, -7, 38},
  {-2, 12, 0, 6, 37, 32, 6, 7, -21, -39, 3, -19, 10, -23, 1, -2, -91, -35, 15, 19, -3, -15, 19, -20, 21, -28, -9, 4, 17, -3, -61, -95, 10, 43, 20, -7, -33, 14, -4, -24, -1, 1, -111, -24, -32, 4, 2, 30, 6, -4, -6, -41, -114, 5, -10, -35, -28, -25, -45, -106, -34, 23, -7, -22},
  {-4, 52, 15, -13, -14, -2, 5, 38, -46, -23, -13, -32, -23, -26, -22, -16, -88, -33, 4, 30, -2, -26, 18, -24, 51, 1, -11, 10, 9, -39, -33, 57, 23, 18, 35, -3, -104, 26, 9, 4, -114, -33, -1, -28, -77, 7, 6, 31, 26, 4, -11, 9, 8, -14, -12, -9, -12, 73, -17, 47, -41, 30, 26, 20},
  {22, 76, 37, -98, -46, 4, 1, 12, -43, -2, -21, -66, -45, -25, -7, -67, -78, 66, -26, -38, -10, -35, 18, -50, 48, -3, 32, 41, -24, -34, -54, -37, 8, 54, 37, -26, -51, 22, -9, 0, -58, -19, -3, -85, -62, -18, -14, 30, 31, -7, -23, -102, 80, 56, 7, 43, -16, -107, -23, 38, -37, 31, 50, 5},
  {19, 117, 93, 7, 86, 42, -48, -42, -19, 18, -5, -81, -21, 20, -9, -13, -40, 36, 45, -40, -10, -13, 84, -3, 46, -73, 34, 23, -114, 17, -56, -111, -3, 42, 35, -44, 45, -38, -32, -61, 41, 44, -4, -63, 9, 4, 14, 27, 52, 2, -20, -52, -71, 18, 65, -8, 27, -68, -78, -124, 5, 7, 78, -36},
  {59, 7, -47, 8, -62, 23, -128, -66, -24, -16, 23, -61, -84, -1, 3, -43, -26, -28, 21, -12, 51, 22, 71, 60, -15, -42, 31, -34, -13, 47, 38, 99, -64, -48, -23, -70, 60, -58, 30, -6, 44, 65, -68, -4, -10, -1, 45, -79, -13, 36, 60, -38, 51, -9, -23, -71, -63, 16, -10, 20, -67, 16, -33, 41},
  {-14, -1, -52, 82, -65, -3, 9, 37, 6, -60, 38, -1, 69, 55, 43, -50, 43, 25, -47, -38, 14, 78, 21, -9, -35, -4, 48, 50, -79, 38, 23, -89, -107, -80, -87, 66, -85, -8, -1, 41, 25, 35, -62, 5, -30, 14, -63, -100, -64, 10, -29, -17, -24, -44, -23, -114, 43, -90, 76, 47, 78, 11, -57, -54},
  {22, 9, -66, -35, 0, 19, -18, 2, -11, -37, 5, 28, -23, -19, 4, 10, -45, -6, -12, -18, 10, -5, 4, -24, 25, -26, -13, -8, 91, 19, -7, -122, 11, 1, 9, -12, 71, 10, 10, -3, -45, 5, 8, 11, -18, -23, -6, 17, 3, 3, 20, -22, 46, -30, -33, 14, 75, 70, -55, -103, -70, -77, -58, 26},
  {-6, -4, -18, -51, -116, 11, -48, -5, -11, -20, -14, 9, 6, -35, -15, -103, -39, 21, 11, -11, 30, -1, 28, -22, 10, -54, -4, 9, 22, -31, 20, 10, 44, -22, -23, -73, 27, -5, 34, 37, 40, -50, 63, 51, -33, -50, 20, -32, -4, 20, 35, 65, -12, -1, -41, -1, 10, 40, 4, -76, -42, 7, -43, -23},
  {-21, 19, 23, 14, -99, -53, -47, -57, 2, 5, -39, -47, -51, 25, -50, -35, -119, -12, 52, 76, 14, -31, 45, 49, 25, -9, 30, 0, 64, 25, 28, 0, 18, -57, -25, -128, 27, -12, 19, 54, -8, -60, 11, 36, -62, -54, 48, -16, 15, 37, 73, -39, -62, 47, 4, 15, 89, -89, 36, -68, -42, 50, 14, 80},
  {-41, 67, 18, -77, -32, 10, -9, -43, -2, -1, -25, -96, 28, 21, -25, -30, -71, -58, 45, 36, -28, -1, 83, 46, -1, -31, 40, 4, 22, 37, -4, 65, -16, 27, 45, -20, 27, 18, -41, -28, 7, -14, -24, -32, -3, -1, 9, 47, 11, -21, 33, -17, 24, 53, 59, 28, -23, -114, -29, -71, 0, 61, 65, -18},
  {25, 40, -38, -23, -35, 18, 5, 22, -27, 16, -24, -40, 38, -28, -4, -120, -1, -56, 31, -16, -43, -26, -3, 50, 5, -52, -9, -24, 67, -2, -61, 81, 39, 5, 42, -1, 98, 61, -12, -48, -39, 16, -109, -39, -64, -10, 61, 19, -11, -6, 10, -35, -21, 29, 19, 2, -47, -113, -94, 107, 31, 14, -6, -6},
  {40, -1, -9, -16, -59, 13, 40, -9, -4, 14, -19, -21, 37, -30, -11, 40, 20, -22, 23, -8, -28, -18, 12, 85, -20, -40, -47, -29, 37, 50, -83, -16, 17, -52, 31, 6, -8, -6, -14, -16, -10, 64, 24, -66, -11, 7, 73, 20, 45, -6, -25, -111, 70, 28, 15, 41, -68, -72, -36, 93, -40, 33, -33, 20},
  {-19, -32, 39, -70, 61, -68, -14, -33, -18, -17, 25, 5, -3, 34, -9, 35, 18, 15, 14, 36, 14, -16, -6, 21, -97, -45, 14, 10, -78, 29, 5, 70, -50, -98, 59, -11, -46, -25, -17, -36, -92, 61, -48, 5, 18, -37, 54, -34, 0, -18, -55, -78, 70, -29, 56, -71, -20, -90, 12, 42, 7, 26, 25, 12},
  {-15, -37, -15, -83, -32, 21, -15, -1, 22, -4, -27, 32, 28, -54, -32, -24, -55, 8, 12, 27, -17, -8, 44, -39, 9, -20, -54, -4, -40, 22, -15, 37, 7, -15, -15, -49, 30, 50, 22, 22, 42, 16, -88, 47, -15, -51, 28, -10, 35, 21, 4, 34, -97, -14, -26, 20, -65, -121, -36, -102, 29, -41, -29, 34},
  {19, -55, -8, -74, 70, -14, -9, -30, 18, 26, 10, 28, 45, -4, 18, -20, -83, -28, -7, -1, 26, 23, 23, 66, -36, -70, -4, 4, -123, 14, -13, -34, 32, -50, -28, -43, -10, 8, 15, -9, -18, 83, 23, 3, -55, -62, 62, -53, 3, 20, 4, -123, -73, 29, 0, -40, -35, -99, 24, 22, -2, -5, -65, -16},
  {-5, 2, 23, -15, 25, -32, -24, -41, -17, 18, -34, -65, 47, -1, -15, 65, 58, -72, 25, 44, -33, 37, 43, 102, -34, -79, 1, 10, -98, 59, 1, -28, 29, -18, -13, -72, -34, 41, 2, -18, 41, 54, 2, 15, -56, 4, 82, -32, 14, 35, 59, 63, 85, 7, -9, -32, 39, -31, -22, -23, -77, -15, -22, 15},
  {11, -16, -7, -20, 30, 28, 26, 3, -8, -16, -7, -5, 48, -9, -2, 23, -28, -49, -13, -14, -13, 2, -51, 22, -3, -26, 8, 1, 9, -8, 9, -2, 15, -20, 6, -36, -66, -10, 11, 22, 6, -8, -22, -4, 4, -33, 21, -9, -3, -10, -6, -69, 33, 2, -35, 43, 35, -108, -12, -98, -59, 4, -13, 19},
  {-3, -15, 5, -24, -120, -20, 52, 7, -4, -19, 5, 30, 64, -15, 31, 35, 10, -8, -15, 16, -23, -15, -22, -19, -27, 23, -38, 63, 76, 3, 5, 12, 15, -15, 8, 38, 29, 21, -18, -7, -2, 14, 66, -36, 53, -57, 21, 3, 7, -2, -1, -93, -25, 10, 16, 36, 69, -37, -19, 66, 29, -43, -14, 24},
  {-23, -26, -23, -16, 43, 11, -13, 14, 0, -18, 9, -23, 25, 39, 12, 34, 53, -14, -24, 1, 38, -20, -43, -46, -3, -18, 15, 43, -35, 5, -49, -10, 21, 35, -43, -14, -87, -14, 52, 23, -101, -42, -65, 6, 43, 11, -2, 12, -61, -65, 31, -26, 89, -30, -37, -53, -23, -113, 33, 59, -21, 20, -40, 0},
  {-17, -37, -120, -88, 8, -30, 99, 38, -74, 8, 25, 46, 58, -49, -2, -16, -52, -47, 0, -68, -23, -86, -65, -20, -15, -56, 5, 28, 40, 31, -29, -71, -49, -25, -3, 50, -24, -16, 4, 28, -70, -18, 6, -50, 21, -107, 17, -70, -110, -75, 26, 36, 55, 53, -1, 36, -42, 8, 45, -28, 30, 60, -77, -18},
  {19, 2, 15, -20, -107, 15, -17, -6, -3, 35, 5, -10, 25, -11, 0, -8, -104, -46, -9, -14, 1, 1, 28, 11, -24, -11, -16, 11, 29, 1, 5, 14, -14, -11, 3, -9, 43, 23, 9, -28, 102, 24, 116, 9, -34, 2, 19, -22, 14, 13, -47, -81, -46, -9, 1, 2, -112, 82, 7, 41, -60, -7, 5, -26},
  {10, -23, -39, -15, -75, -35, 31, 18, -16, 22, 34, 1, -3, 10, 23, 21, 62, 1, -32, 0, 8, 5, -44, 17, -25, 5, 36, 24, 49, -13, -2, 76, 3, -37, -10, 17, -24, -5, 2, -13, 79, 32, -22, -31, -64, 19, 11, -18, -28, -18, -36, -63, 25, 20, -2, -5, -1, -69, -5, -31, -97, 7, -14, 18},
  {26, 3, -13, -72, -15, -1, 31, 24, -19, -4, 36, -17, 7, 26, 47, 19, 44, 6, -21, 4, 27, 0, -43, -8, -15, -4, 29, 46, -49, -37, 17, -45, -25, 1, 20, 6, -109, -25, 5, -21, -90, -24, -8, -23, -47, 35, -12, -16, -34, -5, 27, 51, -93, 6, -33, -13, -3, 21, 13, -55, -82, 12, -7, -22},
  {-4, 23, 21, -45, -68, -40, 9, 7, -33, 16, 52, 26, -28, 14, 7, -45, -33, -40, -19, 19, 22, -11, -18, -27, 7, 28, 21, -44, -14, -37, 55, 60, -38, -1, 18, 4, 24, -24, 5, -2, -73, -17, -115, 0, -35, 15, -21, -11, -4, -3, 11, -21, -42, 12, 3, 12, 55, -77, -1, -32, -20, -4, 37, -17},
  {-63, -2, 40, -33, -69, -37, 9, 25, 58, -30, 31, 5, -34, 56, 28, 19, 28, -29, -10, 49, 35, 6, -17, -59, -7, 19, -1, -26, -118, -14, 18, 56, -20, -10, 9, 25, 9, 24, -16, -7, 40, 0, 63, -8, -11, 38, -17, 30, 13, 5, -31, 18, -104, -13, 18, 12, 7, -2, 29, 51, -57, -15, 41, 7},
  {-47, 16, 27, 2, -122, -15, -28, -25, 27, -70, 4, -30, -30, 5, 9, 56, -87, -30, 26, -2, 37, -36, -27, -53, 34, 50, 16, 47, -23, -4, 44, 64, -29, -33, 11, 1, -103, -5, 30, 36, 5, -97, -101, -42, 39, 50, -28, 46, 28, 1, -75, -59, 62, -34, -16, 16, 25, 75, -21, -36, 51, -22, -13, 24},
  {13, -53, -8, 32, -76, -28, 50, 15, -48, -13, 4, 52, 105, -33, -7, 55, -41, -12, 29, 30, -14, -56, -66, 20, -19, 16, 78, 13, -110, -3, -41, -12, 27, -31, 116, -8, 88, 29, -24, -47, 0, 9, 108, -63, -57, -51, 83, -1, 1, -86, 66, 90, 88, 9, 46, -47, -72, -49, -66, -39, -28, 58, -75, 33},
  {-24, -44, -25, -2, -100, -3, 1, 26, 35, 66, 3, -17, 2, 10, 35, 72, 1, -35, -8, 34, -14, -33, 1, -8, 28, -4, 17, 20, -126, -85, -99, 26, 27, 1, 43, 13, 81, -23, -6, -32, 109, -25, -29, -69, -29, 11, 9, 44, -3, -21, -10, -26, -66, -12, -19, 18, 118, -40, -50, 102, -70, 11, -36, -31},
  {14, 6, 5, -35, -23, 50, 8, 11, -27, 21, 62, 26, -13, 9, 43, -9, -72, 37, -22, -38, 28, 2, -48, -13, 13, -32, 64, 59, -17, -64, 11, 69, 35, 34, 37, -4, -59, -47, -19, -49, -17, 7, -5, -73, -65, 10, -10, 11, -12, -26, 31, -84, 8, 5, -6, -17, -62, 55, -54, -80, 29, 23, -39, -31},
  {17, -13, -8, -79, -77, 59, -20, -48, -6, -6, 74, 3, -17, 12, 51, 21, -34, -58, 44, -18, 51, -66, -15, 30, 10, -3, 52, 37, -19, -61, -17, -39, -4, 15, 84, -17, 30, -54, 0, -73, 32, -38, -30, -37, -68, -7, -13, 36, -43, -15, 64, -14, 105, 5, -9, 73, 95, -126, -14, -99, 71, 45, -32, -120},
  {41, -19, -20, -8, -58, 66, -29, -14, 49, 10, 127, 48, -96, 60, 84, -2, 37, -39, -1, -39, 74, -4, -26, -6, -33, -23, 48, 2, 58, -33, 34, -19, -2, -29, 32, -51, 9, -93, 1, -59, -80, 36, 19, 19, -105, 49, -47, 10, -55, 0, 14, 2, 106, -64, -34, 44, -50, -15, 36, -41, -56, -23, 4, -56},
  {-49, 43, 32, -7, -57, -8, -89, -38, 89, 32, 87, -20, -104, 102, 60, 24, 3, 2, -10, 0, 105, 29, -4, -59, 15, 31, -8, -19, 10, -105, 16, 55, -80, 33, 28, -5, 27, -60, 0, -22, -32, -25, 31, -75, -30, 20, -102, 69, 0, 9, -9, 70, -41, -62, 37, 26, 39, -56, -8, -3, -61, 12, 92, -53},
  {9, -24, -33, -75, 62, -101, -39, 3, 29, 24, 59, 34, -41, 18, -30, 45, 9, 48, -35, 44, 57, 56, -25, -22, -26, 37, 14, -9, 108, -11, 94, -82, -88, -58, -59, -34, 55, -23, 44, 24, 31, -37, -11, 117, 34, 82, -43, -91, -31, 49, -23, -122, 50, -35, -52, -115, -42, -6, 46, 55, 18, -68, -32, -3},
  {59, -7, -93, -79, -47, 69, -71, -24, -38, -10, -10, 42, 14, -9, 13, -3, -84, -16, 46, 33, 25, 56, 28, 43, 10, -75, -9, 32, -44, 4, 88, 1, 36, -12, -45, -33, -9, 11, 50, 51, 72, -19, 64, 46, -62, -12, 1, -74, -43, 8, 52, -7, 108, 18, -57, -2, -98, -15, 60, 21, 5, 9, -32, 39},
  {55, 39, 32, -37, -87, 47, -4, -7, -2, -9, 6, -7, -23, 9, -30, -10, 30, -46, -28, -11, 7, -10, 47, 33, -4, -10, -52, 61, 4, -3, -49, -43, -1, 15, -18, 33, -37, -36, 0, -14, 121, 14, -15, -2, 19, -16, -18, 2, -8, 24, 23, 70, 107, -10, 6, 34, 3, -103, -25, 76, -65, 37, 18, -2},
  {21, 45, 12, -3, -89, 20, 24, 6, -14, -30, -10, 4, 15, -26, -27, 36, -33, -75, -20, 34, -16, 29, 31, 18, -5, -2, 22, -9, 71, 22, 1, 35, 18, -12, -21, -10, 46, -6, -14, 12, -65, -54, 88, -15, 8, 17, -1, -7, 0, 12, 21, -82, -54, -2, -9, -78, 14, 79, 7, -8, 25, 14, 5, 29},
  {20, 25, 39, -53, -54, 8, 12, -16, 8, -12, -5, -18, 33, -36, -25, 66, 63, -49, 4, 40, -6, -6, 21, 0, 0, 10, -36, -2, -47, -9, -12, -73, 17, 4, 7, 24, 61, -21, -8, -13, 50, -29, -73, -25, 46, 28, -9, 31, -1, -8, 18, 29, 68, -1, 6, -43, -65, -73, -9, -3, 22, 4, 19, -36},
  {-19, -14, 23, -58, -44, 2, -21, -17, 24, -1, -6, -20, 48, -31, 0, -7, 30, -48, 27, 26, -15, 19, 13, 4, 4, 8, -40, -39, -104, -9, -4, -117, 12, -1, -6, 17, -75, 48, -18, 4, -10, -12, 55, 5, 54, 35, 16, 2, 22, -17, -11, 67, -69, 20, 18, 4, -97, 0, 4, -67, 12, 9, 11, -9},
  {-15, 3, 9, 8, -6, -31, -6, 0, 13, -7, 6, -14, 18, -14, 1, -36, 36, -20, -4, -13, -7, 16, 1, 3, 9, 5, -32, -19, -27, -24, -13, 46, 1, -12, 0, 29, 27, 16, 3, -5, -76, 28, 11, -10, -40, 15, -7, 10, 14, -2, -29, -53, -62, 30, 15, 22, -57, 21, -6, 7, -126, 4, 38, -12},
  {-13, 4, -41, -19, 90, -46, 16, 1, 2, -10, -4, -10, -7, -11, 10, -42, -105, -18, 20, 0, -8, 3, 6, 20, 19, 7, -42, 1, -31, -8, 32, -78, -16, -3, -4, 7, 67, 37, -1, -3, -69, 31, -47, 17, 13, 1, -9, -5, 19, 0, -31, -100, 27, 33, 3, 10, -117, 13, -4, -68, 17, -3, 38, 15},
  {11, 6, -26, -21, -61, -32, 13, 2, -1, -50, 7, -33, -21, -27, 13, -11, -74, 56, 54, 17, 12, 18, 14, 49, 22, 2, -10, -22, 5, -36, -8, -4, -2, 3, 35, -32, -18, 38, 6, 17, -95, -27, 73, 4, 75, -32, 6, 11, -6, 5, -23, -62, 57, 18, -19, 13, -58, 82, -31, -62, -46, -10, -10, 47},
  {34, 13, 37, 38, 11, -2, 12, -9, 11, -5, -2, 18, -44, 0, -11, -58, 10, -31, -9, -10, 31, -18, -9, 22, 5, -16, -7, 3, -44, -3, -66, -92, 12, 17, 0, -19, 9, -9, 6, 0, 29, 12, -46, 2, 38, 12, 12, 27, 24, 17, 39, 39, -117, -38, -5, -5, -115, -84, -49, -68, -2, -14, -4, 15},
  {-26, -12, 9, -29, -100, 27, 17, -9, 38, 15, -1, 1, -3, 7, 24, -34, 47, -40, 10, 10, 20, -4, -25, 5, -4, 7, -11, 12, 73, -4, -32, -77, -12, 4, -21, 12, -98, -12, -2, 33, -93, -13, -1, -11, 10, -3, -11, 19, -1, 5, 13, -83, -99, -33, 3, 5, 52, 22, 4, 9, 66, 0, -13, 0},
  {-36, -10, -1, -10, 53, 22, 35, -10, 4, 6, 12, 6, -24, 28, 13, -47, -63, -66, 7, 14, 6, -13, -18, 9, -4, 11, 16, -1, -85, 6, 4, -47, -6, 13, 4, 2, -83, -38, -17, 2, -114, 19, 28, -12, -115, 8, -12, 23, 6, -14, 9, -81, 19, -9, 19, 1, 48, 10, 18, -95, 2, 28, 4, -20},
  {-32, -9, -8, -102, 51, 10, 27, 5, 4, 7, -6, 10, 28, 12, 16, 23, -57, -41, 5, 31, -10, -2, -22, 3, 2, 12, 13, 7, -47, -19, -29, 66, 11, 0, 22, 24, -53, 3, -9, -8, -7, -19, -50, -11, -113, 18, -8, 38, -6, -2, -10, -100, -42, -4, 15, -2, 16, 76, -6, -57, -114, 26, -4, -15},
  {-5, -19, 23, -57, -35, -10, 26, 12, 29, 4, 6, 8, 23, -19, 15, -2, 37, -9, -1, 33, -2, -1, -6, 8, -22, 17, -8, 14, 1, -2, -25, 53, 13, 13, 17, 37, 80, 37, -6, -5, 32, -21, -81, 23, -41, 52, 16, 22, 20, 5, -12, 36, -27, 1, 4, 21, 27, -123, -6, -101, -62, -13, 7, -14},
  {20, -22, 20, -86, 49, 12, -5, 0, 25, 12, 10, 5, 0, -9, 21, 16, -94, 34, -4, -5, 23, 2, -8, 16, -24, 0, -5, 25, 16, -24, 24, -122, 10, 4, 25, -27, -74, 19, -2, -8, -79, 17, -103, 17, -16, 41, 27, 2, 16, 3, -26, -99, 45, 1, 8, 13, -119, -119, -26, -99, -94, 4, -9, 10},
  {0, -12, -34, -64, 0, -13, 10, 6, -22, -21, 2, -12, -40, -29, -1, 17, -123, 60, 18, -3, 14, -17, -17, 24, 8, 8, -11, -11, 28, -35, -54, -51, 25, -11, 27, -46, 123, -28, 11, 16, -70, 17, -5, -25, -6, -39, 23, -19, -4, 1, 5, -11, -11, 33, 13, 39, 50, 38, -43, -44, -81, 4, -27, 25},
  {8, 35, 18, -103, -54, 10, -38, -12, 2, 10, -11, 15, -40, -18, 4, 3, 9, -30, 29, -90, 20, -61, 11, -8, 48, -29, -5, -5, 39, -1, -34, -122, 12, -14, 51, -66, -84, 3, -1, -33, -79, -2, 74, -10, 8, 48, 20, 42, 3, -4, 46, 60, 4, -27, -18, 19, -108, 74, -76, 4, -32, -14, -28, 11},
  {-29, -27, -46, 14, 18, -23, -68, -32, 34, -64, 4, 23, 1, 98, 23, -46, 29, -1, 50, -16, 31, -9, 68, 22, 3, -43, -7, 25, 2, -17, -20, -74, -23, -93, -22, -101, -90, -67, 5, -17, -9, 94, -14, -59, -26, -74, 12, -10, -54, -2, 7, 66, 46, 57, 8, 37, 0, -50, -5, -63, 48, 28, -43, -37},
  {3, -28, 6, -47, -60, -26, -107, -46, 25, -39, -11, 28, -10, 6, 7, -69, 35, 44, -20, -23, 14, 47, 52, 7, -16, -48, 3, -13, -72, -2, 22, 44, -62, -72, -29, -40, 34, -27, 19, 11, 32, 50, -93, 2, 63, 11, -9, -24, 16, 43, -38, 12, 22, 20, -6, 31, -2, -51, 4, -85, -42, -43, -35, -40},
  {-9, 6, -8, -13, -1, 2, -64, -5, 18, -11, -5, -6, 3, 53, -12, -72, -57, 44, 31, -26, -4, -4, -5, -3, 1, -11, 32, -25, -124, 42, 30, 106, -12, -17, -14, -46, 87, -17, 5, 33, -58, 6, -1, -29, 27, -3, -12, -8, -21, 3, 40, 10, 56, -54, 27, -29, 26, -23, 24, 17, -49, -4, 0, 11},
  {27, 1, -7, -34, -18, -15, -10, -2, -33, 8, -1, -8, -25, 14, -10, -18, -29, -25, -15, -21, 4, -16, 17, 3, 4, -4, 20, 5, -95, 32, -26, -37, 4, 58, 33, 1, -75, -40, 6, -14, -7, 43, -80, 4, -52, -32, 25, -1, -38, -5, 58, -127, -40, 4, 1, 19, -73, 74, -12, 43, 38, 16, -19, -11},
  {2, -17, 25, -20, -46, 8, 0, -9, 15, 26, -24, 14, 5, -32, 5, -1, 24, -14, 20, 8, 5, -29, -7, 26, 17, 16, -5, -7, -77, -35, -19, -108, 32, 48, 56, -19, -91, 16, 9, -22, -23, -19, 25, 23, -45, 15, 48, 41, 26, -2, 7, -73, 2, 0, -2, 6, -13, -25, -91, 18, -87, -18, -31, 9},
  {-1, 4, -54, -55, 8, 14, 4, -19, -3, 3, -12, 1, 2, 4, 0, -2, 72, 21, -7, -21, -11, -27, 5, 1, 28, -14, -16, 11, -7, -24, -57, -100, 37, -12, 13, -11, 69, -32, -8, -25, 41, 19, -31, -34, -27, -47, 29, 4, 17, -2, 45, -103, 91, 21, 16, 44, -41, 60, -43, -88, 3, -2, -19, -5},
  {35, -38, 45, -24, -4, -96, -4, -27, 7, -5, 24, -7, -39, 20, -17, -27, 93, -36, 67, -7, 38, 6, -32, -1, -39, 45, -1, 29, 112, 37, 122, -57, -62, -91, 13, -88, 80, 9, 0, 13, 79, -11, -117, 127, 24, -33, 18, -32, 38, -11, -14, -52, -75, 55, 32, 16, -105, -2, -12, -62, 54, 12, 7, 43},
  {28, -23, -35, 29, 30, -79, 6, 20, 6, 7, 16, -24, 63, -15, -32, -44, -86, -24, 4, -98, -30, 17, 14, -9, -43, 30, -26, 22, -80, 34, 100, 55, -65, 3, 0, 12, 17, 65, -14, 4, 40, 20, -9, 94, 8, -32, 5, -18, -17, 8, -8, 34, -15, 30, 34, 11, -100, 26, 6, 29, -80, 19, -23, 16},
  {76, -32, -51, 6, -69, 67, -1, 48, 14, 47, 16, 49, 67, -76, 45, -14, -26, 5, -55, -31, 34, 54, 9, 3, -59, -62, 15, 1, 20, -37, 59, 43, 12, 59, 4, 42, -124, 69, 28, 29, -81, 12, 46, 67, 35, 35, 14, -37, -33, 32, -37, 57, -36, -48, -36, -82, -60, 82, 1, -112, -114, -63, -53, -22},
  {31, 9, -66, -74, -78, 13, -6, 57, -29, 11, -18, 49, 43, -17, 8, -74, -54, 37, 6, -20, 9, 57, 19, -47, -4, 1, 16, 9, -13, 23, 50, -41, -15, -6, -63, -1, -3, 81, 32, 69, 31, -62, 37, 76, 61, 31, -6, -65, -28, 42, -11, 66, -50, -45, -50, -85, -9, -46, 60, -84, -106, -76, -65, 53},
  {59, -11, -24, -67, -81, 6, -2, -19, -12, -6, 13, 10, -27, 31, 6, 23, 44, -47, -1, -36, 39, 20, 19, 15, -9, -18, 47, -3, -7, 10, 24, 16, -5, 3, -12, -8, -118, -52, 13, 19, -79, 67, -100, -13, -21, -50, 19, -52, -10, 14, 16, -84, 26, 11, -27, -33, 31, 59, 13, -96, -57, 30, -32, 1},
  {9, -8, -29, -59, -124, 14, 18, 6, -15, -2, 8, 36, 9, -20, 13, -9, -5, -73, -6, -6, 6, -5, -19, -15, 27, 27, 19, -4, -125, -8, 46, 81, -21, 12, 23, -2, -48, 6, 1, 23, -69, -34, -84, -3, -20, -18, -9, -10, -16, -3, 27, -57, -88, -24, -7, -15, -74, -59, -28, -69, -104, -41, -9, 26},
  {21, 14, -9, -83, 54, 21, 13, 0, -28, 3, -3, 27, 5, -5, -3, -18, 34, -14, -26, -14, -10, -11, 1, -26, 14, 9, 3, 6, -27, 13, -45, -115, 0, -17, 1, 9, -105, -3, -4, 8, -81, -30, 17, -29, -28, -41, 0, 1, 6, -13, 29, -55, -26, -23, -2, 4, -91, 55, -21, 29, 72, -8, -29, 2},
  {13, 2, 28, 17, -113, -88, -47, -28, -23, 36, 52, -17, -76, 76, -43, -9, -70, -38, -15, 21, 54, -13, -79, 21, -16, 68, 7, -21, -20, 14, 33, 71, -82, 31, 7, -14, 4, 3, 24, 23, -67, -23, 24, -12, 15, 48, 4, -18, -13, 20, -63, -5, -83, 43, 25, 14, 52, 30, 39, -6, -21, 8, 39, 51},
  {-15, 52, 7, 53, 63, 40, 38, 26, 6, -13, 44, -7, -40, 100, -17, -39, -57, -1, -23, -6, -9, -5, -44, -30, 33, 102, -7, 10, -3, -20, 1, -36, -61, 15, -4, 2, 30, 4, 3, 20, 8, 3, 35, -36, -94, 50, -72, 30, 2, -43, -23, -43, 55, 19, -9, 17, -9, 33, 30, -16, -6, 24, 44, -23},
  {-21, 29, -22, 40, 37, 79, -8, 34, 29, -72, 24, 3, -70, -2, -19, -38, -20, -4, -51, -64, 23, -29, -24, -47, 32, 16, -43, -23, -111, -51, -19, 46, 3, -20, -8, -21, 8, -55, 34, 6, -25, -33, -91, -72, -13, -58, -32, 12, -4, -25, -1, -51, -28, -16, -36, 36, -80, -112, -16, 24, -51, 13, -18, -35},
  {4, 21, -6, 19, 72, -33, -39, -1, -23, -20, -15, -14, -15, -17, -32, 25, 55, 116, -23, -17, -3, -6, 5, 7, -7, -7, -29, 11, -34, 33, -42, -24, 7, 26, 10, -24, -63, 3, 24, 12, -1, -47, -109, 11, 22, -31, 36, -22, -35, 16, -12, 26, -7, 29, -22, -9, 31, -61, 43, 62, -85, 6, -22, 57},
  {-2, -19, 14, -63, -59, -6, 4, -8, 13, -10, 12, 4, -31, 29, -9, 18, 22, 8, -26, 37, 11, 11, -6, 18, -25, -8, -34, -42, -16, 12, -14, 55, -4, 8, -22, 13, 58, -15, 26, 8, 46, 33, -75, 36, -110, 36, 17, -22, -5, 15, -25, -79, -31, 10, -1, -32, 103, -66, 60, -32, -15, 30, 3, 5},
  {-4, 11, 9, -61, 66, -4, -32, -45, 25, -16, 6, 1, -31, -1, -14, 3, -41, -90, 18, 23, 17, -5, 28, 33, 18, -7, -68, -34, -101, -16, 45, -104, 6, -18, -7, -20, 35, -5, -13, -8, -63, 5, -12, 5, 14, -2, 6, 10, 41, -7, 16, 3, -95, -1, 6, -5, 35, -102, -8, -7, -5, 4, 19, -20},
  {-21, 17, 5, -45, 14, 23, -19, 7, 8, -4, 7, 0, -21, -7, 17, 7, -71, -33, -16, 54, 16, -9, -3, -41, 12, -7, -18, 31, 4, -6, -31, -7, 25, -6, 9, 6, -65, 3, -1, -2, -63, -5, -2, 10, 16, 16, 2, 18, 34, -27, 7, -30, -45, -24, -1, -22, -44, -17, 18, 90, -84, 5, 1, 7},
  {-37, 54, 41, -7, 32, -3, -44, -11, -10, 55, 67, -59, -12, 72, 18, 19, -120, -34, -13, -8, 7, -39, 34, -47, -6, 26, 33, 18, 34, -26, -82, -68, -67, -15, 32, -4, -79, -17, -20, -47, 99, 21, -19, -15, -84, 30, -53, 27, 0, -2, -26, 49, -39, 8, 56, 30, -11, -54, 15, 74, -118, 48, 50, -32},
  {-108, 28, 29, 1, -27, -15, 32, 16, 23, 17, 24, 23, 4, 48, 25, -12, 32, 42, -63, -50, -11, -16, -54, -123, 24, 29, -13, -43, -26, -57, -116, -69, -18, -26, 14, 58, -73, -35, 8, 13, -51, -15, 56, -61, 33, 52, -58, 41, 16, 1, -60, -3, 56, -14, -1, 50, -4, -51, 6, 100, 15, 7, 45, -5},
  {-98, 3, 28, -62, 52, -18, 10, -10, 20, 25, -32, -9, -23, 19, -48, -18, -96, -25, -1, -4, -46, -6, 46, -16, 26, 15, -44, -85, 66, -20, -49, -93, 35, -3, -8, 3, -118, 33, 17, 23, 20, -15, 28, -32, -15, -4, 15, 15, 37, 24, -19, 60, -50, 15, -7, 12, -58, -71, -6, 33, -1, 3, 18, 48},
  {56, 17, 15, -3, 78, -35, 51, 15, -25, 36, -16, -6, 17, 4, -12, 25, -25, 23, -34, -2, -6, 17, 35, 17, -32, -2, -12, 44, -39, 29, -49, -41, 39, 41, -23, 35, -28, 0, 22, -34, 5, 65, -97, -47, 18, -24, 21, -34, 11, 29, -24, -90, 35, 36, 23, -20, -17, -43, 30, 4, 7, 56, -15, -18},
  {28, -13, 7, -6, -95, 6, 40, 7, 36, 2, 18, -14, 30, 13, 28, -33, 41, 42, 22, 10, 12, 39, -24, 4, -42, -14, -17, -28, 74, -1, -10, 27, -3, 16, 12, 19, 30, 11, -6, -39, 42, 57, -45, -11, -83, 31, -6, 8, -9, 2, -20, -96, -38, -19, 19, -13, 26, 33, -12, 40, -44, 13, -6, -57},
  {18, 16, -9, -23, -108, -22, 4, -8, -25, -9, 26, -3, -14, -19, -40, 11, -2, -37, 6, 21, -7, -8, 6, 25, 0, -7, -29, -44, -19, 9, 0, -84, -23, -15, -3, 5, -32, -10, 6, -18, -42, 29, -43, 23, -18, -28, 19, -34, -22, 16, 7, 11, 79, 6, 8, -52, -45, -114, -2, 53, -60, 0, 20, -26},
  {-3, -1, -7, -41, -109, -10, 1, 18, -22, -28, 32, -10, -5, -10, 13, 24, -93, -22, 2, 50, 18, -16, -30, -25, -4, 5, -2, 39, -12, 1, 28, 46, 2, -8, 36, 43, 0, 22, 3, -20, 43, -5, 120, 49, -17, 49, -13, 5, -1, -4, -46, 66, -5, 2, 12, -48, -79, -12, 5, -40, 26, -6, 28, 22},
  {30, 16, 11, -2, -37, -6, -44, 90, -82, 88, 77, 32, 53, -96, -19, -16, 59, 46, -100, -49, -52, -42, -44, -106, 18, -58, 102, 14, -34, -6, -64, -49, -42, 73, 74, 71, 48, 108, -39, 1, 4, -84, -54, -53, 5, 99, -18, -32, -19, -58, -67, 34, -89, -94, -47, -86, -46, 3, -85, 63, 27, -79, -37, 49},
  {-44, 33, -84, -24, -54, -43, -25, 3, -6, 16, 41, -32, -19, 2, -7, -41, 88, 27, 12, -7, -16, 0, -33, -46, 37, 34, -7, -5, 15, -24, -26, -67, -31, 15, 29, 15, -69, 52, -14, -24, -82, -60, 12, -42, -25, 74, -65, 57, -36, -27, 8, 56, -71, -37, 37, 42, -65, 45, 13, -17, -1, -34, -43, -31},
  {52, 90, -26, -55, 43, -28, -4, 53, -64, 40, 44, -3, -20, -8, -18, -37, -31, 41, -89, 12, 3, -62, -21, -32, 13, -32, 56, 65, -28, -55, -74, -29, -40, 36, 0, -13, 67, -37, 6, -37, 29, 51, 53, -95, 0, -52, -18, -31, -45, -44, -10, -51, 19, 26, 20, 8, 99, -111, 36, -29, 21, 9, -3, -57},
  {57, 25, -78, -18, -81, 92, 24, 48, 0, 33, 123, 34, -15, 43, 84, 32, -42, -22, -47, -107, 73, -2, -88, -46, 12, -34, 127, 109, -44, -86, -46, 35, -2, 63, 6, 13, 80, -42, 15, -26, 46, 35, -109, -68, -73, -14, -98, -10, -90, -37, 29, 24, 86, -13, -40, 23, 44, -37, 69, -12, -75, 29, -102, -96},
  {39, 3, -32, 2, -95, 65, 41, 0, 17, -24, 49, 45, -12, 21, 52, -3, -28, -16, -2, 7, 15, 39, -13, -8, -9, -37, 26, 18, 95, -45, -18, -38, 5, 38, 37, 27, 17, 10, -19, -33, -10, 57, -113, -73, 16, -23, -51, 30, -28, -10, -29, -77, 37, -14, 26, 3, 5, 36, 28, -39, -6, 8, 23, -78},
  {16, 3, -12, -15, -25, 29, 44, 0, 25, -29, 30, 0, 24, 17, 24, 24, 68, 26, 8, 11, -23, 5, -30, 6, -16, -14, -18, -22, 51, -23, -62, 1, 24, 17, 18, 17, 61, -4, -26, -44, -82, 76, 50, -64, -38, -49, -22, 21, -81, -9, -9, -59, -106, -27, 31, -18, -86, 74, -2, 60, 83, -5, 18, -80},
  {-10, -1, -29, -75, 25, 41, 47, 9, 10, -10, 35, 26, 5, 24, 35, 43, 6, -45, 0, -4, 4, 24, -16, 2, 6, -32, 25, -8, -58, -2, 4, 51, 15, 1, 7, 23, 79, 3, -11, -26, 88, 26, 66, -8, 20, 22, -30, 6, -28, 9, -14, 53, -80, -30, 13, -18, 2, -58, 12, 12, -7, -25, 26, -16},
  {-6, -1, 6, 37, 13, 6, 4, -10, 10, -3, 3, -11, -9, 9, 0, -11, -106, 3, 8, -8, -6, 12, 12, 5, 1, 11, -6, 4, 24, -4, 20, -25, -14, -9, -8, -1, -5, -3, -1, 14, -78, 11, -19, 1, -35, 11, -5, 7, 5, 5, 14, 87, 6, -4, 0, 7, 3, -72, 27, -81, -43, 0, 14, 10},
  {-24, -6, -12, 17, -16, 20, -20, -13, 48, 9, 9, -13, -1, 49, 8, -22, -50, -51, 32, -17, -11, 32, 41, 40, -20, 17, -25, 27, -28, 29, 70, 55, -42, -48, -20, -11, -45, 14, -25, 24, -62, 19, 49, 0, 19, -28, -33, -2, -65, 17, 26, -60, -27, 1, 22, 15, 24, 22, 39, -65, -19, 16, 44, 1},
  {31, 127, 36, 24, -48, 13, -60, -74, -104, -22, -28, -38, -72, 6, -95, 1, -119, -13, 17, -37, -39, -21, 99, -39, 38, -29, 49, 0, 18, 79, 45, -74, -9, -23, -31, -88, 1, -15, 19, 15, 39, -128, 48, 65, 29, -30, -42, -30, 29, 19, 112, -71, -6, -3, 10, -37, 0, 44, 14, 60, -39, 36, 72, 19},
  {-20, 88, -30, -29, 34, 4, -60, -97, -71, -91, -64, -90, 41, -69, -93, 41, 67, -37, 3, -21, -61, 19, 56, 22, 45, -22, 54, 28, -115, 100, 8, 9, -3, 20, -19, -63, -1, -21, 13, 45, -23, -60, 24, -40, 19, -70, -12, -23, 11, -15, 87, -23, 68, 46, 18, -66, 104, 22, 9, -30, -68, 26, 3, 59},
  {26, 58, -8, -14, 18, -6, -80, -53, -66, -98, 15, -27, -12, -23, -14, -45, -121, -3, -15, -64, 18, 42, 70, -22, -13, -65, 13, -9, 34, 40, -8, 1, 20, -54, -13, 26, 87, 23, 23, -18, -50, -7, -77, 62, 29, -13, 6, -60, -23, 21, 24, -6, -46, 61, -21, -95, -92, -112, -10, -31, 38, 46, 12, -56},
  {-17, 8, -44, -74, -65, -21, -6, 24, 10, -24, 5, -12, 52, 17, 42, -2, -68, -30, -3, -25, -22, 43, -27, 0, -6, 6, -39, 18, -8, 25, 89, 77, -65, 1, -19, 49, 37, 40, -7, -10, -7, 64, -121, 1, -62, 46, -36, -29, -9, -23, -11, -14, -62, 5, 18, -30, 17, 77, 62, 42, -13, -4, 67, -26},
  {-2, -3, -32, 10, 78, -25, 21, -7, 7, -60, -6, -29, -40, -13, 14, -1, 50, 46, 83, 0, 5, 4, 28, 52, 33, -5, -16, -30, 57, -45, 12, -12, 16, -4, 7, -71, 97, 17, -1, 29, 15, -6, -107, 5, 42, -37, 8, 0, -8, 15, -3, 15, -86, 20, -19, 25, 89, 34, -25, 43, -69, -6, -1, 32},
  {27, -6, 27, 15, 49, -11, 18, -2, 1, -2, 2, -5, -8, 32, -17, -7, -9, -25, -4, -11, 8, -41, 16, 60, -18, -6, -29, 11, 86, -7, -55, 54, 32, -24, 10, -5, -63, -56, -15, -11, -45, 50, -27, -37, 27, -28, 32, 14, -8, -18, 58, -15, -11, 21, 23, 23, -24, -101, -15, 26, 45, 59, 14, 6},
  {-5, 40, 10, -51, -104, -34, 23, -41, -24, 0, -12, -46, 32, 32, -27, 93, -107, -31, 23, 7, -31, 8, 76, 39, -8, -11, -36, 24, 80, 47, 40, -125, -1, -51, -26, -42, 35, 22, -45, 8, -66, -23, 81, -51, -13, -35, 10, -8, -43, 14, 54, 63, 59, 39, 43, -2, -89, -15, 1, 52, 26, 28, 5, 65},
  {24, 64, 45, -23, -30, 29, -17, 1, -62, -34, -20, -11, 62, -119, -31, 38, -102, -25, 64, 11, -14, 60, 43, -34, 23, -40, -28, -31, 18, 30, 91, 66, -60, -3, -7, 15, 54, 48, 16, 51, 92, -106, 73, 35, 86, 97, -13, -53, -2, -4, 3, -60, 74, 6, -21, -39, -18, -71, -49, -50, -8, -64, -38, 22},
  {13, 28, 2, -15, 11, -4, -20, 39, -70, -17, 9, -12, 20, -21, -52, -25, 69, 24, -18, -9, 1, 1, 21, -43, 0, 32, 11, -59, 47, 50, 37, 52, -23, -8, -42, 13, -3, 39, 35, 67, -53, -40, -24, 56, 5, 12, -42, -64, -17, -6, 20, 10, 31, 48, 5, -95, -46, 68, 58, -53, -15, 25, -16, 4},
  {19, 43, -33, -47, 82, -18, 2, -5, -61, -18, 23, -29, -6, 13, -7, -14, -22, -31, 8, -12, 20, -27, 35, -46, 16, 15, 22, 12, 42, 4, 41, 23, -13, -3, 38, 3, -113, 8, 15, -1, -10, -22, -100, -16, 34, -37, 7, -25, -14, -16, 44, 58, -93, 12, 1, -48, -77, -55, -13, -31, 33, 35, -11, 3},
  {41, -23, 22, -71, -32, 9, -29, 1, 4, 38, 15, -1, -26, 0, 15, 12, -96, 20, 17, 11, 33, 5, 18, 23, -5, 3, -2, 21, -37, -38, 92, -28, -37, -8, 1, -51, -103, -2, 16, 9, 10, 8, 68, 48, -67, 14, 10, -1, 15, 7, -11, -25, 15, 35, -6, 20, -38, 16, -8, -34, -89, 21, -7, 0},
  {26, 3, -39, 9, -43, 3, 1, 2, -32, -21, 13, -18, -47, -18, -11, 3, 89, 61, 5, -13, 10, -25, -14, 17, 21, -14, -14, 4, -47, -21, -52, -37, 13, -4, 19, -53, -88, -30, 8, 0, -48, 7, -78, 19, 31, -43, 34, -36, -8, 9, -8, -64, -7, 53, 14, 21, 58, 62, -42, -44, -47, 3, -2, 36},
  {39, 13, 45, -13, -12, -36, 0, -17, -4, -9, -9, 13, -19, -7, -2, -68, -73, -3, -2, -20, -1, -7, 30, 30, -9, -4, -9, 27, 20, -5, -41, -86, -6, 6, -5, 15, 72, -32, -23, -5, 52, -5, -65, -6, -74, -13, 19, -5, 16, -4, 17, -78, -32, 5, 17, 7, -6, -23, -19, 4, -12, 11, 16, -5},
  {5, 58, 31, 24, -66, -7, 64, 10, -53, 1, 12, -30, 16, -45, -27, -23, 43, -18, 23, -28, -16, -46, -6, -20, 25, 1, 11, -15, -51, 35, 28, 18, -28, 31, 46, 2, -26, 63, -42, -3, 47, -51, -72, -23, 56, 4, 11, -1, -31, -38, 7, 47, -99, -16, 41, -32, 17, -71, -17, -38, -7, 26, 3, 70},
  {-20, 38, -17, -66, -66, 62, 22, -19, -5, -17, -3, -1, -8, 12, -14, 25, -128, -36, -3, 23, 18, -5, -6, 12, 42, -22, -4, -43, -98, 7, -26, -84, 26, 43, -38, -15, 42, -27, 18, 28, -74, -34, 60, -45, 2, 15, -37, 19, -10, 7, 47, 28, 88, -16, -41, -30, -114, -88, 16, 0, -70, 34, 13, -14},
  {21, -20, -128, -76, 61, 33, -13, 0, -27, -50, 9, 44, -32, 59, 10, -41, -25, 64, -14, 30, 42, -16, -6, 35, 25, 18, -1, -27, -111, -3, -33, -56, 4, -99, -64, -56, -20, -23, 60, 41, 53, -18, -56, 15, -16, -87, 12, -96, -73, 33, 75, -96, -101, 16, -50, -77, 39, -60, 121, 26, 5, 69, -87, 9},
  {35, 21, 42, -54, 87, -1, 24, -12, -43, 12, 3, -41, -29, -38, -8, 14, -18, 20, -16, -48, -10, -21, 13, -10, 6, -40, 30, 29, 42, 26, -59, -41, 31, 46, 27, 9, -42, -33, 2, -19, 41, 43, -89, -47, -42, -34, 42, -3, 13, -10, 15, -72, 9, 26, -7, 23, -70, -110, -47, 40, -89, 31, 8, -7},
  {21, -29, 48, -30, 43, 12, 4, 1, -8, 36, -6, 16, -25, -35, 7, 30, 36, 0, 20, 13, 30, -38, -14, 39, 3, -9, 5, -6, 23, -37, -18, -66, 23, 25, 57, -51, 44, -11, 21, -4, -83, 23, -27, 12, -3, 12, 55, 12, 20, 5, -11, -40, 17, 19, -3, 10, -110, -110, -83, -91, 40, -3, -11, 16},
  {25, 8, -48, -54, 17, 30, -8, -21, -16, -1, 6, 13, -22, -10, -9, 4, -5, -4, 9, -16, -16, -35, 5, 21, 16, -20, -30, 4, -39, 3, -51, -9, 30, -18, 26, -13, 91, -21, -2, -23, 101, 40, -30, -24, -81, -38, 43, 8, -8, -6, 27, -37, 43, 21, 13, 19, -53, 49, -65, 9, 73, 0, -15, -1},
  {30, 24, 12, -29, -52, 6, 24, 24, 2, -20, 19, 27, 19, -9, 14, -24, -78, -39, -4, -48, 6, -4, -15, 6, 8, -22, -14, -15, 51, 25, -43, 73, -10, 21, 37, 48, 1, 51, -26, -35, -60, 56, 39, 9, -19, 35, 2, 25, -7, 2, -7, -47, 52, -34, 25, -41, -27, -109, -46, -52, 85, -30, 20, 21},
  {-18, 10, 20, 28, 73, 4, 53, 15, -15, -10, 17, 3, 20, 23, 9, -26, -53, -17, 23, -9, -1, -13, 43, -5, 54, 10, -7, -10, -26, 6, -42, -110, -12, -18, 16, -19, 82, 14, -24, -47, 32, 40, 59, -75, -43, -41, -15, 16, -8, -12, 12, -12, -88, -1, 9, -11, -52, -71, -48, -56, 77, -9, -50, -7},
  {-28, 17, -22, -20, 59, 63, -51, -60, 34, -34, -14, -8, -95, 33, -18, 29, 49, -44, 22, 14, 31, -4, 36, 33, 22, -46, -7, -42, 17, -17, 4, -61, 65, -54, -28, -103, 18, -82, 29, -6, -3, -13, 11, -37, -11, -74, 11, 4, 27, 43, 88, -41, 85, -21, -65, 1, 71, -49, 8, -21, -79, 4, -46, -19},
  {10, -7, 6, -101, 25, 17, -2, -74, 34, 9, -30, -2, -4, 5, -8, -35, 80, 77, -51, 45, 8, 32, -3, 63, -17, -29, -15, 47, 80, 53, -83, -45, 78, -51, -86, 1, -87, -83, 16, -12, -38, 98, -36, -63, 22, -59, 5, -34, 29, 72, 9, -112, -62, 18, -33, 44, 75, -91, 69, 34, -31, 33, 24, -12},
  {11, -6, 58, -65, 28, -27, 28, 1, 1, 36, 16, -55, -21, 4, -3, 27, -84, -10, -20, 4, -9, -8, -4, 12, -31, -36, 15, 20, 5, 8, -83, -93, 12, 65, 46, 6, -23, -4, -10, -49, -63, 55, -34, 31, -105, 21, 35, 31, 43, 1, -4, -4, 42, 10, 42, 12, 78, 68, -42, -78, -89, 19, 17, -16},
  {16, -2, 3, -108, 36, 0, 30, 1, -8, 40, 8, 13, 9, -48, -3, 7, -116, -67, 3, -14, 3, -30, -25, 6, 31, 12, 4, -5, 77, -30, 0, -64, 19, 45, 52, 10, -9, 4, -6, -5, 42, -29, -79, -22, 22, -13, 20, 28, 4, -6, 19, 38, 25, -10, -3, 10, -79, 5, -90, 20, -119, -26, -3, 18},
  {1, 3, -40, 25, 49, 30, 4, 7, 9, 16, 8, 28, 11, 4, 15, 16, 22, -41, -37, -4, 4, -5, -9, 2, 9, 7, -27, -5, 56, 6, -48, 8, 0, -38, 6, 23, 44, 1, -7, -2, 34, 0, -95, -26, 5, -14, 10, 10, 39, -22, 24, -81, -97, -23, -11, 22, -32, 94, -49, 40, -24, -11, -43, -2},
  {30, 14, -51, -28, 2, -12, -12, 25, -4, -23, -1, 25, -6, -35, -7, 11, 17, -11, -43, -23, 9, 19, -26, -31, -10, -22, 4, 7, -77, 31, 37, -88, -6, -8, -2, 1, 21, 55, 18, 0, 85, 9, -57, 40, -28, -1, -6, 10, -12, -7, -2, 7, 5, -40, -14, 7, -52, 19, -47, -90, -39, -51, -22, 40},
  {54, -25, -17, -52, 29, 53, -3, 1, 19, -25, -16, 21, 38, 0, 39, -23, -7, -35, -12, 6, 22, 25, -25, -1, -25, -41, 22, 19, 77, -6, -29, -1, 37, 31, 8, -14, -92, 4, 11, 20, 54, 56, -124, 41, -19, -19, 27, 6, 4, 4, 12, 3, -48, -8, -44, 0, -1, 39, -2, -5, -65, -13, -41, -38},
  {29, -24, 66, -91, 21, 1, -19, -28, 10, 45, -22, 22, 37, -7, -16, -32, -101, -67, -24, 42, 8, 41, 86, 44, -76, -124, 8, 19, -84, 56, 10, -7, 51, -14, -40, -29, -7, 13, 14, -26, -47, 76, -80, 72, 2, -53, 73, -60, 38, 61, -2, -41, 23, 27, -3, -34, -42, -97, 26, 38, -80, -10, 2, -17},
  {-21, -9, 9, -84, -112, -13, 49, 30, -10, 37, -36, -23, 88, -6, 10, 51, 38, 0, -28, -7, -31, 25, 45, 57, -46, -76, 42, 57, 20, 30, -104, 9, 70, 42, -8, 29, -68, 12, -15, -14, -32, 104, 9, -28, -81, -4, 64, -19, -14, 25, -20, -3, 62, -6, 1, -14, -41, -42, 15, -39, -77, -7, -7, -17},
  {6, 4, -15, 30, -13, 0, 13, 6, -12, 21, 2, -32, 13, -16, 12, -12, 10, -26, 11, -32, -9, -7, -7, -16, 30, 3, 27, -6, 6, 4, 33, -94, -2, 15, 25, -19, -78, 7, 3, 22, -94, -17, -83, 3, -41, 15, 7, 11, 0, -15, 2, -40, 89, 0, 0, 12, -66, 24, -38, -104, 25, -12, -19, 25},
  {-15, 25, -7, -72, 91, 48, -18, -18, 39, 4, 3, 17, -9, 2, 18, -15, 63, -69, 34, -17, 12, 7, 19, 2, 49, 28, -44, -6, 18, -36, 18, 5, -12, 0, 1, -23, 46, -32, -1, 23, -71, -13, 16, -34, -57, 13, -23, 22, 25, -8, 24, 42, 27, -25, -20, 31, -90, -76, -37, 15, -76, -32, -5, -13},
  {-20, 3, 0, -86, 20, -1, -1, 0, 15, -37, 0, -4, -17, 18, -6, 48, -9, -35, 14, 69, 11, -21, 17, -51, 11, 27, -22, 19, 62, 6, -15, 46, 16, -14, 23, -3, 45, 20, -7, 0, 37, -29, -9, 0, -13, -13, 13, 33, 16, -30, 20, -107, -18, -38, -5, -20, -77, -8, 15, 34, 77, 5, -3, 42},
  {1, 4, 12, 39, 17, -17, -4, -3, 2, 28, 1, 18, -36, -3, -40, -47, -24, -11, -11, 7, -5, -6, 20, -23, 0, -13, -37, 0, 8, 13, 25, -49, -20, 21, -17, -20, -94, 11, 12, 4, 75, -23, -2, 20, -22, -15, 3, -8, 29, 16, -23, -57, -51, 5, 0, 21, -8, -13, -19, 105, 18, -21, 15, -8},
  {-21, -9, -9, -57, -8, -20, 23, 18, -10, -35, 25, 22, 41, 24, -6, -56, -122, 2, -37, -25, -1, -7, -13, -20, 9, 30, 15, -6, 82, 8, 11, -9, -40, -8, -9, 28, -18, 17, -1, 4, 17, 37, 60, 8, -36, -42, -21, -40, -21, -50, -44, 22, 42, 10, 24, -20, -49, -81, 19, -48, -21, -16, 22, -20},
  {-10, 3, -60, 4, -29, -12, 20, 28, -29, 21, -4, -12, 41, 8, 14, -6, 6, -20, -35, -7, -34, 3, -26, 17, 17, -31, 35, 62, -20, -34, -38, -5, 28, 15, -10, 18, -21, -22, 6, 14, -48, 19, -8, -39, -7, -68, 8, -31, -42, -31, -14, 72, -64, 12, -26, 20, 21, 53, -19, 45, -68, -19, -18, -14},
  {2, -15, -4, -9, -17, 16, 21, 10, 17, 6, 6, -45, 11, -7, 42, 28, -6, -9, 10, -35, 19, 1, -16, 7, 19, 23, 9, 36, 12, -18, -4, -37, 8, -20, 24, 1, -53, -16, 0, 5, 17, -3, 62, -24, -2, 11, -1, 21, -5, -10, -5, 77, -73, 0, -13, 18, 26, 27, -5, -91, -125, -6, -22, 5},
  {-40, -3, 5, -38, -97, 25, 18, -20, 40, -3, 15, 11, 7, -7, 24, 24, -27, -38, 39, 21, 21, 10, -14, -18, 20, 22, -33, -39, 84, -26, 47, -81, -16, -1, 11, -1, 5, 21, -4, 31, 73, -33, 69, -21, 21, 25, -34, 24, 12, -11, 5, -68, -8, -24, 1, 11, -18, -16, -35, -19, -99, -27, 20, -1},
  {-2, 28, 20, -98, 37, -18, 2, -2, 4, -13, 23, 0, -20, 0, -3, 24, 32, -52, -9, 38, 12, -5, -9, 11, 7, 3, -62, -10, 54, -30, -13, 67, -21, -4, -1, 12, 67, -11, -5, -10, -91, 13, 71, 12, -1, 23, -1, 5, -10, -4, -6, 53, -95, -37, -3, -67, -2, 7, -10, -39, -32, -29, 16, -26},
  {-7, 23, -4, -50, -37, -1, -26, 7, -5, -4, 7, -20, -17, 1, 1, 1, 75, -29, 17, 48, 18, -1, -1, -35, 5, -9, -15, 37, -76, 8, 38, 29, 15, 1, 2, 15, 13, 27, -1, -11, -11, -34, -88, 45, 35, 45, -9, 21, 2, -19, -34, 58, 5, -14, -1, -33, -60, 35, 22, 51, 19, -23, 29, 34},
  {0, -40, 6, -56, -52, 20, 22, 14, 12, 29, -25, -15, 50, -5, 4, 11, 69, -49, -15, 7, -27, -6, 2, 31, -6, -7, -15, 5, 71, -31, -75, -71, 53, -6, 8, 20, -111, 36, 9, -24, 120, 51, 0, -17, -28, 3, 40, 25, 3, -6, -10, 76, -31, 8, -14, 31, 81, -58, -5, 98, -22, 1, -24, -8},
  {-20, 6, 6, 35, -48, -58, 1, 20, -1, 15, 10, -25, -11, -10, 0, 12, -44, -25, 2, 25, -17, -1, -1, -21, -1, 41, 5, -16, -48, -19, 23, -22, -35, -49, 31, 17, -125, 39, -4, 1, 3, -35, 46, 2, 6, 49, -15, -1, -11, -4, -40, 42, -95, -10, 0, -7, 1, 31, 9, -39, 5, -1, 15, 59},
  {-14, -8, -2, 9, -18, 22, 8, 3, 21, 37, 17, -3, -32, 41, 35, -9, -73, 54, -6, 18, 25, 3, -13, 6, 5, 13, -7, 43, 49, -35, 13, -93, 1, -3, -6, 6, 19, -43, -6, 1, 54, 4, 12, -29, 2, 35, -24, -6, 6, 3, 5, -97, -30, 11, -24, 14, -11, 44, -8, 21, 37, 32, 2, -21},
  {-19, -2, 28, -34, 74, 6, 8, -3, 0, -9, 28, -14, -8, 10, 19, 23, -110, 6, 12, 47, 35, 0, -10, -27, 24, 45, -12, 11, -98, -8, 11, -124, -18, 7, 37, 2, -96, 2, 7, 28, -26, -28, 44, -3, 4, 40, -20, 27, 8, 2, 16, 8, 68, -17, 3, 8, 11, 10, -12, -40, -81, -5, 0, 15},
  {21, 10, -17, -77, -59, 19, 42, 5, 8, -20, 33, 29, -26, 23, 10, -23, -63, -23, 6, 40, 13, -3, 2, -2, -4, 2, -28, 1, 12, -28, -8, -81, -22, 20, 17, 23, -91, 2, -3, -19, 61, 7, -128, -8, -48, -1, -24, 1, -35, -7, -14, 7, -12, 8, -9, -26, -124, -43, 8, -48, -75, -14, 31, -65},
  {2, 13, -18, -16, -103, -16, 19, 9, 29, -19, 32, -4, -13, 22, 20, 28, -123, 8, 27, 59, -5, -5, -25, -2, 5, 10, -18, -15, 11, -17, -3, -28, -20, 21, 7, 8, -79, -18, -7, -12, 45, -10, -76, -7, -17, -1, -25, 38, -68, -17, -28, 22, -62, -41, 7, -11, 26, -2, 23, 17, -18, -23, 33, 1},
  {-13, -16, -23, 17, -22, 2, 55, -16, 10, -19, 28, -4, -5, 17, 28, 1, 11, -7, 5, 17, 18, 2, -28, -3, 3, -30, 28, -16, 57, 7, 27, 24, 11, -12, 22, 13, -96, 6, -12, -27, 31, 16, -82, 15, -25, 34, -23, 20, 8, 22, -5, 19, -4, -18, 32, -20, -5, -63, 2, -17, -35, -8, 35, 26},
  {42, 45, 15, 26, -84, 52, -10, -4, 9, 10, 20, 14, -51, 41, -10, 6, 6, -38, -40, -33, -2, 2, 53, 43, 1, -23, -25, 65, -56, 2, -6, -84, -5, 21, -45, 14, -10, -39, 1, -1, 62, 25, -66, 15, 3, -8, -38, -12, -8, 16, 31, -37, 67, -26, -12, 39, -33, -17, 15, -50, 83, 36, 4, -1},
  {12, 51, -11, 27, 11, 48, -11, 13, -16, -11, 0, 5, -19, 5, -7, 6, 16, -38, -6, -5, 5, 8, 35, 0, 1, -8, 45, 1, -46, 1, 29, -125, 12, -33, -39, -42, 28, -19, 2, 23, 17, -32, -49, 18, 49, -6, -38, -16, 18, -1, 55, 49, -55, 8, -26, -37, 42, 98, 12, -64, -5, 13, -8, -14},
  {23, 37, 17, -35, -4, 11, -68, -41, -5, -2, 0, -62, -17, -10, -35, 74, -34, -49, -21, 13, 10, 7, 65, -2, -10, -14, 10, -21, 63, 29, 15, 64, -2, 3, -36, -10, -109, -44, 16, -15, 34, -20, 29, -1, 63, -6, -18, 1, -13, 19, 33, -76, 72, 5, -1, 0, 68, -64, 20, 47, 36, 0, 17, -12},
  {14, 21, -1, 15, -36, 21, -22, -53, -12, 11, -31, -60, 38, -12, -33, -13, 73, -61, 24, -1, -16, -32, 42, -10, 29, -2, 9, 14, 46, 15, -48, -27, 35, -18, -12, 3, -66, 8, 4, 7, 49, -11, -13, -21, -9, -8, 22, 7, 20, -13, 40, -25, -80, 29, -7, 27, -9, -101, 9, 53, -58, 7, 9, -15},
  {-14, -12, -66, -2, -12, -37, 39, -5, -14, -38, -32, 4, 46, 14, 1, -40, -35, -10, 42, -6, -39, -9, -4, 45, 18, 15, -44, -23, 29, 21, 32, -38, 8, -2, -11, 1, 48, 52, 1, 21, -81, 10, 33, 50, 34, 4, 32, -37, -11, 2, 19, 30, -16, 66, 6, 5, 57, -2, 22, 55, 3, 23, -19, 53},
  {33, 11, -46, 9, 80, -36, -4, 4, -4, -11, 1, 4, -7, -49, 2, -56, 74, 18, 26, 7, 12, 13, 27, 58, 15, -20, -29, -11, 73, -25, -9, -49, 0, -8, 41, -46, -3, 2, 17, 5, 76, 27, -68, 38, 89, -27, 33, -13, -3, 39, -16, -50, -91, 20, -4, -16, 21, 60, -30, 114, 57, 9, -4, 12},
  {15, -14, 23, 50, 31, 6, -10, -8, 15, -18, 12, -1, -22, -2, 12, -25, 35, -66, -10, 31, 17, 3, -6, 32, -24, -7, -8, 39, 68, 5, -6, 57, -50, 6, 43, 38, 43, 18, 12, -18, -25, -18, -13, 57, -5, 5, 16, 16, 0, 11, -9, -33, 2, 8, 12, -4, -104, -4, 4, 60, -85, -18, -37, 2},
  {28, 30, -15, 1, -69, -35, -17, -29, 12, -4, 11, 23, -71, 33, 1, -68, -107, -13, 19, -19, 48, -15, 25, 26, 11, -24, 9, 10, 107, -17, -4, -5, -5, 2, -20, -94, -80, 16, 5, 20, 67, -22, 68, -2, -54, -10, 11, 4, -4, 31, 75, 30, 80, -33, -15, -19, -42, -18, -29, 66, 61, 15, -41, 25},
  {-19, 5, 17, -107, 60, 25, -48, -35, 44, 32, -13, -21, -78, 24, 29, -2, -47, -53, 15, 3, 46, 4, -1, 47, -17, -8, -9, 17, -17, -9, -10, 21, -22, -16, -41, -37, -42, -81, 21, 21, 24, -23, -16, 10, 29, 7, -40, 11, 13, 9, 9, -117, -41, -39, -16, 21, -58, 72, -9, -63, -31, -10, -1, -35},
  {-23, -19, 13, -44, 30, -19, -60, -36, -3, 25, 18, 24, -70, 55, -4, -10, -53, -72, 4, 7, 34, -13, -24, 39, -50, 37, 32, -36, -55, 30, 37, -10, 2, -48, -7, -63, -21, -93, -3, -1, -59, -16, -5, 41, -90, -1, 26, -3, -10, -2, 44, 8, 25, -12, 6, 39, -77, -84, 53, -59, -88, 41, 11, 21},
  {-24, 8, 0, -36, -84, -5, 73, -13, -25, 9, -14, 35, 51, -15, -18, 14, -119, -46, 3, 37, -39, -39, -16, -3, 20, 18, 21, 17, -83, 15, -60, 43, 50, 11, 43, 40, 11, -32, -13, 0, 22, -44, 23, -29, -36, -21, 19, 13, -11, 1, 17, -77, -3, 15, 12, 34, 0, 21, -8, -108, -125, 12, 3, 10},
  {-14, -5, 0, -91, -120, 13, 41, -5, 16, 4, -10, 29, 50, -10, -10, 8, -116, -44, 25, 30, -34, -18, 0, -26, -3, 2, -14, 16, -58, -13, 12, -45, 17, 16, 38, 11, 13, 43, -6, 0, -49, -41, 56, -5, -4, 29, 10, 23, 15, -10, -11, -28, 59, -4, 3, -2, 50, 19, -10, 25, -26, -24, 8, -16},
  {23, 13, -16, -46, -48, 2, 24, 6, 0, 16, -12, 3, -9, -31, -21, -15, -11, 10, 1, 2, -30, -2, -16, 1, 5, 12, -22, -6, 17, -25, -24, -36, 29, 10, -10, 6, -73, 20, 25, 10, -31, -26, -9, -1, 4, 38, 14, -19, 9, -3, -29, -9, 27, 9, 9, 39, -7, -43, -25, 42, -44, -10, 19, 16},
  {18, 13, -12, -4, 20, 9, -24, 12, 8, 3, -7, -15, -77, -57, 2, 1, 27, 34, -13, -19, 23, -22, 12, 22, 30, -19, 35, 22, -66, -20, -54, -3, 46, 12, 51, -19, 71, 9, 5, 7, 105, -60, 83, -2, 117, 20, 30, 33, 26, 7, -45, -41, -53, -26, -20, 28, 5, 49, -63, -68, -81, -62, -27, 12},
  {42, 8, 21, -69, -112, 1, -66, -21, -7, -9, -2, 35, -45, -23, 13, -47, -50, -79, 54, -86, 24, -31, -2, 6, 19, -53, -20, 11, -47, -7, 8, 24, 9, -20, 51, -68, -12, -8, -1, -25, 31, -11, -38, -24, -71, 29, 40, 39, -9, 21, 50, 46, -65, -16, -8, 8, -69, 8, -85, 18, -8, -19, -11, -7},
  {-5, -68, 9, -48, -110, 11, -23, -21, 39, -28, 18, 73, 78, -13, 61, -44, -33, -25, -4, -46, 20, 68, -6, 10, -49, -63, 20, 34, -27, -45, 44, -92, 17, -15, -15, 28, -93, 3, 13, -21, -2, 74, -85, 32, 18, 44, -2, -10, -24, 24, -44, -63, -64, -38, -20, 1, -61, 71, -22, -83, -12, -49, -50, -57},
  {-7, -68, 66, -11, -112, -24, -9, -3, 32, 4, -7, 36, 52, -39, 11, -2, -60, 62, -27, -41, -19, 60, 8, 13, -67, -26, 21, 34, 51, -25, -26, 57, 4, -22, 11, 61, -72, -32, -3, -34, -8, 54, -30, 7, 20, 11, 26, 12, 20, 21, -35, 43, 79, -36, 20, 39, 75, -9, -53, -67, -29, -60, -28, -74},
  {-15, -19, 42, -38, -83, -12, 67, 7, 51, 31, -46, 48, 83, -29, 3, 64, -34, -89, -12, 16, -105, 36, -40, 28, 14, -23, -14, -18, 66, 53, -54, -99, 51, 34, 3, 106, 43, 4, -30, -49, -64, 88, -30, -35, -1, 39, 32, 64, 60, 11, -39, -1, 78, -128, 42, -8, -88, -31, -24, 11, -105, -65, 29, 18},
  {-14, 30, -52, -74, -59, -3, 31, 10, 0, 19, -16, 26, 29, -23, -17, -48, -98, -75, 20, 64, -38, 23, -6, 3, 21, 45, -19, 14, 35, 35, 22, 46, 15, 20, -13, 17, -16, 27, 8, 12, 74, -74, -72, 31, 4, 2, -9, 2, -8, 33, 32, 15, -120, -45, -14, 53, 75, 41, 15, 5, -13, -28, 25, 18},
  {42, 8, -14, -107, 23, 5, -47, -3, -17, 52, -26, 19, -4, -29, -18, -39, -97, 30, -12, 4, -27, -2, 10, 13, 20, -2, 13, 33, 12, 28, -13, 40, 21, 32, -1, -12, -46, 32, 13, 9, -89, -45, -18, -4, -50, -3, 12, 7, 12, 22, 9, 36, 59, 11, -24, -6, -24, -121, -19, -83, -12, -26, 4, 41},
  {6, 0, 6, -83, -44, 30, 19, 12, 35, 13, -37, 7, -5, -28, -2, -4, 24, 47, -40, -15, -8, 15, 0, -5, -3, 21, -21, 22, -81, 2, -94, 29, 11, -17, -38, -10, -107, 20, 18, 32, -85, 31, 40, -52, -58, 42, -14, 5, 22, 18, -6, -34, 0, -19, -9, 22, 59, -96, -4, -17, 4, -8, 11, 34},
  {49, 10, 41, 2, -2, -9, 39, 4, 8, 12, 32, 9, -13, 15, 26, -50, 92, -43, 54, -29, 10, -35, -19, -36, 3, 43, 15, 53, -79, 20, 28, -37, -39, -35, 42, 4, -15, 24, -22, -26, 11, 39, 68, 26, -30, -12, 3, 33, 56, -33, -10, 30, -28, 29, 38, 8, -87, -11, -62, -62, 58, -4, 21, 7},
  {50, 12, 27, -21, 90, -7, -7, 44, 14, 12, -13, -8, 36, -69, -16, -28, -47, -19, -43, -52, -15, 14, -15, -7, -27, -23, 7, 48, -23, -17, -6, -59, -8, 40, 11, 38, -42, 53, 19, -3, -11, 54, -30, 62, -44, 51, 17, 25, 19, -7, -17, 23, 88, -56, 4, 23, -50, 71, -76, -88, 80, -29, -5, -3},
  {43, 6, 36, -41, -46, 25, 19, 52, 30, 49, 3, 10, 47, -40, 52, 10, 37, 1, 7, -25, -6, 31, -21, 16, -30, 3, 1, -13, -11, -56, -11, -115, 25, 43, 67, 60, 16, 97, -13, 3, 77, 27, -36, 29, 10, 95, 22, 72, 4, 6, -45, 16, -116, -80, 22, -50, 76, -80, -40, -91, -27, -48, -12, 3},
  {-5, -10, 26, 17, -67, -22, -41, 9, -7, -7, -8, 37, -12, -6, 14, -64, -40, 25, 41, 14, 15, -24, 20, 8, 3, 38, -18, -47, -96, -28, 39, 2, -21, -8, 19, -33, 2, 35, -11, -19, -46, -45, -64, 59, -85, 61, 39, 3, 35, 7, -34, 14, -102, 7, 26, -75, 25, -26, 18, -63, -108, 1, -18, 4},
  {-18, 19, 46, -72, -115, -36, -82, -45, 27, -41, -21, -25, -45, 18, -17, -62, -125, -81, 72, 54, 22, 0, 34, -2, 29, 47, -35, -60, 5, -27, 87, 69, 9, -22, 17, -68, 40, -2, -12, -4, -5, -63, -101, 21, -65, 38, -1, 36, 47, 8, 17, -2, 43, -8, 12, 24, -1, -110, -3, -32, -30, 0, 32, 33},
  {6, 4, 12, 44, -69, 6, -95, -13, 50, -37, -14, -5, -8, -11, 15, 8, -11, -59, 48, 55, 8, 47, 43, 14, 7, -5, 8, -18, 43, 8, 23, 61, 6, -3, 5, -23, 9, 46, 9, 17, 42, 23, -112, 46, 11, 28, 8, 30, 13, 22, -6, -99, 5, -17, -9, -45, 2, 29, -18, 4, -80, -3, -14, 27},
  {17, 21, -8, -63, 70, 7, -18, 15, -20, -6, -8, 14, 30, 0, -33, -7, 52, -55, -17, -36, -22, 1, 26, -36, 18, -21, 9, -17, 93, 46, 21, -74, -34, -33, -60, 31, -73, 25, 30, 52, 91, -39, 39, 14, -121, -17, -15, -32, -34, 7, -5, -115, 72, 13, -17, -16, -29, -22, 8, -103, 21, -24, -29, 46},
  {21, -6, -7, -13, -115, -12, -26, -13, -1, -6, 8, -12, -50, 76, -9, 5, -13, -23, -9, -29, 22, 1, -58, 13, -6, 28, 9, -10, -22, 10, 12, 58, -2, 29, -5, -9, 60, -2, 27, 28, -49, 3, 7, -4, -31, 8, -9, -13, -29, 5, 3, 91, -109, 50, -9, 39, 20, 23, 10, 18, 24, -9, 25, 27},
  {0, 21, -12, -72, -107, 54, 25, 51, 23, -6, 13, 0, 11, -22, -13, -70, 31, -43, -13, 0, -19, 17, -3, -45, 3, 23, -74, 10, -105, -18, -21, -37, -11, -13, 15, -20, 30, 69, -4, 0, 30, 29, 20, 29, -21, 20, 0, 20, -10, 0, -10, 41, -91, -13, -17, -8, -77, -14, -1, -109, -127, 21, -20, -7},
  {-24, -7, -37, 44, -72, 19, -21, 30, 29, -8, 11, 1, 13, 6, -25, -37, 7, 18, -5, -27, 17, 2, -13, -3, -15, 30, -37, -38, -41, -4, 15, 18, -25, -34, -23, -12, -24, 21, 12, 19, 66, 37, -21, 36, -51, 12, -4, -24, 12, -8, -6, -101, 49, -32, 1, -22, -122, -118, 30, 18, -83, 36, -32, 32},
  {-50, 18, 65, 12, -83, -79, -52, -12, 10, 1, 5, -43, -20, 40, -18, -30, -64, 30, 58, 44, 8, -22, -10, -24, -10, 76, -19, -60, 21, -14, 29, -86, -40, 27, 37, -11, -29, 41, 6, 21, 25, -85, 29, -3, -56, 61, -9, 60, -17, -19, -19, -72, -24, -44, 42, -47, -85, -107, 0, 56, -95, -37, 44, 47},
  {-42, -5, 59, -99, -62, -47, -53, -27, 46, -19, -10, -54, -22, 14, -17, 43, -70, -102, 57, 60, 1, 31, 18, 17, -39, 2, -41, -65, -74, -7, 55, 38, -12, -29, -32, -17, -11, 7, -4, 0, -91, -19, -68, 53, -103, 75, 4, 2, -16, 4, -19, -103, -16, 13, 31, -82, -112, -122, 1, -100, 29, -11, 32, 0},
  {-11, 0, 8, -9, -10, 31, -45, -25, 12, -15, 2, -36, -26, 9, 0, 43, 46, -80, 21, 8, 18, 24, 23, 8, -19, -28, 13, -31, -36, -5, 41, 67, 8, -14, -20, -25, -42, 35, 8, 9, -103, -23, 63, 34, -45, 28, 5, 17, -15, 15, -8, -46, 47, 23, -15, -44, -115, 70, 4, -68, -67, 7, 5, 7},
  {-21, 43, 2, 22, -105, -3, 3, 17, 3, 25, 0, 32, 12, 8, -10, -13, -29, -36, -35, 3, -6, 25, -4, -44, 11, -24, 3, 21, -61, 24, -27, -77, -18, -29, -16, 14, -2, 11, 10, 12, -28, 32, 100, -23, 30, -12, -25, -5, 30, 3, -9, -101, -22, -26, 2, -33, 0, -90, 18, 48, -80, -18, 6, -14},
  {-31, 24, 17, -15, 83, -3, 5, -10, 33, 44, -3, -15, -41, 5, -15, -2, -83, 3, -29, 74, 10, -1, 49, -58, 32, 24, -46, 35, 96, -10, -35, 36, -48, 2, -12, -30, -88, 6, 20, 4, -16, -25, 43, -18, -82, 17, -29, 23, 5, 33, 8, -68, 24, 22, -13, 29, -124, 18, -17, 76, -93, 18, 42, -23},
  {-45, 0, 35, -94, 43, -47, -43, 13, 62, 24, -4, -27, -2, -5, 3, 1, 20, -9, -25, -30, 5, -16, 8, 1, -6, -7, -48, -68, -51, -21, -67, -69, 13, -40, 17, -7, 54, 12, 3, -16, 89, 45, -22, 16, 9, 35, 29, 36, 33, 15, -44, 29, -106, -23, 1, 17, 1, 72, -39, 92, -92, -17, 5, -23},
  {24, 13, 50, -64, -118, -42, -25, -4, 18, 41, 1, -13, -17, 16, -2, 3, 53, -53, 33, -34, -1, 35, 47, 44, -19, -11, -20, -21, -60, 19, -11, 73, 16, 17, 5, -19, 51, 5, -3, -3, -21, 41, -9, -9, 15, 26, 29, -1, 21, 28, 10, 8, -44, -6, 36, -23, 38, -52, -1, -29, -82, 17, 12, 1},
  {-21, 37, 19, 9, 78, -33, 14, -1, -5, -5, -33, -8, 17, -3, -43, 53, -90, 34, 37, 43, -28, 40, 30, 0, -8, 9, -19, 0, -96, 24, 28, 55, 12, 31, -45, 4, 17, -4, 14, 24, -37, -6, 16, -59, -34, 9, -10, -3, 8, 8, -17, -14, 86, 7, 13, -8, 19, -86, -4, -55, -3, 1, 6, -10},
  {20, 4, -34, -54, 4, -24, 26, 11, -45, -45, -12, -26, 35, -7, -30, -10, 61, 72, 34, 17, -16, 13, 1, -2, -31, 9, 29, -22, -7, 0, -6, -67, -16, -22, 3, -26, -79, 27, 14, 11, -103, 19, -63, 0, -70, -78, 17, -40, -99, -8, 22, 42, -75, 33, 0, -39, -104, -119, -11, 52, -27, 16, -33, -19},
  {23, 2, -23, -43, -51, 5, 44, -15, -52, -10, 16, 17, 27, 8, -22, 48, 66, -61, -12, -14, -1, -33, -28, 3, 14, -24, 23, -8, -109, 15, -46, -67, -9, -23, -18, 33, 54, -43, 14, -36, -60, 60, -33, 12, -32, -64, 25, -33, -15, -2, 23, 30, 88, 53, 18, -65, -83, -90, -12, 18, 75, 30, -20, 4},
  {17, -17, -39, 6, 87, -6, 13, 17, -24, -31, 25, 16, 28, -24, 5, 23, 3, -52, 7, 51, -3, -16, -25, -19, -12, -3, -18, 27, 12, 22, 24, 67, 9, -30, 22, 42, 39, 50, -14, -18, 1, 43, 53, 50, -3, -6, -6, -30, -34, -8, 7, -45, 6, 28, 9, -28, -107, 2, 17, -19, -22, 17, -7, 19},
  {-6, -18, 22, 6, -46, 5, -4, 22, 8, 56, 0, -21, 10, 10, 12, 49, -59, -47, -18, 6, -11, -31, -19, -3, 8, 3, 6, 35, -83, -40, -42, 45, 40, 22, 20, 27, -32, 28, 14, -15, 105, -1, -20, -22, 46, 9, 11, 16, 16, -10, -29, 83, -35, -21, -22, 30, -68, 36, 22, -48, 4, -24, -9, -12},
  {-5, 37, 20, 30, 20, -73, -29, 39, -38, 2, 18, -26, -30, -37, -30, 20, -50, 6, -5, -28, -12, 38, -2, 4, -20, 19, -33, -50, -39, 40, 24, -61, -39, -51, 26, 15, -10, 81, -29, -11, 30, -36, -64, 13, -64, 54, 1, -16, -9, -5, -31, -24, 11, -5, 23, -31, -13, 37, -5, -44, -21, -11, 22, 44},
  {3, -26, -10, -38, 86, -10, -1, 8, 26, 20, -1, 6, -35, -1, 19, -28, 43, 30, -2, 47, 19, -6, -8, 35, -2, 24, -8, 50, 28, -28, 48, -63, 3, 4, -14, -4, 67, -32, -8, 19, -8, 35, 25, 4, -31, -17, 22, 3, -13, -9, 28, -98, -18, 28, -4, -22, -39, 47, -2, -66, -34, 37, -34, -35},
  {60, 33, -12, 12, 90, 32, 59, 13, -40, -6, 50, -9, 17, -6, 20, 55, 71, -30, -46, 0, 26, -11, -31, -47, 20, -11, 33, 89, -97, -4, 50, -60, 4, 44, 5, 31, 41, 6, 5, -5, -15, 59, 65, -30, -35, -25, -27, -38, -46, -22, -6, -100, -56, 13, -36, -12, -106, -55, 45, 31, -23, 18, -27, -42},
  {41, -18, -43, -81, 48, 35, 31, -6, 4, -46, 21, 59, -15, 13, 21, 12, 72, 14, 2, 29, 20, 10, -1, 15, -26, -28, 13, 5, -78, -39, -5, -77, 12, -11, 15, 3, -43, 0, -5, -8, -120, 41, -45, -18, -15, -38, 2, -24, -63, -1, -16, 2, -121, 16, -19, -27, 9, -125, 13, -45, 44, 24, -13, -80},
  {10, -6, 15, -51, -73, -9, 43, 3, 9, -6, 19, 4, 22, -6, 5, 12, -85, -32, 12, 16, -19, 11, -41, -7, 3, 2, 11, -39, -18, -22, -27, 41, 19, 6, 24, 15, -49, 13, -17, -24, -121, 43, -54, -20, -18, -9, -17, 19, -50, -8, -11, -90, -71, -11, 16, -18, 13, 1, -7, 3, 10, -8, -6, -57},
  {13, -1, -23, -31, -1, 0, -2, 1, -28, -24, 31, -3, 11, -6, 15, 31, 34, -64, -9, 2, 25, 19, -18, 17, 18, -34, 25, -15, -90, 15, -20, -54, -7, -9, 28, 26, -79, 11, 7, -23, 78, -5, 118, 34, -53, 44, -22, -7, -5, 19, 6, 4, -2, -1, 13, -14, -22, 44, -2, -4, -40, -3, 22, 6},
  {19, -4, 18, 42, 37, -7, -11, 33, 22, 40, 32, 25, -20, 33, 14, -13, 79, -27, -19, -26, 25, 3, -3, 18, -11, -14, -3, 28, 41, -14, -10, -95, 20, -7, -22, -6, 43, -33, 7, -23, -7, 6, 73, -16, -48, -16, -12, -1, -30, 34, 11, 0, -32, 7, -6, 27, -9, 29, 32, -28, -18, 21, 9, -39},
  {10, 66, 47, 8, 109, 62, -36, -6, -26, 22, -24, -37, -26, -13, -40, -20, -121, -15, -11, -34, -29, -8, 79, 18, -2, -24, -1, 31, -40, 43, -34, -72, 18, 25, -34, 3, -3, 16, 7, 3, 109, -55, 26, 50, 14, -20, -30, 3, 36, 14, 42, -79, 6, -20, -4, -9, -73, 25, -10, 115, 75, -1, 39, 12},
  {-7, 76, 6, -27, 70, 25, -71, -27, -44, -26, -36, -43, 14, -14, -47, 26, -60, -41, 9, 25, -26, 22, 40, -60, 24, 8, 36, -23, -121, 61, 50, -126, 5, 34, -38, -23, 85, -21, 13, 53, -44, -74, 80, 34, 19, 16, -37, -8, 38, -9, 67, 14, -66, -10, 7, -32, 54, -26, 42, -90, -17, 3, 6, 31},
  {2, 39, 28, -31, -61, 11, -49, -16, -6, -19, 10, -27, -24, 44, -5, -34, 30, 16, -25, -33, 28, 33, 19, -56, 8, 20, 28, -10, 11, -8, -13, 95, -35, 9, -39, 12, 75, -13, 43, 38, -4, -37, 50, -1, 17, 49, -49, -16, 13, 2, 11, 28, -42, -15, -23, -19, -47, 45, -3, 12, -110, -24, 30, -37},
  {30, 33, -53, 16, -106, -51, -56, -19, -13, -22, 8, -18, 1, 13, 16, -59, 62, -14, -15, -12, 4, 44, 38, 17, -21, -28, -23, 13, -86, 58, 89, 46, -57, -35, -37, 32, -84, 13, 14, -1, -101, 24, 52, 76, -28, 5, -10, -49, 1, 22, -19, -88, 60, 30, -17, -34, 37, -95, 44, 21, -108, 10, 39, -17},
  {24, 57, 28, 13, 66, 11, -37, 11, -8, 24, -8, -53, -63, -35, -33, 5, 56, 40, 2, -45, 1, 58, 33, 24, 43, -13, -26, 13, -83, -22, 47, -48, 0, -1, -26, -30, -35, 6, 27, 13, 13, -18, 9, 4, 39, 11, -14, -6, 55, 32, -2, 43, 28, -37, -26, 1, -106, 55, 22, -87, -115, -23, 37, -10},
  {1, 12, 9, -1, 76, 15, -17, 26, -4, -1, 16, -15, -58, -13, -12, -25, -95, 56, 4, 14, 24, 32, 20, 2, -11, 8, -29, 5, 66, 8, -4, -21, -56, 18, 6, -4, 60, 41, 15, 17, -47, -31, 63, 66, 64, 61, -3, 10, -4, 40, -25, -52, 92, -30, -18, -26, -101, -51, 26, 25, -12, -40, -13, 62},
  {-16, -25, -26, 27, 57, 16, -16, 1, 53, 8, -24, 18, -5, 47, 10, 5, 34, -41, 34, -14, 12, 19, 47, 70, 16, -7, -4, 33, -68, -10, 6, -57, 27, -39, -16, -32, -56, -2, -15, 27, 106, 35, -5, -28, -82, -40, 5, 13, 1, 27, 65, -4, -75, 5, -3, 2, -35, 6, 7, -32, -47, 0, -40, 6},
  {37, 43, 43, 24, -114, 25, -8, -13, -2, 11, 12, -5, -49, 23, -15, -35, -89, -15, -29, -26, 11, -5, 34, -14, 4, -1, 11, 16, 22, -20, 13, 6, -24, 29, -2, 26, -50, -59, 10, -4, 44, -44, -21, 7, 22, 26, -30, 7, 21, -4, 36, 51, -46, 5, 19, -19, -65, 58, -36, -32, 43, -9, -1, 20},
  {22, 27, -4, -25, 73, 21, -36, -7, -50, -32, -9, 9, 21, -35, -38, 54, -28, -38, 16, -11, -6, 31, 26, -22, 3, -7, -8, -86, -121, 28, 34, -5, -28, 31, 8, -24, 35, -7, 25, 36, 76, -78, -15, 32, 36, 4, -14, -49, 1, -12, 75, -24, -23, 30, 4, -69, -38, 96, 24, -70, -43, -9, -46, -38},
  {60, 34, 9, 18, 85, -32, -82, 1, -52, 19, 24, -32, -56, -14, -35, -42, -118, -15, -8, -53, 54, -4, 13, -16, 14, 18, 37, -24, 67, 8, 39, 63, -15, 25, -3, -39, 39, 1, 55, 7, 18, -57, -90, 54, -46, 11, 4, -18, 5, 38, 58, -35, -24, -8, -10, -62, 62, -56, -28, 52, -39, 24, -12, 26},
  {36, -3, 2, -2, -2, -30, -61, 5, -24, 18, 20, -24, -41, -23, -10, -13, 36, 5, 10, 5, 52, -49, 22, 13, 2, 27, -1, -28, 54, -4, 103, 35, -66, -3, 20, -37, 55, 20, 22, 13, -98, -6, 21, 69, -95, 9, 28, -10, 7, -1, 28, 19, 61, 21, -8, -12, -100, -128, -23, -45, 3, 22, -24, 37},
  {16, 59, 24, -5, -127, 17, -15, 15, -13, 9, 5, -34, -27, -13, -14, -2, 71, 72, 25, -6, 18, 13, 23, -1, 51, -10, -19, 23, -17, -32, 35, 33, -1, 11, 18, -34, 6, 9, 5, 24, 66, -36, 70, 13, -28, 1, -15, -4, 14, 18, 4, 19, -15, 3, 8, 24, -82, -62, -27, -78, -71, 1, 50, 17},
  {-15, -3, -40, -59, -103, -24, -24, 34, -29, -19, 13, -2, -51, -17, -16, 41, -68, 51, -10, 3, 13, -20, -13, -48, -1, 18, 15, -31, -9, 44, -13, 60, -58, -25, 14, -5, 109, 15, 31, 35, -9, -43, -102, 50, 50, 10, 13, -27, -4, -5, -7, -99, 55, -32, -28, -46, -11, -71, 7, 11, -84, -37, -48, 122},
  {10, 56, -7, -28, -56, -61, 25, -29, -54, 3, -24, -15, 21, -7, -26, 51, -60, -36, 25, 33, -45, 14, 28, -3, 3, 2, 2, 27, -94, 51, 20, -91, 7, 13, 12, 7, 8, 29, -6, 6, -57, -18, -69, 7, -67, -54, 19, -23, -13, -4, -4, 50, -70, 2, 17, -24, 54, -51, -18, -35, -78, -24, 19, 48},
  {-23, 29, -40, -46, 50, 23, 49, 4, -27, -21, 6, 27, 42, -4, -19, 7, -65, -3, -23, 25, -18, 6, -10, -16, 51, 18, 15, -3, -64, 2, -21, -41, -9, 36, -31, 52, 65, 6, 3, 31, -44, 10, -62, -1, 53, 3, -21, 11, -8, -1, 19, -60, 23, -35, -8, -20, 36, 108, 1, -33, -60, -16, -25, 20},
  {-2, -7, -68, -86, 9, 37, 19, 20, 10, -46, 10, 61, 57, 2, 34, 21, -73, -54, 0, -12, 22, 43, -15, 6, 8, -11, 13, -38, 63, -12, -3, -101, 31, 7, -14, -11, -12, 0, 26, 40, -86, 10, -39, -19, -59, -31, -10, -29, -24, 1, 58, -91, 19, -30, -46, -37, -18, -26, 26, -116, 19, -2, -42, -47},
  {36, -14, -29, -30, -38, -11, 7, 12, 9, 18, 15, -19, -49, 51, 10, -36, -78, -12, -16, -39, 29, -4, -2, 23, 13, 6, 26, 22, -88, 15, -33, -119, 12, -20, -51, -11, -85, -24, 21, 4, 30, 43, -7, -27, -93, -48, -13, -1, 12, 20, 66, -82, -102, 0, -7, -9, 69, -116, 38, -100, -12, 40, -1, -1},
  {18, -41, 34, -34, -92, -3, 26, 2, 4, 28, 12, -5, -35, -2, -14, 8, -69, -68, -16, -13, 37, -89, -4, 33, -8, 3, 24, 3, -23, -27, -15, -31, 21, 6, 62, -33, 76, -43, 0, -28, 39, 33, -65, 33, -51, -66, 67, 6, 14, -8, 36, 41, -65, 37, 16, 36, -74, -34, -90, 15, -69, 29, -51, -11},
  {6, 27, -23, -72, -16, -15, 10, -12, -31, 2, -4, 15, -10, -4, -22, -32, 29, 45, 19, -17, -6, -53, -8, 11, 37, 4, -18, 16, 67, -22, -27, 38, 36, -3, 35, -8, -104, -46, -4, -16, -71, -6, -87, 2, -17, -35, 29, 1, 14, -16, 14, 29, -23, 37, 9, 28, 13, 56, -61, -78, -35, -12, 0, -1},
  {-14, -11, -34, 40, -119, -10, 35, 20, 26, -18, -17, 2, 50, 34, -8, -3, -93, -69, -16, 49, -22, -3, -22, -31, 8, 45, -23, 3, -69, 41, -48, 14, -16, -17, -21, 48, 74, 5, 6, 20, 0, -21, -20, 43, 11, 55, -15, 11, 10, 1, -4, -94, -72, -22, -9, 4, -23, 20, 16, -61, 77, -10, -7, 55},
  {4, 30, 17, -4, -68, 18, -1, 6, 34, 34, -2, 7, -3, 35, 22, -43, 117, -58, -22, -37, 24, 14, 17, -51, 32, -8, -13, 35, 24, -29, -39, -114, 20, -23, 4, 14, -56, -1, -10, -14, -21, 54, 72, -75, -45, -46, -13, 70, 42, 15, -4, -28, -65, -12, 12, 19, 62, -58, -30, -47, 49, -25, -2, 0},
  {-5, 23, -85, -5, 17, 72, 24, -5, 17, -16, -16, 16, -35, 13, -15, 33, -42, -2, -11, 15, -2, -20, 21, -6, 40, -42, 10, -4, 5, -8, -42, -20, 34, 26, -28, -10, -45, -46, 9, 21, -13, 30, -117, -53, -26, -62, 0, 34, -10, -2, 36, -37, -7, -48, -44, 19, 77, -16, -28, 67, 41, -24, -23, 15},
  {-19, 1, -36, -55, -58, 54, -7, -9, 56, -9, -8, 28, 10, 3, 35, -6, 5, 30, 5, 16, 40, 25, -37, 13, 42, -6, 13, -30, 40, -26, -42, -74, 83, -12, -30, -25, -112, 0, 36, 65, -100, -26, -2, -46, -42, -40, -6, 0, 8, 40, 32, -67, -25, -50, -68, 13, -2, -101, 16, -79, -97, -24, -48, 26},
  {18, -3, 12, 38, 35, -32, 9, -3, 17, 48, 7, -48, -74, 21, -14, 3, 52, -13, -45, -1, -1, -5, 8, -22, 3, -1, 17, 54, -54, 24, -96, -49, 4, 24, -20, 29, -103, -86, 12, -6, 41, 33, 64, -29, -110, 5, -9, 21, 31, 30, -8, 24, -13, 21, 8, 56, -61, 42, -8, -81, -44, 1, 18, -15},
  {32, -24, 8, -44, -119, 15, 38, 23, -22, 32, 36, 9, 15, -36, -7, 18, -116, -49, -48, -59, 16, -29, -51, 19, -18, -6, 21, 19, -46, -5, -32, -77, 13, 23, 24, 7, 1, 2, 11, -13, -84, 32, 28, 18, -78, -13, 19, -18, -19, -15, 13, 51, -27, -25, -4, 9, -27, -113, -63, -107, -100, -16, -9, -7},
  {-11, 0, -77, -74, -41, 18, 19, -10, -15, 1, 16, 53, 4, -10, 35, -20, -108, -19, -13, -14, 6, -15, -13, -19, 33, 22, -19, 11, 59, -6, -15, -31, 8, 15, 27, -6, 84, -43, -19, 0, -71, -18, 26, -57, 11, -72, -2, 10, -10, -33, 57, -64, 10, -9, -16, 12, 56, 8, -30, -15, -119, -30, -15, -11},
  {-16, 1, 19, 39, 74, 7, -31, -22, 63, 13, -15, -19, 10, 16, -30, -5, -9, -2, 0, -12, -11, -1, 24, -3, 14, -16, -34, 7, 8, -9, -25, -102, 22, -3, -14, -3, -82, -11, -8, -14, -59, -13, 16, -22, -82, 12, 6, 49, 43, -4, 33, -110, -1, -23, -6, 20, 72, -75, -34, -80, -26, 17, 20, -19},
  {5, -5, 12, -86, -61, -5, 15, -12, 22, -26, -15, 3, 0, 34, -13, -7, -1, -11, 0, 15, -4, -7, -10, -23, -15, 16, -9, -26, 97, -10, -4, -106, 36, -4, 9, -22, -67, -12, -10, 5, -68, 53, -117, -31, -57, -54, 25, 37, 21, 10, 27, -73, 38, 52, -5, 6, -23, -49, -36, -42, -89, -11, -27, 3},
  {-30, -34, -4, -25, 21, 50, -16, -20, 31, -12, -22, 27, 9, -49, 3, -7, -62, -99, 34, 17, -10, -2, 62, 4, -5, -71, -69, -25, -6, -18, -4, -23, 38, -47, -5, -49, 23, 36, 12, 4, -18, 13, 54, 18, -23, -67, 29, 1, 41, 33, 1, -83, 51, -25, -29, 13, -40, -87, -53, 0, -113, -60, -44, -5},
  {-24, -21, 3, 22, -98, 16, 43, -18, -20, 6, -31, 27, 58, -57, -18, 53, -80, 10, 1, -22, -16, 6, 23, 53, 0, -42, 21, 37, 15, 27, -47, 65, 90, 4, -8, -3, -111, 20, 16, 19, -18, 5, -11, -19, -21, -51, 48, -38, 15, 33, -19, 73, -50, -11, -20, 23, -31, -66, -53, -47, 27, -21, -73, 44},
  {46, 24, -18, 12, 38, -6, 6, -5, -37, 21, -5, -32, 16, -13, -8, -9, -24, -13, -26, -20, -24, 18, 31, -9, -13, -70, 6, 46, 48, 48, -48, -109, 9, 6, 12, 23, 105, -11, 3, -34, -75, 49, 22, 7, -34, -24, 7, -33, -36, 30, -9, 58, -104, 25, 2, 19, -2, -47, 18, -56, -27, 27, 13, -39},
  {5, 10, -38, -56, 52, 47, 21, 6, -3, 13, 27, 35, 2, -33, -6, -48, 47, -84, -24, -34, 3, -10, -13, -5, 40, 8, -5, -1, 35, -2, 27, -94, 15, 15, -2, -9, -69, -30, 5, 20, -82, 18, -33, -13, -7, 2, -26, -20, 9, -5, 25, -73, -109, -18, -20, 12, -27, -87, -18, -43, -122, -3, -4, -7},
  {-9, -8, -2, -122, 45, 2, -1, -14, 13, -43, 29, 24, -21, 2, 35, 51, -16, -18, 0, 8, 34, 6, -3, -6, 6, -1, -8, 22, -102, -22, 21, -45, 21, 11, 39, 12, -44, -16, -6, -10, -23, 14, -2, -39, 3, -45, 17, 5, 8, -6, 37, 30, -69, -39, -2, -7, 34, 54, 5, -66, -126, -13, -19, -8},
  {-17, 4, 21, -29, -13, 30, -5, 8, 23, 4, -6, -4, -2, -16, -33, -21, -87, -27, -3, 21, -22, 0, 31, 9, 17, -19, -1, 18, 69, 2, 37, -15, 35, -19, -35, 32, 19, -14, 3, -17, -72, 3, -33, -8, -44, -21, -2, 48, 54, 51, 14, -50, 31, -12, -9, -2, -61, -39, 0, 58, -64, 10, 15, -25},
  {1, 9, -31, -30, -10, -1, 29, -23, -17, -7, -9, 11, -9, -12, -15, 7, -60, -40, -18, 29, -16, 17, 22, -14, 15, 12, -15, 13, -72, 32, -11, -108, 9, 4, -27, -5, 26, -27, 12, 7, 63, 27, -64, 1, -90, -50, 31, -30, -23, 16, 43, -70, -61, 67, 9, 10, -57, 61, -1, 49, 1, 16, -10, 18},
  {21, -49, 7, 27, -6, -52, 1, -16, 34, 12, -21, -6, 8, 23, 0, -19, 66, -45, 3, 41, -16, -47, -31, 50, 10, -25, -8, 16, -49, -26, -14, -10, 47, -28, -1, -45, 62, -22, 1, -12, 72, 65, -33, -6, -25, -57, 44, -4, 20, -5, -25, 18, -26, 44, 6, 19, -90, -102, -21, 19, -79, -6, -17, -16},
  {27, 10, 0, 8, -94, 21, 28, 3, 1, 27, -18, -39, 22, -18, 11, 49, -7, 29, 26, 11, -3, 24, 15, 42, -7, -31, 14, 51, 10, 20, -33, -28, 32, 18, -9, -1, 42, 3, 4, 13, -21, 44, -48, -28, 4, -5, 37, -21, 9, 0, -16, 24, -7, 6, -9, -41, 43, 48, -17, -67, 13, 15, -39, 24},
  {56, 11, -28, 11, -121, 39, 20, 17, -15, -8, 26, 10, -18, 4, 29, -15, -46, -32, -10, -15, 25, 29, -6, -26, 16, -15, 25, 36, -123, -9, 14, 16, -10, 10, 20, 13, -67, 1, 23, -2, -8, -11, -27, -17, 12, 1, -35, -17, -17, 0, -3, 29, -74, -8, -42, -2, -66, 38, 5, -72, -89, 9, -6, -30},
  {15, 25, -23, -97, -46, 36, 11, -1, 14, 28, 33, 29, -60, 23, 2, -32, -63, -94, -19, 31, 38, -19, 1, 0, 23, 12, -21, -31, -87, -60, 14, -108, -41, 16, -11, -7, -78, -55, 6, 13, 9, 7, -110, -3, -54, -23, -47, 2, -7, -8, 0, -43, 91, -17, -21, -14, -82, 76, -17, -66, 0, -10, 6, -69},
  {15, 18, 24, -62, -46, -13, 6, 4, -13, -24, 23, -30, -32, 22, -2, 15, 16, 7, -20, 49, 19, 1, 28, -5, -6, -11, -46, -1, -91, 13, 34, 52, -12, 22, 12, 38, -46, 14, -4, -12, 90, 9, -84, 35, -41, 8, 1, 6, -8, 4, -9, -5, -117, 2, 16, -15, 46, 82, 4, -32, -82, 6, 43, 30},
  {-26, 25, -7, -48, -45, -1, 22, 26, 3, 30, 45, 17, 33, 27, 27, -7, 10, -17, -32, 33, 18, 29, -30, -13, -23, 2, 9, 36, -68, -17, 44, 78, -76, -32, -8, 78, -4, 8, -25, -25, -118, 32, 8, 48, -38, 59, -52, -26, 12, 44, -46, -98, 103, -12, 1, -5, 37, -82, 58, 118, -17, -2, 33, -41},
  {-31, -13, -4, -54, 103, -1, 37, 12, 1, 0, 5, -10, 41, 16, 12, 35, 37, -55, 0, 15, -16, -12, -23, -19, -1, 31, -1, 25, -102, -16, 5, 32, -20, 5, 17, -9, -31, 64, 6, -14, 73, -4, 94, 5, 43, -2, -11, -6, -11, -3, -29, -21, -35, -22, -7, 21, 50, -67, 1, 14, -28, -7, -13, 21},
  {25, -8, 1, -23, 39, -16, -37, 22, -8, 37, 24, -13, -19, -7, 20, -8, -5, 17, -14, -2, -2, -6, -13, 5, 3, -13, 19, 33, -95, -40, -28, -56, 20, -1, 26, -28, 78, -20, -8, -30, 29, 4, 43, -22, -22, 32, 19, 12, -26, -28, -22, 10, 56, 7, -16, 3, 75, -78, -17, 32, 17, 18, -10, -23},
  {-15, -4, 12, -46, -17, 73, 38, 29, 19, 22, 46, -16, 12, 25, 73, 14, 33, 39, 15, 7, 24, 20, -47, -17, 17, 40, 16, 57, -33, -71, 4, -5, 6, 43, 56, 27, -121, -17, -11, 17, 80, -46, 32, -26, -32, 53, -36, 52, -13, -37, -16, -17, 13, -27, -21, -13, 44, 67, 0, -54, -68, 9, -23, -32},
  {45, 16, -13, -23, 33, 59, 38, -1, 4, -21, 55, 24, 0, -4, 24, -3, -98, -52, 13, 10, 34, -13, -29, 10, 20, 1, -13, 23, -78, -26, -4, 33, -6, 26, 52, -2, -42, 11, 6, -41, 38, 22, -70, -9, -30, 16, -31, 34, 1, -10, -19, -43, -19, -4, -3, 14, -22, -98, -3, -88, -52, 6, 1, -66},
  {13, -11, -14, -99, -18, 24, -1, -13, 51, -8, 36, 9, -42, 27, 30, -27, -82, -4, 16, 33, 30, 19, 25, 17, -34, 0, -54, -19, -96, -28, -17, 58, 3, -17, 16, -5, 78, -3, -5, -20, -86, 42, 2, 7, -38, 13, -20, 31, -12, 27, -13, 28, 13, -25, 10, 20, 76, -29, 8, 44, -42, -17, 37, -61},
  {-18, 5, 2, 28, -48, 43, 40, 15, 22, 2, 50, 12, 9, 29, 42, 3, -18, -30, -19, -4, 14, -4, -37, 7, 13, -24, 8, 8, 7, -31, -16, -9, 6, 25, 30, 38, 81, 13, -25, -36, -86, 48, -14, -19, -39, -10, -44, 22, -18, 5, -18, -10, -18, 0, 43, -22, -93, -59, 13, 86, -10, 9, 43, 2},
  {-2, -7, 10, -43, 58, -6, 28, 21, 2, 6, 15, 13, 24, -3, 4, -24, 36, -10, -26, 10, -2, 14, -8, -15, -14, 16, 5, 16, -62, 24, 26, 80, -33, -9, -9, 30, -29, -3, 1, 14, -49, 6, 56, 51, 53, 19, -11, -23, -2, 25, -10, -58, 49, -1, 1, -20, -37, -60, 5, -29, -13, -18, 5, 16},
  {11, 15, 22, -45, -69, -1, 4, 20, -2, 16, 21, 11, -15, 22, -2, -14, -67, -23, 7, -15, 1, -8, 3, 19, -12, 9, -11, 31, 6, 10, 9, 53, -15, -10, -8, 21, 77, -22, -9, 3, -51, -9, -29, -1, -7, 10, -11, 1, -25, 14, 5, -75, -28, 2, 12, 10, -29, -95, 12, 28, 6, 11, 2, 2},
  {7, 51, 36, -42, -121, 46, 6, 2, -6, -2, -20, -18, -18, -7, -31, 2, 80, -28, -3, 14, -26, -23, 33, 20, 20, 23, 8, 25, 37, 9, -10, 4, 25, 30, -8, 27, 70, -8, -7, 14, 51, -68, 34, 5, 87, -3, -11, 22, 40, -17, 16, -25, -103, -10, 10, -21, -64, 43, -30, -89, -29, 0, 28, 23},
  {-11, 47, 26, -7, -122, 23, -10, 4, -6, 4, -20, -13, 8, -29, -12, 20, -1, -17, -30, 18, -22, 51, 23, -32, 6, -2, 10, -30, -82, 38, 4, 74, 5, 40, -29, 30, 73, -2, 0, 23, -85, -30, 25, 14, 34, 37, -36, 11, 17, 13, 40, -82, 14, -15, -8, -33, 76, -11, 35, -52, -76, -10, 3, -6},
  {20, 35, 29, -7, 15, 6, -44, -20, 15, -7, 10, -11, -5, 7, -12, -60, -50, -33, -9, -17, 11, 29, 0, -60, 13, 15, 20, -7, -94, -29, 1, 21, -20, 15, -14, 15, -67, 6, 21, 28, -79, -63, -104, 8, 4, 55, -37, 18, 17, -5, 2, -37, -109, 4, -17, 8, -106, 5, -18, -25, 24, -25, 34, -14},
  {15, 18, -26, -22, -47, -30, -27, -17, 1, -9, 27, -29, -26, 24, -2, -32, 15, 25, -7, 11, 14, 21, 3, -7, -3, 27, 7, 15, -47, 14, 81, 8, -56, -50, -21, 8, 67, -9, 11, 0, 36, -1, -93, 31, -71, 30, -23, -21, -11, -3, -40, -73, 61, 29, -4, 13, 1, -47, 26, 68, -83, 16, 35, -15},
  {16, 43, 48, -58, -86, -5, -11, 27, -18, -8, 11, -60, -61, -31, -18, 23, -22, 7, -1, -44, 26, 16, 22, 12, 22, 12, -20, 18, 83, -31, 40, -121, -29, 16, 17, -12, 36, -1, 3, -9, 15, -15, -21, 18, -15, 13, -6, -2, 27, 5, -15, -75, -29, -21, -11, 2, 13, -29, 6, -20, -115, -18, 34, 3},
  {-20, 24, 12, 2, 82, 19, -13, 14, 2, -19, 13, -22, -70, -36, -14, -17, -106, 22, 12, 18, 25, 30, 43, 8, -17, 12, -39, 17, -81, 23, -18, 38, -59, 28, -6, -3, 86, 49, 13, -4, 5, -60, -118, 61, 27, 62, -21, 10, -3, 23, -38, 64, -22, -20, -18, -3, -58, -105, -12, 40, 33, -50, 5, 49},
  {18, 15, 2, -33, -12, 3, 24, -14, 2, -19, -12, 1, 19, -8, -1, -10, -111, -18, 2, -12, -16, 0, 31, 14, 16, 9, -19, 14, 65, 15, 24, -68, -3, -6, -4, 12, -35, 15, 4, 11, -47, 7, -10, -4, 24, -18, 15, -19, 0, 21, 33, -103, 63, -9, 10, 9, -67, -17, -7, 62, 28, 3, -18, 27},
  {3, 3, -8, -28, -82, 9, 33, -17, 21, -32, -4, 28, 21, -6, 13, -40, 82, -13, 27, 42, 4, 20, -3, 22, 7, 14, -5, -27, -63, -19, -9, -16, -5, -8, 5, 12, 22, -17, -16, 7, -58, -11, 45, 0, 40, 10, -11, 14, 10, -8, 22, -40, -124, -4, 6, -37, 22, -59, -4, 3, -39, -4, -8, 16},
  {7, -10, 13, -75, 34, 37, 26, -7, 12, -22, -7, 30, 40, -17, -3, 20, 12, -88, 18, 22, -27, 28, 20, 16, -5, -13, -27, -51, 67, 1, -38, 54, 27, 30, 8, 4, -65, 21, 3, -26, 42, 26, 19, 3, -15, -11, 19, 5, 4, -11, 34, 3, 55, 18, 5, -52, 26, 43, -7, 51, -44, 17, -20, -17},
  {8, 21, 4, -36, 27, -19, 44, 26, -12, -18, -8, 19, 21, -20, 8, 32, -5, -10, -12, 19, -23, 29, 20, 10, 12, 2, -10, 15, 13, 16, -4, 62, 30, 1, -23, 27, 7, 52, 5, -20, 5, 49, 3, -1, -54, -6, -6, -30, 12, 7, -16, 9, -34, 6, 4, -13, -110, -50, -18, -70, 20, 17, 30, 5},
  {22, 3, -16, -59, 6, -20, -18, 45, -2, 20, 9, 7, -63, -18, 13, 1, 33, 20, -37, 3, 36, -5, -1, 22, 9, 31, 1, 28, 7, -17, 33, -14, -33, -22, -22, -20, -51, -12, 36, 29, -27, 11, 52, 36, -98, 7, -21, -33, -7, 26, -16, 69, -42, 23, -35, 18, -101, 61, -17, -33, -11, -2, -14, 27},
  {19, 33, 13, -22, -18, 19, -26, 34, -15, 11, 18, -41, -67, -5, -2, -9, 12, 40, -32, 11, 50, 34, 3, 2, -5, -15, -22, 19, -96, -20, -2, -22, -33, -1, -8, -33, 50, -42, 31, 30, -37, 12, -89, 44, -55, 27, -21, -23, -13, 41, -17, -75, -20, 11, -34, -1, -114, -81, -14, -116, -28, -15, -1, 34},
  {-9, -23, -56, 42, 25, -20, -30, 57, -21, -20, 23, 47, -46, -4, -10, 26, -2, -9, -43, 30, 28, 7, -13, -44, -27, 55, 6, -23, 48, 54, 20, 30, -76, -51, -38, 21, -81, 24, 35, 41, -70, -15, 84, 45, 27, 41, -11, -59, -3, 22, -26, 13, 58, -55, -34, -47, -7, -36, 19, -68, -12, -47, -47, 119},
  {39, 22, 10, -61, 26, 33, 10, 0, 17, -20, 7, 3, -6, 2, 4, -50, 50, -67, 3, -13, 6, -5, -7, 2, 16, -3, -41, -4, 16, 8, -39, 16, 8, 5, 3, -3, -31, 23, -11, -17, 88, 22, 51, -29, -20, -10, 5, 26, -12, 15, 21, 48, -59, -39, 10, -32, -74, -85, -28, -56, -15, -32, 13, 42},
  {-9, 26, -11, -35, 53, 44, -1, 1, 19, -21, -6, -12, 12, -9, -1, 12, -117, 15, 6, 26, 0, 20, 6, -6, 20, -19, -7, -6, 62, 24, -41, -101, 2, 49, 13, 28, -6, 16, -4, -3, -19, 12, -60, -32, 4, 15, -27, 39, -4, -4, 7, -80, 33, -25, -16, -13, -97, -44, -17, -128, -43, -9, 10, -4},
  {12, 54, -28, -79, -72, 26, 40, 37, -28, -25, 2, 14, 9, -15, 16, -12, 39, -24, -22, 3, -5, 12, -8, -21, 35, -19, 5, 15, -33, -14, -51, -105, 26, 52, -5, 8, -23, 35, 9, 5, -108, -14, -62, -23, -83, 1, -23, -1, -7, -2, -9, -69, 39, -26, -39, -28, -82, -34, -10, -108, -111, -11, 13, -19},
  {4, 35, -43, -71, 84, 4, 28, 5, -13, -15, -7, -29, -3, -23, 7, 11, 22, -7, -6, -8, -8, 1, 13, -8, 44, 5, 24, 16, -62, 2, -20, 13, 31, -26, -14, -9, -4, 14, 18, 13, -6, 3, -46, -32, -94, -39, -10, -18, 8, -1, 33, -11, 56, 5, -35, 15, -67, 58, -23, -7, -127, 10, 7, 8},
  {2, 26, 16, -48, 61, 37, -10, 36, 3, 17, 23, 0, -83, 26, 4, 8, -101, 13, -44, -61, 39, -7, 23, -35, 36, -7, 45, 32, 13, -28, -13, -41, -40, -13, -23, -34, -59, -41, 27, 42, -96, 36, -81, -11, -18, -48, -49, -17, -15, 11, 25, -40, -113, 18, -29, 11, 31, -18, -30, 25, -117, 20, 0, -2},
  {17, 11, -69, -1, -82, 37, -56, -21, -7, 5, 22, 8, -76, 25, -21, -35, 2, 18, 4, 6, 37, 10, 27, 8, 18, -30, -7, -12, -33, -8, 62, -77, -36, -31, -9, -56, 36, -64, 24, 8, -42, 45, -31, 28, -61, -32, 14, -50, -12, 31, 32, -74, -109, 0, -14, -32, -25, 52, -20, -71, -123, -14, -22, 20},
  {-24, 6, -46, 29, -65, -39, 49, 47, -17, -21, 2, -17, 66, 31, -39, 7, -76, -39, 20, 38, -29, -33, -10, -16, -4, 35, -11, 2, -18, 74, 13, 65, -41, -52, -23, 77, -18, 17, -4, 4, -20, -6, -43, 57, -58, 0, 1, -30, 2, -3, 2, -24, 13, -37, 15, -30, -98, -25, 22, 12, -59, 22, -22, 53},
  {29, 12, -31, -22, -59, 25, -54, 1, -20, -10, 9, 51, -14, -9, 0, -27, -81, 26, -11, -63, 23, -5, 14, -28, 10, -46, -19, -28, 91, 6, -1, -83, -8, -20, 24, -22, -114, 19, 7, -14, 41, 21, -119, 0, -15, -40, 22, -8, -2, 11, 23, -2, 69, -39, -10, 9, -48, 73, -67, 17, -7, -65, -47, 32},
  {-15, 10, -26, -84, -47, 27, 7, 22, -14, -30, -15, 17, 28, -43, -2, -34, 60, -43, 14, 21, -1, -26, 36, -53, 16, -54, -25, -6, -37, -31, -16, -43, 37, 7, 27, -33, -115, 30, 5, 0, 28, -39, 65, 9, -53, -30, 17, 25, 1, 1, 2, -27, -97, -26, -41, 2, -49, -37, -43, -125, 75, -35, -20, -12},
  {-23, 29, -13, 9, -116, -7, 7, -25, -10, -11, -8, -24, 12, -30, -3, -29, 56, -40, 23, 26, 4, 3, -5, -16, 34, -15, 37, 5, -27, 9, -10, -98, 61, -40, -13, -40, 66, 16, 23, 32, -64, -30, -103, -3, -52, -36, 18, -10, -1, 19, 23, 20, -96, -5, -24, 24, 22, 27, 8, -109, -101, -3, -8, 46},
  {-48, 83, 7, -24, 58, 1, -51, -38, 1, 20, -10, -127, -29, 23, -1, 7, 74, -57, 37, 22, -7, -17, 46, -39, 53, 22, 46, 33, -107, 23, -11, -19, -28, 51, 17, -43, -120, -9, -1, 15, -96, -90, 8, -36, -52, 39, -57, 44, -4, -6, 42, -36, -80, 6, 9, 32, 47, -20, -4, -70, -66, 9, 46, -2},
  {33, 54, 22, -74, -68, 52, 10, 3, -49, 39, 30, -31, -2, -22, -11, -62, -61, -35, -8, -23, 0, -68, -18, 27, 2, -40, 19, 24, 25, -17, -104, -43, 31, 7, 58, -19, 30, -9, -14, -73, -52, 37, -10, -16, 47, -5, 36, 1, 10, -10, 35, 7, 79, 36, 27, -13, 77, -76, -82, -61, 5, 28, 14, -24},
  {10, -27, -26, -46, -127, 24, -7, -18, -6, 24, 15, 9, 10, -29, -5, 25, 13, -49, 0, 3, -3, -33, 8, 44, -28, -10, -11, -8, -104, 1, -44, 7, 41, -47, 38, 18, -96, -18, -5, -43, -125, -1, 23, -21, -19, 3, 57, -4, 14, 6, 20, -71, 67, -6, -14, 28, -9, -31, -40, 25, 39, -32, -25, 2},
  {-4, 15, 4, -45, 15, -60, -35, -36, -22, -15, 18, -14, 18, 65, -31, 11, -88, 24, 31, -13, -3, -39, 6, 12, -1, 4, 9, 12, -11, 20, 20, -92, -58, -46, -5, -88, -62, 20, -15, 2, -115, -37, -101, 19, -71, -35, 10, -15, -19, -20, 20, -39, 51, -10, 4, -8, 56, 67, 46, -36, 14, 30, -24, 11},
  {12, -31, -44, -70, 42, 23, 4, 6, 15, -26, -8, 44, -1, -14, -15, -9, 11, -13, 5, 29, 0, 12, 16, -32, -2, -29, -21, -7, 5, 9, 31, 14, 7, -9, -36, -25, -96, 16, 23, 34, -71, 14, 0, 53, 6, -52, 24, -21, 2, 17, 7, 28, 10, 7, -32, 5, -93, -63, -26, -60, 31, -30, -40, 14},
  {9, -34, 2, -90, 14, 15, -4, -3, 8, 6, -13, 27, 81, -63, 4, -16, -38, -39, 29, 5, -8, 23, 44, 28, -18, -75, -38, 9, 41, -9, -1, 62, 38, -54, 1, -55, 54, 61, 4, 1, -32, 47, 7, 50, 10, -66, 64, -30, 20, 31, -22, -82, -65, -15, 3, -14, 45, -13, -27, -85, -47, -30, -33, -12},
  {16, -1, 17, -1, -16, -35, -24, -52, -12, 27, -47, -49, 74, -59, -12, 64, 18, -21, 27, 27, -36, 40, 60, 90, -39, -89, 19, 23, 28, 66, -13, 72, 35, -18, -18, -47, -107, 73, 5, -8, 66, 42, -71, 44, -38, 0, 93, -40, 31, 33, 1, 11, 83, 3, 7, -24, 49, -101, -29, -35, -104, -19, -36, 48},
  {17, -1, -9, -53, -46, 16, 8, 1, -23, -14, -21, -34, 55, -11, -13, 24, 22, -46, 16, 11, -45, 7, 15, 22, -32, -34, -5, 26, -123, 39, 39, -71, 15, 0, -27, -19, 50, 24, 0, 2, 75, 24, -48, 46, -46, -17, 14, -42, -31, 18, 29, -33, 58, -4, -11, 35, -86, 30, -8, -110, -43, 3, 1, 18},
  {-4, -42, -41, -111, -8, 28, 59, -11, 1, -2, 1, 25, 18, 11, 1, -20, -114, -34, 3, 27, -2, -44, -45, 2, -2, 60, -18, 20, 60, -22, -3, -17, 14, -14, 1, -8, 13, -25, -8, 15, 8, -10, 39, -72, -5, -60, 4, -2, -18, -15, 24, 13, 42, 9, -28, 20, -101, -66, -3, -54, -71, -39, -18, 32},
  {-6, -45, -40, -29, -22, -47, 26, 4, 2, -33, 17, 21, -18, 60, 22, 8, -68, -2, 2, 16, 27, -56, -24, -43, 15, 45, 13, 44, 42, -13, -47, 47, -9, 15, -1, -23, 41, -35, 1, 24, 3, -31, -65, -11, -24, -37, 14, -19, -23, -60, 61, -98, -25, 9, 4, 54, -93, 4, 23, 99, 28, 21, -44, 43},
  {-5, 5, -5, 41, 86, -29, 39, -38, -20, 18, 2, 1, 7, 10, -14, -5, -114, 8, 12, 9, -4, -19, 17, 29, -6, -6, -4, 18, -47, 44, -47, -30, 29, -51, -31, 10, -117, -39, 9, -19, 91, 45, -62, -14, -41, -54, 10, -25, 5, 13, 30, 45, 15, 72, 15, 14, 83, -55, 46, -21, -83, 59, 8, -16},
  {-1, 4, 0, -47, -94, -30, -19, 16, -5, 7, -8, -8, 34, -23, -18, -64, 50, -22, 19, 13, -13, 3, 12, 10, -9, -6, -23, -15, 16, 6, 17, -7, -32, 6, 17, -2, -52, 80, -1, 2, -62, 11, 36, 37, 27, 23, 22, -14, 9, 5, -29, -6, 60, -20, 16, -6, -56, 46, 0, 46, -95, -14, 4, 28},
  {14, -24, -8, 81, 36, -55, 38, 16, -19, 19, 27, -2, 21, -29, 14, 67, -4, -12, -8, 19, 13, -15, -34, 32, -16, -16, 22, 19, 20, -16, 20, 25, -1, -17, 46, 17, -43, 14, -11, -36, 96, 14, 104, 14, -41, -2, 35, -8, -23, -35, -30, 52, -59, 13, 14, -17, 46, -61, -8, 74, 25, -2, -33, 3},
  {28, -21, -18, -50, -78, 1, 32, 5, -6, -11, 1, -45, 18, -16, 16, 28, 69, -22, 2, 58, -10, 17, -8, 34, -38, -5, 30, 40, -60, -2, 5, -5, 6, -8, 22, 13, -104, 7, -4, -1, 88, 13, -110, 18, 16, 9, 36, -16, -29, 6, 19, -77, -13, 14, -10, -16, 93, 14, 1, 25, 22, 4, -5, 14},
  {20, -26, -3, -110, 41, 20, 46, -24, -5, -13, -7, 32, -11, 17, 1, -20, -23, -44, 0, 44, 3, -26, -8, 18, 9, 20, -1, 17, -28, -8, 25, -21, 2, -36, -3, -4, 63, -51, 6, 30, 89, 17, -59, 21, -46, -17, 5, -23, 29, 13, 26, -54, 28, 40, -17, 47, -45, 8, 13, -25, -5, 13, 8, 1},
  {-40, -7, 0, -24, -78, -7, 26, -5, 53, -12, 10, 17, -23, 58, 17, -21, -54, -55, -15, 59, 11, -26, -22, -2, -12, 37, -45, -7, 24, -39, -3, -39, -1, -47, -1, 0, 31, -41, -16, 16, 95, 18, -45, -27, -42, -12, -21, 37, 8, -5, 18, 5, -84, -22, -5, 1, -27, -30, 9, 61, 52, -23, 37, -22},
  {-1, -5, 35, 0, -34, -2, 44, 11, 3, 24, 39, -32, -18, 42, 44, -17, -75, 16, -39, 22, 33, -19, -21, -30, 8, 53, -17, 69, 2, -12, -1, -116, -11, 35, 21, 49, -99, -28, 3, 1, -44, -7, -83, -11, -11, 24, -29, 40, -3, -22, -15, 34, 100, -37, 24, -8, 30, -25, 37, 104, -40, 15, 22, 7},
  {34, 3, 15, 9, -83, -18, 7, 1, -15, 6, -1, -3, -20, -4, -22, -13, -92, 7, -20, 8, -13, -30, -8, -11, 1, 16, 6, 7, 65, 33, -36, -106, 0, 2, -1, 12, -54, -16, 11, 6, 35, -7, -88, 18, -25, -13, 17, -2, -9, 3, -5, 18, -25, 18, 22, 8, -50, -67, 1, 57, 56, 28, 2, 19},
  {-5, -37, -49, -12, -74, -3, -20, 46, 17, 47, 21, -18, 24, -5, 24, 50, 81, -22, -37, -3, 10, 23, -16, -8, -12, -19, 25, 27, 72, -51, -54, -31, 22, -2, -1, 24, -5, 0, 11, -18, 112, 12, -13, -26, -28, 21, 3, 5, -28, 3, -25, -77, -10, -21, -29, 0, -33, -44, -12, 2, 6, -6, -38, -22},
  {-2, -24, -20, 49, 33, 24, 0, 5, -7, 21, 26, 16, -26, 30, 34, -32, -78, 80, -10, -5, 8, -23, -40, -9, -2, 21, 69, 48, -116, -58, 23, -96, 26, -10, 7, 5, -14, -69, -12, -5, -92, -10, -88, -56, -66, -8, 7, -14, -42, -33, 12, -69, 46, 25, -13, 15, -11, 63, -17, -4, -118, 41, -31, -3},
  {21, 9, 23, 26, 65, 41, 9, -10, -25, 28, 60, -7, -31, 26, 19, -47, 22, -25, 23, -1, 31, -47, -38, -11, 25, 26, 26, 61, 102, -37, -21, 84, 5, 39, 66, -5, 54, -41, -6, -60, -42, -39, -45, -49, -122, -1, -34, 35, -6, -16, 54, 39, -99, -3, 22, 43, -71, -72, -19, -59, -18, 27, 15, -81},
  {36, 13, -3, -35, -83, 17, -15, -17, 7, -9, 114, 55, -41, 22, 58, -58, -16, -13, 0, 27, 60, 15, -20, -4, -65, -8, 34, -3, -37, -17, 13, -120, -28, -6, 54, -9, -5, -31, -10, -70, 90, 31, 37, 18, -96, 61, -36, 0, -52, -2, 9, 17, -78, -40, -5, 14, 7, 99, 54, -18, -14, -18, 32, -104},
  {4, 53, 25, 2, 36, -24, -71, -16, 63, 42, 87, -20, -125, 87, 23, -22, -43, -39, -28, 14, 74, 15, -16, -66, -18, -9, -7, -16, 50, -47, 61, -55, -63, 22, -7, -26, -73, -92, 3, -43, 56, -47, 44, 6, -42, 50, -72, 35, -34, 26, 5, -11, -108, -56, -2, 4, 45, 32, 59, -27, -77, -16, 47, -27},
  {-51, -11, -58, 5, -25, -24, -7, -25, 54, 33, 83, -13, -80, 81, 21, -16, 20, -31, -20, 15, 69, 66, -48, -44, -8, 44, 36, 24, -56, -31, 87, -13, -53, -19, -22, 9, -98, -59, 5, -23, -43, -41, 21, 23, 40, 62, -85, 1, 1, 32, -29, -43, 77, -52, 1, 26, 14, 80, 52, -82, 71, -31, 35, -13},
  {2, 15, -15, -44, 49, 11, 22, 0, -2, 5, 17, 4, 17, -3, 0, 4, -14, 8, -7, -22, -8, -10, -2, 3, -5, -5, 18, 2, -41, 0, -24, 89, 11, 14, 4, 18, 19, -18, -4, -5, -43, 25, 72, -10, -16, -13, -2, 5, -18, -11, -11, -73, 10, 3, 9, 2, -92, 25, -6, -33, 20, 4, 12, -13},
  {-7, -16, 11, 51, -47, -26, 8, 21, 1, -24, 10, 11, -6, 12, -7, -9, -31, -27, 2, 24, -11, -19, 0, 14, 2, 24, -25, -19, 117, -4, 30, -93, -4, -30, 11, 13, 3, -27, -5, 11, 16, 25, 111, -1, -103, -15, -8, 7, -14, -6, 16, -76, 88, -20, 5, 9, -44, -40, 5, -17, -14, 12, -11, 18},
  {-21, 0, 8, 41, -41, -24, -27, -18, 35, -29, -43, -37, 10, -28, -12, 56, -63, 12, 2, 55, -25, 17, 12, 30, -12, 4, 0, 19, 70, 4, -2, 52, 34, -1, -14, 11, -28, -10, -5, 15, 30, -13, -59, 9, 80, 43, 12, 26, -2, 10, -16, -52, -66, 1, -5, 11, -62, -36, -33, 45, 75, 5, -7, 42},
  {-37, 11, 2, -35, -51, 23, 8, -39, 13, -12, -42, -35, 19, -6, -14, 26, -61, 7, 33, 38, -33, 9, 26, 0, 31, 20, 5, -7, 52, 19, -37, -14, 24, -10, -38, -7, -55, 24, -22, 19, -3, -28, 80, -7, 39, 25, -1, 35, 7, -21, 35, -89, 96, 32, 6, 21, -25, -79, -19, 54, -14, 17, 14, 6},
  {-11, 33, 13, -11, -48, 10, 4, -14, 7, 1, -24, -24, 13, -16, 7, -25, -14, -33, 30, 20, -25, 22, 25, -2, 32, 19, -29, 11, -64, 3, 22, -48, 5, -3, 1, 16, -99, 37, -17, 18, -88, -27, -41, -3, -18, 51, -13, 32, 22, -11, 25, -80, -94, 26, 10, 15, -48, -16, -42, 20, -34, 11, 41, 20},
  {-24, 8, 6, -49, 60, -22, 25, -6, 13, 1, 5, -35, -5, -19, 15, -53, -39, -19, -8, -9, 7, -9, -1, -8, -3, 28, -3, 7, 83, -8, 10, 69, 3, 2, 15, 48, -90, -15, -2, 9, 14, 14, -47, 4, 23, 14, -17, 15, 9, 0, -8, 15, 43, 21, 10, 24, 16, 38, -1, -12, -51, -1, 39, -13},
  {10, 16, -6, 4, 58, -44, 10, 2, -5, 3, -2, -26, -5, -8, 2, -1, -105, -48, -9, 0, -8, 16, 17, 27, 22, 11, -3, 20, -50, 7, -5, 48, -4, -3, -10, 10, 39, -2, 18, 9, -55, -6, -72, -13, 9, 8, 2, 2, 21, 18, -13, -13, -49, 37, -3, 17, -102, 23, 9, 33, -65, -11, 24, 14},
  {-2, 10, 5, 48, 53, -10, -6, -14, -4, -3, -12, -8, -12, -16, 12, -2, -71, -32, 27, 2, -1, 8, 18, 19, 12, -13, -2, 2, 35, 0, 19, -55, -2, 6, -4, -10, -81, 30, 13, 22, -68, -16, 63, -1, 24, 15, -3, 9, -3, 9, 9, -30, -46, 1, -19, 13, -38, 38, -11, -66, 32, -15, -10, 33},
  {36, 15, 40, -14, -44, -27, 21, -14, -37, -40, -18, 40, -15, -36, -28, -34, -10, -47, 24, 1, -10, -36, 11, 20, -3, -2, -31, 31, 25, 32, 11, -30, -21, 2, 20, -8, 31, -21, -16, 10, 89, -50, 83, -29, -48, -20, 27, 3, 1, 1, 25, -89, -47, -24, 27, -26, 39, -36, -21, -41, 68, -9, 15, 43},
  {-8, -16, -9, 21, -52, -4, 43, 2, 10, -32, 1, 16, 12, -7, 16, -23, -71, -51, 33, 22, -3, -2, -17, 20, 1, 2, -20, -1, -63, 8, -11, -51, -21, -22, 29, 6, -118, -17, -26, -6, 81, 2, 78, 15, -11, -3, 5, 17, 3, -21, 33, -61, -14, -2, 26, -21, -113, 1, -18, -54, 13, 11, -16, 1},
  {-2, -12, 3, -102, -64, 28, 24, -14, 26, 6, -3, 16, 33, -12, 7, -18, -83, -62, 18, 30, 9, 18, 6, 42, 4, -25, -17, -13, 58, -17, -16, -74, 27, 16, 0, 2, -106, -10, -2, -16, -17, 25, 62, -21, -63, -6, 22, 25, 10, 5, 14, -47, -81, 26, -17, -25, 73, -43, -24, -95, -115, 41, -4, -23},
  {-3, -14, 4, -6, -101, 14, 64, -11, 26, -4, 7, 15, 59, -5, 24, 22, -9, -23, 7, 21, -26, 4, 3, 1, 19, 10, -15, 23, -77, -18, -5, -90, 8, -3, 5, 16, -68, 17, -29, -32, -82, 42, -37, -32, -52, -2, -7, 18, -10, -14, -28, -27, -47, 9, 2, 7, -71, -45, -13, -83, -90, 15, 30, -25},
  {1, -10, 15, -72, -40, 0, 26, 0, 22, -6, 20, -7, -10, -13, 14, -13, -113, -8, 0, 21, 7, -15, -10, -13, 14, 29, 3, 35, -84, -29, -11, 52, 1, -2, 19, 9, -89, -8, -7, -3, 68, 10, 58, -43, -58, -4, -3, 9, 4, -24, -30, -80, -2, 16, 13, 49, -5, 0, -15, -22, -45, 5, 24, -12},
  {11, 1, 20, -82, 3, 9, -5, 0, 16, -4, 17, -19, -34, -11, 19, -42, -32, 19, 6, -2, 25, -22, -6, 35, 3, 4, -19, 32, 5, -17, -17, 72, -4, -3, 19, -34, -117, -9, -9, -7, -54, 22, -63, -3, -33, -3, 4, -4, -10, 1, -49, -82, -41, 24, -1, 52, -86, -99, -36, -21, -54, -1, -14, -8},
  {8, 5, -27, 94, 67, -37, 11, 21, -2, -21, 7, -23, -51, -18, -2, -19, 47, 39, -9, 27, 2, -8, -7, 8, -7, 16, -35, 8, 1, 15, -54, -1, -9, 9, 20, -26, 47, 10, 1, 19, -7, -29, -22, 11, 22, 6, 10, 9, -13, 13, -48, -56, -112, 21, -21, 12, 7, 38, -16, 25, 12, -23, -26, 65},
  {-1, 12, 25, 26, -90, -19, 10, 2, 21, -18, 12, 30, 13, 17, 10, -72, -36, -3, 19, -9, 24, -20, -10, -4, 18, -12, -12, -19, -71, -8, -63, 37, -12, -19, 6, 0, 42, 64, -29, -29, 77, 36, 66, -14, -47, 19, 11, 22, -7, 4, 40, 3, 39, -53, 13, -16, 50, 14, -48, -32, 59, -31, -4, 14},
  {5, 1, 0, -17, -44, 25, -27, -3, 38, -25, -4, 4, -16, -12, 2, -34, -114, 22, 0, -5, 14, 1, 2, 2, 15, -37, -24, -3, 67, 6, -56, -96, 2, 23, -10, 7, -81, 19, 9, 3, -87, 35, -22, -39, -79, 21, 3, 42, 1, 12, -11, 4, -115, -25, -20, -11, -120, 36, -47, -16, -30, -2, 5, 16},
  {3, 19, -16, -82, 60, 25, -13, 13, 25, -39, -7, 3, -15, -4, 14, -47, -3, -45, 8, 11, 13, -7, -13, 33, 13, -24, 9, -9, -88, -14, -58, -70, 16, 10, 18, -18, -72, 21, 10, 7, -8, 10, -47, -27, -43, 8, 8, 20, 1, 10, -17, -32, 34, -30, -25, -15, -24, 52, -5, 11, -51, 12, 0, 53},
  {-4, 33, -14, -18, -109, 6, -24, 18, 32, -6, 5, -26, -37, 24, 15, -42, 69, -53, 1, 11, 7, 29, -26, 25, -4, 11, -5, 23, 68, -11, -50, -13, -6, 4, 11, -2, -96, -12, -2, -20, -80, -14, -28, -28, -42, 49, -24, 25, 24, 15, -19, -4, 60, -13, 13, 34, -8, 31, 31, -110, -26, 12, 56, 6},
  {-1, -12, -3, -34, -64, 19, -27, 2, 29, -9, 12, 18, -16, 23, 32, -24, -45, -7, -12, -22, 23, 43, -10, 42, -45, -17, 13, 12, -21, 0, -30, 62, -6, -15, -15, -4, 26, -15, -12, -16, -61, 20, 8, -27, -41, 23, -2, -12, -9, 12, 5, -84, 22, 15, 15, 32, -13, 23, -11, 7, 21, -1, 10, 3},
  {-48, -34, -17, -123, 59, 18, 5, 13, 46, 6, -20, 44, 34, 4, 24, -4, 28, 13, -18, 4, 14, 31, -12, 23, -36, 1, -16, 0, -16, -3, -41, 9, -2, -11, -21, 4, -17, -16, 5, 15, -119, 55, -61, -23, -71, 7, 1, -4, -12, 18, 12, -50, 5, 8, 0, -12, -10, -90, -7, -82, -93, -5, -23, -2},
  {-12, -53, -43, -9, 23, 26, 11, 19, 20, 35, -8, 13, -33, 0, 21, 34, 78, 3, -54, -46, 1, -7, -32, 8, -18, -10, 9, 1, 1, -6, -58, -76, 36, -42, -11, 35, -103, -21, 21, -2, -83, 32, -33, -26, -64, 14, 16, 6, 11, 5, 12, -3, -89, -39, -9, 6, 79, 16, -27, 8, -67, -8, -31, -3},
  {28, 29, 4, -7, -122, -10, -26, -3, -21, -31, 15, -9, -31, -9, -3, -16, -80, -28, 20, -63, 14, -21, -12, -26, 2, -27, -20, -16, -46, 27, 0, -74, -19, -21, 46, -27, -105, -7, -7, -23, -22, -4, 36, 1, -79, 3, 2, 25, 10, -45, 13, -26, 105, 16, 0, 27, 41, 56, -54, -105, 43, -28, 3, 14},
  {15, -27, -54, -85, 56, -33, -48, -11, -33, -31, -19, 28, 25, -14, -4, -108, -26, 22, 28, -27, 19, -26, 33, -19, -19, -44, 1, -10, -32, -14, 48, 35, 29, -59, 3, -47, 41, -17, 12, 1, -117, -24, -116, 51, -43, -59, 63, -31, -36, 18, 15, -112, -16, 34, -14, 20, 13, 45, 15, -64, 14, 17, -52, -30},
  {0, -34, -32, -40, 66, 7, -72, -14, -41, -36, -28, 17, 46, -19, -16, -86, 32, 27, 33, -18, 3, 14, 53, -15, -26, -42, 45, 13, -8, 10, 72, -22, 7, -57, -5, -55, 50, -11, 24, 36, -106, -30, 33, 52, -16, -27, 48, -65, -41, 22, 8, -66, -68, 60, -21, 8, -19, -55, 39, 1, -93, 9, -40, 12},
  {2, 60, 33, -69, -92, -37, -38, 0, -24, 14, -12, -23, 21, -10, 25, -37, -20, -13, 47, -19, -30, 38, 68, -61, -15, -15, 39, 26, -114, 2, 74, 26, -45, 42, 44, 21, 24, 66, -36, -10, -6, -66, 30, 31, 5, 71, -16, 23, 4, -13, -28, 24, 26, -10, 35, 16, 78, 73, -6, 33, 31, -30, 67, -16},
  {29, 45, 6, 5, 70, 20, -68, -6, 0, 2, 18, -39, 7, 4, 32, -12, -67, 4, 29, -55, 9, 33, 45, -5, 34, -15, 71, -6, -122, 9, 100, -12, -45, 14, 2, -35, -46, 22, -19, -2, -33, -27, 68, 18, 24, 55, -40, 2, -14, -15, 32, 35, 58, -65, 4, -5, -9, -86, -13, -56, -102, -12, 8, -38},
  {35, 0, -25, -17, -122, 38, -1, 13, 11, 7, 5, 2, 20, -15, 27, -35, -78, -49, -33, -56, 9, 38, 5, 20, 19, 11, 31, -24, -84, -13, 14, 110, -40, 6, -31, 19, -51, -9, 13, 25, -101, -44, -15, -28, -31, 25, -30, -6, -21, 14, -8, -106, -98, -44, -37, -18, 5, 34, -41, -95, 0, -13, -46, -22},
  {24, -5, -23, -21, 56, 4, 21, 45, -21, -27, -4, 34, 33, -13, -4, -21, -50, -13, -41, -24, -17, -3, 17, -30, 4, 11, 21, 2, 69, 15, -33, -35, -46, -26, -42, 51, 56, 4, 17, 34, 14, -14, -56, -11, -13, -29, -8, -33, -12, -9, 2, -114, -50, -5, -16, -47, 31, 60, -6, -38, -75, 2, -72, 18},
  {22, -40, 17, -61, 36, -35, -11, 2, 25, 35, -3, 10, -40, -17, -14, -52, -22, -1, -17, 25, 18, 21, 5, -29, -14, 15, -32, 0, -93, 23, -4, -36, -33, -38, -22, -56, -61, 2, 32, 22, -63, 11, 81, 19, 1, 7, 24, -16, 35, 37, -14, 75, -93, 27, -9, 10, -30, -36, 4, -19, -65, -20, -16, 40},
  {50, -51, 32, 9, -87, -88, -12, -4, 21, 33, 28, 15, 18, 29, 15, -37, -81, -24, -18, -2, 21, 1, -2, 43, -27, -15, -26, 7, -105, -1, 52, 66, -48, -25, 16, -39, -14, 8, -3, -20, -53, 63, -16, 59, -22, -6, 35, -17, 23, -10, -17, 51, -105, 27, 37, -13, -89, -100, 34, 95, 17, 40, -3, -17},
  {4, -25, -19, -20, -75, -9, 33, 29, 27, -22, 9, 1, 68, -2, 17, 19, -8, -85, -16, -15, 6, 18, 3, 39, 3, -17, -1, -8, 4, -9, 11, -18, -24, -8, 23, -16, 56, 60, -6, -2, -21, 32, 98, 16, -85, -1, 26, 3, 0, -43, -23, -73, 27, 5, 15, -9, -82, 61, -19, 9, -1, -10, -1, 27},
  {15, 18, -54, -77, -92, -2, -4, 10, -16, -19, 1, -13, 8, 8, 4, -13, -87, -23, 16, -12, -7, 0, 4, -21, 46, 27, 1, 6, -33, 11, 55, 23, -24, 26, 0, -16, 34, 4, 5, 36, -6, -62, -43, -13, -15, -8, -13, 3, -38, -27, -5, 49, -86, 41, -10, 7, -52, 33, -4, -21, -109, 3, -11, 26},
  {17, -11, -35, -44, 77, 36, -30, -23, 22, -23, -7, 6, -6, 27, -1, 43, -119, -88, 39, -27, 30, 4, 28, 15, 19, -21, -41, -35, -118, 12, 21, -105, 0, 1, 7, -39, -55, -10, 11, 29, -6, -41, 17, -15, -77, -20, 15, 18, -36, 0, -6, -105, -121, 1, -22, 4, -91, 67, 8, -12, 61, 9, -45, 32},
  {48, -20, -3, -111, -27, -18, -21, -8, -15, -16, 8, 18, 3, 22, -21, 11, -97, -67, -8, 0, 41, -49, 0, 25, -19, -15, -8, -26, -99, 10, -13, -1, -20, -18, 24, -43, 84, -28, 14, -7, -86, 2, -72, 16, 18, -35, 36, -14, -24, -17, 22, 68, -92, -3, 12, -59, 3, 83, -1, 63, -64, 5, -33, 5},
  {-7, 7, -5, 26, 48, -42, -5, 15, -23, -5, 0, 1, 17, -18, -40, 2, 19, -20, -9, 72, -10, -49, -9, -38, -34, 10, 10, 12, -42, 42, -20, 20, -14, -32, 31, 2, -78, 45, -6, -3, -71, -24, 102, 45, 2, -6, 38, -12, 2, -33, -24, 51, -1, -27, 32, -84, -89, -44, 35, 20, 67, 13, -1, 47},
  {27, 15, 6, 56, -10, 2, 1, 9, -18, 39, 28, -15, 17, 19, -2, 7, -86, -48, -33, 4, 20, 2, 6, 5, -24, 14, 17, 36, -55, -15, -34, -50, 2, 29, -3, 26, -69, 17, 14, -23, 48, 50, 27, -43, -35, 26, 7, -20, -20, 14, -45, -26, -11, -13, -10, -5, -77, -50, 20, -22, 33, 8, 16, -21},
  {-20, 25, -3, -18, -73, -4, -4, -1, 16, 28, 63, -16, -57, 44, 31, 22, -63, 10, -62, -13, 57, 14, -50, -6, 5, 9, 26, 16, 65, -44, -18, 11, -29, -51, 9, 31, -108, -48, 1, -36, -47, 29, 35, -49, -10, 48, -27, 24, -13, -8, -70, -57, 66, -12, -12, 4, 15, -40, 43, 48, -17, 7, 38, -46},
  {-53, 0, -1, 19, -124, -35, 6, 20, 2, 4, 51, -4, -34, 33, 2, -8, -30, -11, -55, 4, 36, -29, -76, -40, -9, 52, 1, -41, -11, -32, 6, -26, -68, -46, 12, 22, 36, -36, 5, -5, 59, -30, 58, -5, -23, 36, -26, -16, -22, -23, -49, -85, -40, 25, 42, 15, -28, -71, 57, 32, -90, 30, 50, 1},
  {-69, -8, 9, -94, 33, -43, 37, -16, -17, -18, 0, -8, -8, 37, -35, -1, -28, 18, -7, 28, 0, -8, -9, -16, 0, 45, -6, -21, -49, -5, 7, 32, -10, -30, -9, -1, -78, -15, 6, 3, 28, -3, -61, 7, 30, -8, -12, -3, 20, -2, -3, 45, -53, 38, 31, 6, 10, -54, 12, -10, -18, 42, 32, 24},
  {-11, 9, 21, -20, 77, -25, 22, -3, -13, -23, 5, -13, 9, 15, -34, -14, 57, 8, -29, -8, -4, 8, -11, -6, -15, -20, -21, -11, 78, 11, -103, 11, 22, -16, -30, 42, 42, -13, 26, -13, 61, 82, -65, -17, -24, -35, 21, -8, -4, 14, 13, 31, 62, 22, 2, -49, -97, 44, 6, 82, 50, 39, 1, -4},
  {12, -7, -3, -80, -115, -1, 17, 4, -36, -55, -2, -14, 49, -32, -16, 13, 0, 4, -24, 20, -24, 17, -46, -11, -1, -38, -9, -8, -123, 38, -59, -62, 31, -19, -25, 56, 13, -8, 5, -21, -11, 54, 10, -5, -53, -31, 10, -23, -11, -12, -23, -80, 94, 11, -7, -76, -27, -48, -7, -23, -26, -3, -14, 12},
  {25, -9, -48, 12, -98, -6, 33, 29, -55, -37, 11, 37, 59, -62, 16, 25, -86, -5, 7, 25, -17, 2, -41, -7, -14, -22, 10, 21, -97, 14, 39, 31, -1, -31, 11, 52, 8, 39, 3, -1, 78, -15, -30, 34, -26, 2, 2, -47, -39, 2, -32, -85, 26, 46, -9, -29, 5, 82, 42, -22, 11, -3, -7, 20},
  {-24, -35, -15, -49, -32, 40, -59, -8, 46, 40, 55, -14, -80, 31, 63, 30, 5, -11, 51, -1, 49, -74, -33, -14, 62, -14, 46, 3, 66, -119, -56, -38, 43, 7, 60, -59, 77, -70, -17, -50, 119, -28, 109, -120, -33, 9, -40, 91, -4, -40, 4, -59, 9, -24, -16, 55, -80, -51, -61, -15, -97, 27, -48, -80},
  {-35, 32, 55, -11, -76, 41, -25, -2, 23, 15, 100, -9, -48, 67, 48, -47, -85, 68, 9, -17, 29, -41, -71, -55, 97, 58, 48, 23, 3, -84, 25, -3, -33, 25, 44, -14, -43, -97, -2, -34, 44, -38, 32, -102, -48, 28, -103, 77, -8, -32, 16, -14, -17, -14, 19, 68, 38, 97, -24, -48, -119, 33, 41, -54},
  {-71, 45, 48, -18, 70, -25, 9, -3, -2, -47, 65, 16, 7, 13, 20, -25, -122, -49, 15, 12, 5, 8, -9, -38, 25, 40, -5, -25, 20, -6, 7, -116, -68, -5, 49, 31, -58, 22, -24, -50, -85, -56, -4, -11, -16, 82, -74, 59, 34, -9, 11, 15, -111, -35, 52, 64, -57, -31, -21, 24, -42, 10, 54, -60},
  {-64, 28, 35, -57, 49, -52, 7, -7, 12, -51, 9, -15, 22, -8, 11, 2, -92, -31, 38, 57, -13, 37, 7, 7, -25, 30, 3, -41, -40, 57, -39, 52, -73, -36, 43, 9, -26, 38, -26, -14, -107, -74, -14, -24, -65, 105, -32, 30, 17, 4, -39, 19, -26, -30, 38, 30, 38, -68, -34, -82, -14, -38, 65, -19},
  {-6, 8, 35, -33, 76, -91, 3, -6, -34, -58, -15, 12, -12, -22, -22, 21, 31, 40, 50, 35, -9, 0, -12, 5, -2, 19, 48, -4, -88, 28, -19, -71, -24, -7, 47, -24, 80, -20, 1, 7, -37, -70, 6, -40, -59, 14, -1, 8, -33, -20, 8, 21, 24, 4, 15, -21, -118, 4, 12, -101, 4, -1, 28, 46},
  {-16, -13, 30, -75, 29, -17, 46, 3, 4, -55, 28, 11, 16, 15, 24, 23, 111, 78, 42, -10, -2, -47, -92, -1, 23, 23, 123, -44, 29, -30, -31, -35, 33, 12, 73, 30, 45, 15, -16, -31, -82, -15, 39, -42, -113, 12, -33, 43, -14, -43, -22, -76, 87, -29, 45, 27, -95, -71, -22, -78, -30, 15, 8, -39},
  {-20, -30, -13, -10, -69, 47, 60, 13, 6, 24, 43, 67, 46, 62, 50, 30, 70, -1, -9, -29, 19, -4, -59, -13, 37, 19, 86, -20, -37, -38, -10, -57, 13, -1, 36, 33, -33, -24, -8, -31, -35, 20, 29, -37, -17, 11, -51, 20, -17, -13, 20, -82, 108, -20, 29, 10, 79, -90, 7, -36, 25, 0, 22, -47},
  {47, 89, 51, -9, 58, 37, -38, -8, -16, 29, 4, -30, -40, 24, -21, -35, -17, -54, -21, -27, -8, 4, 32, -9, 38, 16, -11, 66, 57, 8, 22, 55, -32, 32, -37, 5, 45, -22, -8, 9, -64, -55, 81, -9, -23, 15, -41, -2, -6, 20, 28, 66, 9, -31, 4, 16, 22, -48, 27, 70, 66, 26, 30, -11},
  {-31, 38, -11, 15, 81, 76, 10, 13, 59, -18, -34, -9, 38, -28, 1, -12, -43, -44, 41, 46, -37, 70, 30, 3, 11, -9, -2, -13, -27, -6, 33, 59, 38, -6, -53, 3, 87, 14, -7, 48, -33, -56, 117, -1, 76, 74, -49, 19, 30, 29, 8, -43, -46, -27, -46, -74, -55, -54, 6, -73, -9, -50, -29, -5},
  {36, 43, 28, 21, -35, 9, -60, 0, -10, 10, -12, -13, -1, -28, -50, 29, -92, -53, -64, -6, -23, 61, 53, -35, -19, -27, -13, -39, -40, 38, -54, -23, 21, 15, -82, 40, 78, -3, 43, 13, -88, -10, 90, 27, 17, 22, -29, -38, -5, 32, 15, -29, -21, 4, -19, -34, 49, -105, 10, -39, 1, -22, 21, -52},
  {3, 83, -22, -57, 36, 8, -65, 20, -47, -29, -6, -6, 23, -30, -42, -40, -119, -1, -15, -50, -27, 66, 56, -31, 29, -37, 13, 6, -21, 45, 52, 61, -10, -25, -55, -17, 33, 33, 57, 50, -64, -32, -108, -3, -29, 0, -43, -64, 27, 31, 12, -5, -83, 9, -45, -11, 86, -18, 0, 31, -12, -56, 35, -14},
  {60, 77, -41, 36, -71, -57, -10, 35, -81, 19, -2, -31, -60, -25, -21, 40, -55, 31, -61, -75, 12, 38, 62, 13, 10, -62, 17, 42, -39, 46, 43, 76, -13, 8, -78, -4, 115, -65, 82, 36, -11, 65, -35, 17, 42, -51, -39, -113, -15, 88, 32, -58, -2, 22, -66, -25, -26, -12, 65, -85, 63, -3, 8, -40},
  {45, 25, -7, 15, -73, -22, -56, 95, -43, 30, 6, 48, -125, -20, -15, -30, -97, 49, -27, 26, 31, 25, 17, -1, -12, -3, -10, 0, -44, 18, -2, 64, -90, 56, -73, -46, -3, 4, 89, 77, -17, -38, -101, 103, 68, 29, 0, -71, -26, 127, -18, 15, -83, 22, -89, -31, -7, -109, 67, 78, -46, -43, -25, 100},
  {-34, 1, 22, 30, -88, -4, 9, -37, 55, -30, -29, -8, 63, 7, -4, -7, 15, 73, 59, 58, -43, 53, 67, 14, -2, 15, -28, -15, -49, 62, -25, -32, 27, -19, -16, 39, -81, 78, -14, 3, 98, 25, -78, 6, 3, 29, -28, 46, 45, 44, -11, 28, -49, -1, 7, 28, -18, 77, 26, 28, 31, -17, 1, 5},
  {48, 15, -3, 6, -3, 29, 17, -18, 17, -19, 21, 28, -41, 32, -3, -80, 63, -37, -35, -27, 44, -4, -1, 21, 11, -5, -12, 24, -89, -19, -13, 3, -4, -19, -44, -28, -6, -15, 1, -1, 21, 25, -11, 21, -3, -10, -6, -22, -28, 12, 44, -16, -93, -19, -8, -15, 18, 99, -9, 76, 37, -16, -50, 22},
  {0, -5, -18, -46, -65, 58, 9, 0, 33, -14, -4, 12, 1, 17, 9, 49, 51, -66, -14, 27, 2, 56, 19, 8, -7, 0, -2, -24, 41, 19, -3, 44, 2, 25, -56, 5, -10, -37, 19, 14, -47, 7, -85, 0, -23, 18, -19, -21, 1, 15, 32, -11, -82, 11, -31, -71, -59, 47, 39, -7, -37, 0, -35, -17},
  {32, 16, -10, -78, -109, 20, -2, 33, -27, 14, 13, 15, 4, -24, -6, 50, -119, -28, -22, -14, 23, 41, 15, -4, -17, -27, 33, -9, 53, 8, 22, 69, 19, 0, -28, 6, -45, 13, 38, 8, -64, -23, -102, 18, -47, 10, 10, -47, -52, 19, 43, -25, -35, -34, -36, -41, -60, -120, 5, 42, -72, 3, -21, 8},
  {54, 60, -11, 16, -125, -61, -43, 3, -79, 21, 0, -37, -29, -52, -82, -15, -37, 1, -17, -25, 10, -52, 13, -69, 25, 1, 63, 23, 2, 62, 70, 3, -52, 23, -5, -25, 49, 35, 49, 53, -86, -99, 53, 85, 54, -7, -4, -53, -5, 14, 53, 24, -42, 41, -44, 25, 94, 44, -4, -11, -113, -19, -5, 72},
  {86, 29, -23, 3, -23, -124, 7, -16, -83, 10, 18, 32, -92, 1, -128, -15, 17, -56, -33, 6, 34, -74, 24, 8, 18, 24, 2, -13, 68, 42, 26, 57, -44, -3, -9, -65, 58, -27, 32, 48, 4, -42, 21, 53, 36, -70, 49, -56, -23, 42, 43, 45, -94, 111, 1, -5, 12, -30, 40, -14, -48, 28, -5, 81},
  {86, -28, -25, -6, 24, -39, -48, -1, -13, 29, 9, 44, -114, -52, -38, 1, 44, -63, -20, 54, 68, 19, -1, 60, -44, -11, -6, -19, 5, 45, 18, 120, -15, -15, 56, -45, -103, -18, 49, -2, 73, -5, 66, 107, 57, 9, 71, -41, 19, 93, -6, -59, -77, 12, -14, -51, 75, -9, -16, 94, -53, -65, -37, 107},
  {-48, -21, -3, -11, 35, -25, 53, -9, 51, -7, -8, -9, 30, 25, 9, -28, -53, -52, 14, 34, -17, 32, -1, 26, -16, 22, -37, 20, 0, 68, 26, -15, -13, -9, -2, 18, 54, 72, -26, 6, 85, 29, -96, 25, 13, 22, -9, 48, 14, 14, 6, -33, 69, 30, 25, 53, -99, -5, 44, 57, -81, -31, 21, -5},
  {44, 2, -43, -45, 58, 3, -7, 11, 3, -12, -7, 49, -3, -5, 10, -68, 51, 18, -40, -24, 23, -2, 13, 9, 23, -27, -16, -4, 114, 3, -44, -96, 3, 11, -11, 12, -32, 2, 11, -9, -4, 27, 73, -25, -34, -32, 9, -13, -9, 27, 48, 7, 60, -49, -24, -24, -15, -7, -47, 49, 18, -59, -52, 22},
  {11, 5, -21, -105, 91, 68, -8, 19, 17, 8, -5, 22, -31, -13, 8, -40, 16, -79, -14, 9, 34, 22, 5, 2, 14, -24, -14, 11, 90, -35, -37, 53, 46, 36, -14, -2, -49, -53, 18, 31, 23, -27, 41, -29, -40, -7, -20, 11, 22, 10, 32, -26, 85, -59, -68, -38, -53, 77, 3, -73, 4, -28, -21, -43},
  {-5, 11, -1, -5, 68, 0, -8, -15, 9, 16, 15, -2, -68, 27, 12, 9, 8, -49, 30, 11, 41, -14, -37, 5, 23, 39, 26, -25, 32, -14, 19, 61, -6, 7, 4, -61, -101, -35, 6, 13, 20, -31, -102, -37, -65, -12, 3, 25, 5, -10, 74, -102, -58, -32, -4, -19, -34, 31, 7, -16, -67, 18, -3, 30},
  {2, -16, 6, -20, -101, -23, 6, -63, 25, 20, 17, -34, -16, 58, 8, -20, -74, -61, -6, 11, 17, -55, 16, 51, 1, 14, -7, 51, 65, 20, -57, 48, 14, -24, 20, -28, -72, -56, -30, -44, -80, 63, -48, -37, -31, -74, 31, 21, 37, -4, 40, 9, 82, 70, 33, 78, -31, -46, 34, -90, 2, 78, 23, -25},
  {8, -42, -18, 4, 76, -2, 43, 11, -16, -5, 0, 56, 46, 9, -5, -40, 5, -27, -15, 18, -10, -51, -61, 39, -15, 1, -9, -3, 16, 2, -58, -64, 41, -38, 33, 2, 65, 11, -4, -17, 62, -3, -20, 5, -30, -46, 53, -21, -17, -2, -9, -86, -79, 58, 21, 5, -11, -97, 28, -105, 26, 6, -17, -19},
  {-42, -73, -81, -73, -101, -60, 62, 43, 60, -13, -10, 56, -17, 4, -18, -33, -107, -2, -57, 20, -8, 42, -41, -2, -41, 65, -23, 17, 26, 23, 3, -112, 21, -74, -49, 39, -28, 83, 40, 34, 93, -16, -121, 35, -12, 65, 0, -34, 7, 54, -24, -45, -20, -4, -14, -11, -51, -6, 30, -77, -57, -67, -48, 82},
  {-24, -4, -19, 84, 30, 14, 4, 90, 26, -62, 12, 39, -6, 17, 25, 9, -98, 46, -17, 0, 10, 73, -5, -47, -18, 38, 55, 49, -79, 28, 6, 32, -3, -9, -44, 25, 77, 76, 54, 73, 25, -107, 39, 19, 60, 84, -54, -31, -51, 21, -57, -48, -98, -76, -84, -55, -76, 42, 8, -81, 0, -128, -57, 67},
  {31, 13, -92, -82, -126, 59, -15, -19, 4, -42, -23, 52, -18, -13, -6, -25, 59, -9, 0, -40, 0, -8, 27, -16, 39, -64, -42, -34, 126, 8, -24, 4, 29, -15, -17, -19, -57, 13, 8, 0, -4, 43, 37, 2, 6, -39, 16, 17, -20, 25, 41, -10, 16, -37, -33, 26, 65, 68, -85, -52, -80, -54, -65, 15},
  {-6, -1, -18, -51, -86, 23, -24, -9, 22, -21, -19, 16, -4, -14, 11, -17, -69, -48, -1, 16, 17, 17, 45, -10, 12, -39, -20, -1, -39, -2, -40, -114, 42, 8, -24, -14, 23, 1, 12, 11, -111, 8, -62, 5, 11, -11, 13, -5, 19, 14, 23, -36, -74, -22, -31, -13, -97, 54, -25, -29, 47, -45, -4, -12},
  {-27, 15, 38, -52, -26, -46, 2, -22, 1, 41, -15, -30, 3, 1, 8, 20, 25, 34, 8, -3, -25, 30, 35, 9, 3, 15, 20, 14, -16, 30, -41, -94, -16, 11, -9, 37, 61, 1, -17, -30, -57, 36, 34, 14, -100, 28, 6, 21, 30, 19, 0, -13, -14, -11, 27, -9, 74, -8, -33, -35, -39, -25, 35, 2},
  {27, -25, -1, -25, 64, -1, 30, 3, 2, 13, 44, -1, -8, 19, 13, 43, -86, -61, -51, -60, 0, -4, -21, -19, -28, -5, 38, 18, 17, -11, 28, 50, -10, 30, -7, 47, 55, -62, -1, -45, -3, 54, 23, 1, -53, -11, -17, -13, -16, 0, -11, -91, -103, -24, 12, 26, 24, -47, -8, -121, -12, 7, -1, -29},
  {-12, -46, -63, -40, 9, -2, -3, 10, 64, -2, 5, 61, -40, 65, 13, -97, -93, 54, -35, 34, 42, 3, -73, 3, -16, 57, -35, 28, 13, -41, 9, -28, 6, -7, -51, -10, -101, -43, 20, 39, -112, -2, -12, -36, 9, -28, -30, -10, -52, 3, 25, 70, 27, -28, -45, 38, -34, 15, 53, 29, -11, -42, -23, -1},
  {-5, -50, 1, -110, 15, -10, -6, 54, 70, 11, -10, 49, -46, 30, 2, -18, 57, 31, -51, 8, 12, 15, -33, -14, 10, 51, -43, 25, 51, 6, -113, -15, 16, 29, -56, -6, 103, 12, 33, 40, 55, -45, 31, 38, -58, 64, -11, 18, 11, 27, -7, 94, 20, -2, -35, 32, -93, -14, 27, -47, -108, -25, -8, 83},
  {30, -105, -34, -34, -36, -123, 19, 47, 40, 21, 57, 19, -80, 1, 77, 9, -113, -10, -87, 37, 76, 39, -88, 21, -128, 47, 17, 11, -47, 6, -19, -125, -61, -37, 33, 68, -44, 37, 2, 0, -108, 44, 6, 89, -40, 103, 10, -44, 9, 38, -77, -25, 63, -4, 56, -67, 46, -39, 54, -31, 51, 26, -14, 43},
  {18, -4, -37, -87, 48, 22, -40, 14, 29, 6, -14, 39, 12, -26, -2, -65, 15, -31, -3, -42, 7, 21, 27, -44, 4, -41, -41, -4, 8, -7, 6, -35, 10, -32, -27, 2, 75, 40, 23, 3, 64, 10, 69, 24, -7, -28, 10, -8, 24, 6, -2, 21, 54, -31, -38, 23, -49, -64, -59, -35, -75, -75, -48, 8},
  {28, 18, -13, 0, 60, 7, -8, 9, -1, -3, 2, 6, 47, -45, 7, 37, 36, -45, -19, -14, 5, 3, 4, -8, 2, -17, -17, 19, -66, 1, -10, -45, 30, 13, 12, 11, -22, 48, 2, 5, -36, 7, -5, 29, -82, 23, 3, 5, 15, 2, -24, -47, -112, -31, -24, 2, 36, 8, -21, -76, -20, -44, -10, 0},
  {32, 20, -3, -54, -125, 5, -7, 23, 30, 36, -16, -41, -4, -18, 43, 2, 15, 22, -3, 31, 14, 22, 20, 1, -1, 6, 18, 34, 46, -10, -40, 2, 14, 36, 12, 42, -113, 12, -3, 10, -55, -17, -86, 18, -38, 67, -8, 31, 19, 26, -14, -52, -25, -19, -9, -7, -46, -18, -3, -76, -87, -9, 15, -17},
  {37, 25, 50, -27, -64, 9, 20, 40, -1, 23, 66, 35, -46, 15, 42, -38, 51, -76, -28, -25, 27, 0, -15, -42, -20, 6, 48, 10, -94, -26, 69, -108, -47, 30, 34, 16, -99, -18, -7, -34, 67, 13, 31, 46, 17, 57, -34, 3, 12, -14, -19, -95, 8, -1, 23, 10, -114, -20, -10, 32, 18, 24, 35, -60},
  {-4, 11, 49, -102, -58, -27, -48, -12, 56, 8, 8, -7, -30, 38, 11, -74, -2, -64, 3, 7, 13, 10, 20, 0, -9, 10, -12, -36, 58, -10, -1, -1, 14, 6, 19, -1, 63, -18, -17, -26, -31, 16, -38, -22, -24, 47, 5, 51, 57, 16, 1, -39, -102, -60, 13, -24, 50, -86, -12, 53, -45, -24, 31, -12},
  {65, -67, 32, 23, -105, -63, -53, -10, 39, -29, -6, -19, -14, 29, 14, -42, -81, -69, 19, 37, 20, -17, 5, 88, -10, 3, 15, -17, 10, 20, -33, -120, -11, -49, 57, -54, 13, -1, -2, -9, -121, -12, 20, 50, 15, 31, 68, 22, 40, 17, -31, 49, -19, -24, 12, -61, 68, -84, -1, -13, -95, 21, -83, 56},
  {26, -33, -54, 28, -73, -91, 104, 21, -80, -26, -44, 32, 127, -26, -87, 0, -91, -47, 22, -49, -51, -39, -13, 6, -29, -20, -6, -22, -31, 87, -2, -10, -92, -78, 21, 49, 48, 62, -3, 41, -99, -5, -86, 30, -19, -128, 84, -105, -50, -3, -5, -38, -57, 61, 36, -48, -99, 19, 58, -16, -40, 17, -64, 94},
  {12, -26, -5, 30, 32, -13, -11, 0, 7, 24, -6, 14, -13, -19, -31, -1, 63, -16, 8, 1, -14, 5, 17, -6, -6, -10, -37, -6, 51, -8, 13, -102, -2, 5, -1, -16, 85, 14, 16, 7, 74, -3, 12, 37, 25, -9, 17, -9, -3, 11, -11, 80, -2, 18, -13, 22, -70, 41, -27, -11, -102, -21, -22, 11},
  {2, 14, -12, -95, 84, -17, 10, 26, -9, -17, 16, 25, 8, -9, -16, -7, -118, -24, -28, 1, -8, -17, -14, -35, 16, 25, -18, 24, -51, 5, 31, 40, -26, -18, -14, -2, 113, 11, 13, 14, -22, -12, -7, -5, -68, -32, -3, -22, -4, -19, -27, 44, 60, 6, 9, 1, 40, -76, -11, -26, -127, -8, 1, 18},
  {-3, 30, -29, -73, -70, 33, -23, -10, 2, -8, 6, -26, -19, -1, -16, -23, 17, -29, -9, -30, -5, -15, 20, 11, 43, 4, -14, 26, 67, 18, -1, 35, -12, 5, 0, -10, 33, -9, -1, -13, -68, -16, -4, -21, -71, -40, -15, 24, 17, -25, -1, 72, 9, -6, 10, 16, 91, -63, -32, 18, 14, 7, 12, -30},
  {3, 22, -57, -69, -60, 16, -30, -15, -7, -18, 10, -17, -14, -3, -4, -57, -16, -85, 14, -33, 3, -26, 39, -11, 41, -41, 2, -8, 73, -1, -10, -33, 9, 0, 10, -16, 10, -26, -6, -24, -85, 20, 49, -31, -108, -50, 2, 13, -15, -29, 3, 5, 53, -1, -11, 33, -88, -35, -37, 49, 38, -9, 6, -39},
  {-34, 20, -28, -78, -20, -22, -48, -3, -50, 16, -14, -12, 24, 23, -28, 24, -53, -28, -8, -77, -18, 7, 33, 8, -12, -74, 26, -17, 23, 78, 18, -57, 0, -11, -17, -34, 53, -23, 2, 7, -74, -9, -65, -2, -74, -44, 33, -33, -39, -9, 35, -38, 28, 48, 11, 5, 84, -7, 37, 17, -63, 16, 21, 32},
  {1, -24, -5, -74, -5, -36, -17, 18, -52, -5, 2, 22, 8, 3, -38, 54, -64, -85, -34, -53, 15, -28, -14, 28, -33, -22, 70, -3, -12, 44, 10, -108, 15, -36, -10, -6, -127, 0, 24, 5, -81, -30, -84, 46, 17, -50, 59, -61, -59, -48, 35, 23, 92, 30, 6, -29, 72, 77, 10, 69, 81, 54, -35, 23},
  {14, 32, -13, -9, 82, -24, -35, 25, -1, 45, -12, 0, 13, 14, -11, 2, -112, -29, -31, -33, 3, 32, 15, -26, 1, -13, 47, 27, -121, 9, -23, -117, 24, -50, -31, -39, -93, -11, 11, -12, -33, 23, 23, -31, -15, 35, 18, 1, 7, -31, 31, -69, -89, -11, -1, -40, 61, 108, 40, -14, -78, 39, -7, -15},
  {2, -43, -9, -11, 17, 11, 15, 26, -16, 28, -15, -1, 52, -21, -8, 5, -38, -52, 11, -5, -23, -17, -21, 24, -20, -27, -10, 6, 20, -17, -61, -59, 53, -2, 13, 12, 60, 57, 11, -23, 75, 55, -64, -5, 5, 9, 60, -7, -17, -4, -11, 65, -13, 6, -7, -21, 56, -85, -5, 88, -53, -3, -46, 4},
  {-26, -12, 12, 60, -76, -20, 15, 15, 13, 13, -14, -22, 2, -5, -4, 32, -106, -53, -3, 18, -14, -9, 9, -14, 11, 7, -13, -9, 37, -12, -8, -123, 6, -34, 6, 19, 26, 22, 7, 2, -76, 10, 87, -11, 14, 30, 22, 7, 13, 8, -32, -48, 85, 3, 0, 25, -52, 65, -2, 32, -53, 5, -2, 21},
  {-28, 14, 23, 27, -124, -25, 17, 5, 14, 2, -4, -14, -4, 1, -3, 8, -3, -19, 4, 30, -9, 27, 22, 11, -2, 42, -44, -12, -33, 20, 0, -57, -26, -37, -13, 23, 39, 35, -9, 3, -34, 0, 116, 17, 32, 58, -9, 6, 33, 10, -9, -51, 71, 4, 15, -2, -86, 20, -9, -68, 5, 3, 25, 26},
  {-4, 15, 18, -79, 75, -5, 14, -4, 30, -11, -16, -20, -11, -24, -8, 5, 53, -36, 29, 40, 7, 35, 13, 12, 5, 21, -46, -24, -41, 12, 2, -84, -4, -14, -14, 24, 61, 29, -1, 3, -34, -7, -98, 6, -20, 61, -3, 32, 44, 16, -11, 59, -124, -21, -14, -31, -62, -85, -15, -70, 22, -17, 21, 5},
  {8, 25, -14, -41, 48, -50, 13, 2, -34, -37, 3, -16, -5, -26, -47, -17, 67, 10, 12, 50, -10, 8, -1, 11, 3, 4, -34, -44, -29, -4, -3, 66, -16, -11, -16, 1, 38, 7, 12, 8, -75, 12, 4, 28, -9, -5, 13, -10, -23, 0, 0, -40, -112, 24, -2, -76, 48, -64, -5, 114, 11, 8, 4, -5},
  {63, 16, 3, -128, 8, 16, 11, 9, -56, -60, 16, 13, 20, -63, -11, 6, -96, -23, -19, 16, 6, 17, -12, -9, 0, -52, 5, 4, -6, 29, 20, 15, 16, 8, -1, 37, 12, 11, 31, 1, 103, 52, -39, 67, -46, 3, 12, -59, -9, 20, -18, 55, 53, 22, -21, -118, 40, 39, 25, -20, -48, -24, -21, 11},
  {21, -8, -79, -59, -34, 2, 43, 27, -59, 7, 40, 33, 12, -42, 21, -16, -60, -32, -2, 30, 6, -19, -58, -9, -13, -19, 41, 30, -47, -3, 48, 53, -10, 8, 23, 50, 27, 8, 23, 19, -97, -22, -57, 44, 20, -17, -16, -38, -29, 1, 4, -31, -81, 50, 1, -16, 7, -31, 24, 37, 80, 12, -11, 3},
  {37, 45, 33, -41, 54, 39, -19, -4, 0, 2, 15, -1, -31, 24, -14, -13, -65, -31, -23, -23, -8, 4, 53, 48, -7, -21, -36, 56, 24, 10, -25, 80, -17, 2, -43, 11, -55, -23, -5, -4, 20, 14, 68, -1, -17, -4, -37, 1, -4, 22, 20, 43, 109, -22, -3, 29, 6, 94, -8, -46, 41, 27, 23, -5},
  {-20, 70, 4, -18, 58, 64, 0, -17, 0, -28, -23, -23, 8, -14, -30, 44, -49, -44, 2, 27, -37, 25, 48, 6, 9, -10, 26, -10, 78, 40, -11, 5, 21, -9, -61, -21, -82, -7, -10, 22, -20, -62, -65, -18, 77, 10, -43, -10, 27, 14, 35, 28, 75, -4, -7, -71, -27, 54, 16, 26, -29, 15, 23, -1},
  {6, 27, 13, -26, 44, 26, -59, -36, -6, 19, -13, -67, -7, 12, -34, 63, -24, -62, -39, -6, 1, 2, 40, -22, 4, 2, 32, 3, 36, 34, -59, -62, 13, -13, -55, 42, 97, -55, 25, 4, -64, 20, -12, -8, -21, -29, -19, -7, -8, 24, 43, -13, -20, 4, 0, 19, -97, -92, 54, 42, -19, 13, 31, -24},
  {-21, 29, -9, -62, -52, -15, -74, -28, 17, 1, -11, -44, -3, -21, -2, -89, 6, -22, -15, -32, -4, -6, 19, -27, 11, -11, 31, -1, -11, 21, -34, -61, 22, -17, -33, 39, 98, 12, 26, 20, -34, -28, -93, 7, -39, 42, -14, 11, 55, 21, 19, 20, -25, -6, -32, 26, 4, -85, -5, 52, -106, -48, 30, -7},
  {57, 23, -44, -19, 39, -36, -25, 10, -4, 3, -6, -34, -40, -4, -2, -43, -16, -42, -26, -58, 19, 6, 26, 74, -4, -30, -13, 3, -97, 8, 56, 35, -2, 9, -47, -42, -26, -13, 39, 23, -60, 36, 19, 72, -15, 24, 9, -38, -4, 68, 16, -28, -97, 3, -23, -19, 3, -36, 39, -5, -45, 3, 8, -7},
  {46, 26, -52, 46, 64, 17, -25, 15, -8, 34, -9, 32, -88, -19, -15, -45, 50, 35, 16, 18, 21, 46, 51, 11, 12, -22, -7, 23, 46, -10, -11, -112, -41, -8, -61, -69, 85, -21, 58, 61, -85, -4, -91, 30, 97, -23, -2, -37, -16, 84, 38, -83, 60, 24, -56, 6, -82, -104, 53, -9, 21, -1, -29, 11},
  {-28, -30, 12, -44, -1, -13, -3, -11, 19, -40, -18, -3, 75, 36, -21, 4, 47, 57, 66, 46, -40, 1, 1, 35, -13, 10, -29, -12, -79, 55, 2, 86, 12, -6, 3, 35, -15, 90, -30, -25, -83, 40, 69, 32, 32, -3, 20, 41, 20, -3, 1, -24, -50, 9, 29, 27, 1, 87, 36, -6, 67, -6, -21, 25},
  {31, 16, -9, -25, -62, -17, 19, -11, 6, 14, 7, -1, -56, 33, -7, -47, 78, -3, 11, -14, 20, -16, 36, 10, 16, -8, 7, 16, 93, 1, -31, 77, -31, -8, -17, -35, 29, 6, -19, 7, 127, 3, -47, -16, 13, -28, 7, 5, -10, 14, 53, -85, 77, -24, 20, 5, -85, 83, -25, -80, -71, 1, -6, 48},
  {-11, 19, 14, -22, 42, 42, -19, -24, 39, 15, 12, 13, -87, 24, 12, -6, 88, -50, 2, 0, 26, 11, 5, 31, 6, -3, 31, -2, -42, -9, 29, 68, -7, 0, -41, -39, -25, -98, 16, 12, -104, -23, -101, 16, 43, 14, -37, 4, 36, -7, 25, 9, 76, -16, -19, -17, -49, -7, 1, 52, -27, 12, -23, -40},
  {4, -46, 6, -32, -33, -28, -77, -31, 6, 27, 48, 13, -57, 60, 1, 17, -39, -66, -4, 8, 75, -2, -38, 25, -54, 26, 49, -35, 75, -2, 122, 12, -44, -66, -26, -59, -24, -84, 28, 20, 61, -29, -30, 76, -96, 11, 2, -39, -23, 10, 50, 5, -80, -6, 0, 10, -32, 6, 77, -62, 10, 23, -53, 8},
  {42, -17, 28, -44, -16, -45, -58, -73, -36, 13, 7, -18, -10, -31, -42, 12, 47, -31, 13, -18, 32, -64, 22, 16, -12, 12, 8, 9, 35, 21, 55, -5, -8, -28, 26, -41, -102, -2, 18, 11, -12, -70, -15, 57, 1, 5, 36, -21, 17, 7, 59, -30, -91, 40, -22, 21, -18, 39, 0, -51, 0, 11, -48, 3},
  {52, 11, 16, -30, 14, 4, -20, -40, -42, 7, 6, 25, 23, -2, -29, -24, 72, -29, 53, -25, -13, -74, 42, 17, 29, -16, 6, -22, 86, 0, 44, -115, 25, 24, 42, -79, -47, 34, 2, 2, -7, -18, -60, 7, 37, -9, 60, -27, -16, -12, 5, -83, -112, 56, 25, -26, 24, 80, 6, -69, -101, 24, -17, -1},
  {49, 33, -10, 28, -57, 2, 5, -73, -68, 14, -28, 24, 0, -128, -72, 1, 23, -99, 94, -26, -19, 14, 78, 87, 30, -71, -52, -43, 97, -7, -7, -34, 79, 4, 109, -116, -55, 12, -18, -67, -7, 25, -80, 41, 71, -57, 109, -27, 53, 25, 39, 42, 72, 33, 41, 8, -37, -66, -72, 113, 20, 15, -16, 7},
  {-34, -18, -3, 10, -3, -28, -8, -6, 61, -13, 13, -1, 46, 6, 20, -33, 15, -11, 27, 35, 12, 63, 19, 26, -24, -4, -8, 45, -105, 52, 31, 14, -38, 12, 48, 51, -55, 127, -29, -1, -93, -38, 23, 75, 51, 40, -8, 17, 15, 31, 16, 39, -33, -24, 42, -61, -12, -15, 14, 42, -81, -38, 29, -16},
  {30, 45, -51, -13, 91, 3, -40, -25, 20, 42, -6, 9, -11, 10, 0, -16, 56, -71, -1, -65, 13, 44, 34, -2, 24, -66, -21, 1, -43, 1, -3, 13, 11, -4, -12, -24, 94, 44, 1, -1, -12, 35, -8, -36, -59, 14, -9, 25, -20, 35, 32, -3, -72, -51, -22, 14, -48, -44, -83, -99, -42, -32, -20, 11},
  {36, 2, 17, -57, -125, 2, -20, -4, 30, -7, 3, 33, 0, -27, 8, 0, -76, -4, -33, 8, 23, 60, 19, 7, -24, -26, 31, 16, -111, -11, -16, 42, -8, 5, -38, 16, -107, -67, 37, 10, 53, -6, 2, 39, -12, 52, -18, -6, 5, 28, -29, -78, 22, -60, -40, -20, -24, -5, -20, -58, 48, -64, -12, -43},
  {18, -39, 47, -81, 75, -8, -5, 22, -11, 58, 25, 38, 17, -43, -8, 42, -66, -9, -20, -41, 8, 48, -14, -6, -56, 4, 31, -24, 22, 6, 24, -112, -29, 16, 7, 39, -8, -5, 15, -13, 32, 42, 31, 36, -50, 32, 8, -27, 19, 10, 2, -106, 3, -50, 4, -9, -39, 12, 1, -67, -83, -36, -41, -13},
  {6, -54, 5, 57, 6, -32, 19, -28, 28, 29, -8, 31, 35, -2, 0, 80, -84, -88, -2, 4, -18, -2, -25, 59, -38, 2, -16, -1, 3, 63, -2, -86, -1, -12, 12, 40, 86, -10, 1, -20, 64, 50, 16, 31, -9, 14, 49, 19, 37, 6, 18, -52, 87, -58, 28, -23, -60, -39, 15, -101, 30, -3, -23, 18},
  {7, -29, -53, -61, 28, 11, 19, 4, -15, 17, -5, 84, 8, -5, -28, -15, -101, -7, -3, 29, -7, -63, -56, 8, -3, 40, -11, -8, -18, 14, -36, -13, 25, 0, 24, -44, -55, -5, 33, 26, 105, -61, -48, 24, 0, -32, 46, -2, -32, 9, 24, 18, 18, 22, -17, 31, -121, 8, 21, 0, -29, -2, -55, 20},
  {44, -1, -101, -48, -55, 30, 31, 36, -38, 11, -36, 74, 37, -43, -61, -45, -55, 23, -45, -27, -48, -14, 7, -52, 23, -12, -43, 13, 81, 7, -37, -56, 75, -5, -38, -4, -88, 46, 33, 30, 57, -48, -19, -29, -60, -14, 27, -24, -5, 18, 19, -32, 60, -2, -40, 30, -52, 75, -12, -40, 38, -71, -65, 53},
  {-3, -2, 0, -35, 18, 12, 27, 101, 31, -56, 13, 25, -9, 1, 27, -6, 49, 13, -25, 7, 13, 51, -26, -52, -6, 37, 48, 53, -18, 6, -34, -11, 1, 23, -10, 35, -38, 100, 17, 55, 10, -86, 55, 40, 48, 99, -48, -12, -32, 1, -32, -90, 83, -100, -16, -40, 44, -86, -7, 17, -61, -76, 16, 41},
  {73, 15, -28, -28, 29, 16, 5, 14, -10, 15, 2, 17, 17, -55, 14, -39, -8, -47, -15, -95, -10, 52, 4, -89, 13, -46, -26, 28, 20, -2, 28, -52, 24, -16, 19, 27, 46, 37, -1, -20, 20, 48, 54, 9, -42, -18, 11, 20, 16, 29, -4, -78, -86, -23, -15, 47, 13, 12, -100, -2, 47, -62, -38, -8},
  {25, 20, -17, -87, 1, 35, 18, 22, 25, -12, -6, 1, 34, -38, -1, -24, -36, 14, -33, -1, -8, 30, -4, -12, -11, -5, 11, 17, -83, 0, 2, 84, 16, 31, 1, 35, 42, 28, 12, 12, 12, 13, -93, 26, -44, 45, -23, 10, 8, 13, -26, 48, -31, -74, -10, 21, -77, -31, -10, -35, -22, -60, 22, 4},
  {18, 12, 41, -80, -14, -1, 20, 26, 32, 65, 11, 7, -32, -49, 41, -32, 29, 34, -14, 23, 5, 24, -10, 32, -18, 20, -3, -5, -61, -11, -65, -71, -1, 58, 52, 47, 50, 19, -8, -22, 2, 28, 1, 7, -23, 94, 6, 60, 62, 28, -20, -93, 34, -55, 12, 0, -31, -100, -43, -6, 0, -39, 10, -33},
  {23, -34, 42, -19, -94, -32, -20, -3, 29, 39, 2, 37, -56, 44, 20, 39, -45, -28, -12, -12, 2, 5, -19, 24, 14, 1, -20, -21, -61, 7, -1, 4, 0, -5, 1, 10, -56, -59, -5, -41, 11, 52, 99, 29, -45, 49, 2, 33, 61, -1, -12, -31, 76, -60, 25, -20, -3, -60, 41, 42, -25, -24, -6, -36},
  {-38, -1, 11, -66, -103, 4, 31, -9, 23, -14, 12, 34, -43, 20, -13, -1, -86, -21, -41, -14, 7, 10, -61, -43, 37, 52, 1, -17, 114, -22, -2, -116, 21, 10, -8, 19, 0, -66, 16, 7, -61, -21, 26, -101, 49, -24, -44, 37, 23, 15, -3, -44, -40, -58, -21, 78, -43, -30, -23, -71, -58, -28, 19, 15},
  {6, 17, -3, 7, -74, 11, -32, -4, 30, -34, -11, 0, -66, 25, -6, -22, 42, 72, -42, 39, 30, 19, 18, -90, 56, 16, 12, 47, 9, -11, -36, -78, 3, -3, -49, -34, 51, -30, 33, 51, 65, -99, 15, 19, -7, 13, -22, -1, 34, 1, 38, -55, -83, -8, -50, -1, -93, -102, 17, -8, -84, -14, -12, 51},
  {28, -24, -28, -49, -25, -76, 51, 71, 25, -14, 43, -12, -20, 35, 64, -9, 96, -16, -88, 56, 69, 64, -60, -55, -72, 72, 26, 62, 8, 9, 42, 75, -77, -36, -19, 105, 88, 26, 26, 41, -87, -9, 92, 83, 9, 89, -53, -69, -50, -10, -33, 43, -18, 24, 24, -92, -7, 26, 67, -44, -66, 29, 30, 10},
  {38, -12, 5, 0, 27, -21, 4, 6, 0, -1, 4, -4, -53, 22, -25, -27, 18, -2, 28, 10, 2, 2, -16, -34, -1, 32, -6, -11, 81, 1, 55, -63, 3, 7, 17, -18, 47, 23, 12, 13, -53, -16, -85, 48, -38, -5, 12, -1, 2, 20, -15, -64, -92, 29, 1, 22, -35, -99, -42, 41, -103, -32, 3, 32},
  {-1, 8, 0, -46, 4, 29, 13, 3, 23, -28, -1, -7, 1, -6, -33, -44, 9, -85, 8, -6, -22, -11, 24, -26, 19, 17, -41, -7, -95, -6, -8, 24, 15, -41, 18, -45, -21, 28, -7, -1, -64, 5, 64, 21, -65, -76, 0, 32, 21, -13, -5, 41, -23, -29, -4, 16, 1, 55, -48, 18, -73, -11, -15, 21},
  {-38, 17, -33, -97, 46, 18, 12, -18, 20, -24, -23, -41, 31, -6, 7, 3, 81, 54, 21, -3, 1, 33, 19, 35, 34, 28, -7, -28, 40, 20, -27, -4, 25, 4, -21, 7, 51, -1, -7, 19, 16, 9, -109, -37, -94, -18, -21, -4, 12, 1, 24, 11, 38, -25, -8, 12, -9, -32, -4, -38, 3, 7, -11, 24},
  {-31, 32, 34, -10, 69, -47, -36, -13, 32, 34, -10, -14, -89, 39, -10, -117, -48, 12, 11, 39, 1, -9, 50, -22, 6, 1, -6, -37, 44, -13, -12, 54, 0, 28, 1, 9, 72, -77, 5, -9, -65, 0, -77, 16, -101, 9, -14, 34, -6, -11, 4, 47, -122, 18, 12, 8, -116, 114, 23, 46, -83, 16, 60, -64},
  {-38, 16, 27, -44, 56, -7, -28, 21, 33, 5, 21, 16, -16, 24, -14, -10, -75, -74, -18, 0, -1, 16, 15, -24, 22, 18, -14, -45, -65, -18, 44, 41, -9, 10, -7, 23, 18, 22, -3, 5, 34, 23, -65, -4, 35, 45, -29, 25, 21, 17, -22, 58, -74, -24, 14, -35, 31, 17, -12, -59, 23, -10, 41, -17},
  {0, 17, 27, -32, -25, 15, -38, 20, 14, -10, 2, -20, -36, 8, 12, -3, 16, -18, -16, 53, 10, 4, 14, 2, 2, -9, 0, 8, 39, 27, -5, -58, 27, -16, 30, 15, -37, 37, -5, -12, -90, 14, 59, 38, 3, 59, 11, 23, 33, 14, -30, -57, -56, -27, 0, -93, 66, -73, -6, -10, -47, 4, 19, 8},
  {16, 28, 34, -27, -40, -83, 89, 19, -28, 6, 3, 16, 5, 41, -34, -18, -76, -53, -31, -68, -8, 2, 7, -35, -37, 16, 17, 20, -60, 51, 31, -48, -81, -76, -68, 16, 105, -19, 6, 17, -88, 60, -44, 11, -56, -103, -12, -60, -32, 31, -20, -4, -35, 88, 26, -33, 85, 90, 100, 52, -39, 57, 35, 15},
  {-25, -39, 11, 19, 83, 2, 29, 21, -1, -6, -9, -4, 32, 32, -22, 38, -98, -55, 10, 31, -25, -9, -24, -22, -10, 48, -39, -5, 77, 3, -28, -106, 17, -18, 12, 5, 58, 42, 9, 16, 99, 65, 6, 12, 4, 5, 13, -1, -25, -6, 8, -76, 53, 25, -1, 10, -57, -84, 0, 48, -3, 17, -18, 46},
  {-45, 11, 41, 35, 27, -29, -5, -4, 32, 20, -13, -7, -52, 23, -3, -18, -101, -54, 19, 12, -4, -21, 26, -64, 33, 23, -48, -25, 3, -31, -17, 5, -11, -33, 11, -25, 30, 14, 10, 11, -24, -72, -38, 17, 17, 12, -4, 33, 52, 12, -41, -59, 85, -9, -4, 17, 49, 62, -22, 72, -40, 9, 16, 24},
  {-59, -12, -1, -5, -113, -8, 45, -12, 8, -10, -32, -6, 37, -22, -27, 65, 40, -2, 30, 20, -30, 24, 39, 36, 14, 13, -35, -18, -64, 39, 7, -26, 23, -10, -26, 15, -95, 24, 8, 30, -47, 11, -75, -35, -26, 14, 17, 1, 45, 13, 5, -35, -76, -11, 11, -32, 36, -65, -10, 3, -8, -3, -37, 52},
  {18, 3, 17, -26, 47, -5, 29, -5, 11, -13, -42, -43, 14, -31, -11, -4, 10, 20, 19, 39, -6, 13, 15, -2, 16, -11, -11, 7, 16, -17, 4, -53, 26, 20, 8, 19, 53, 8, 13, -2, 61, 29, 45, -52, -28, -32, 10, 33, 0, -1, -31, -85, -69, -15, -10, -9, -117, 31, -40, 22, -104, -18, 4, -56},
  {58, 2, -10, -94, -9, 3, 1, 14, -56, -8, 64, -23, 5, -30, 0, 12, -25, 31, -13, -24, 23, -1, -19, -5, -61, -54, 56, -28, -98, -20, 44, 52, -47, 4, 50, 29, -34, 1, 5, -43, 100, 45, -104, 29, -25, -61, 29, -58, -127, -21, -20, 70, 61, 28, 27, -98, -70, -71, -31, 6, -90, -5, -32, -61},
  {6, -17, -36, -97, -88, 1, 3, 30, -128, -51, 59, 37, 18, -42, 7, 64, -94, -49, -29, -58, 14, -49, -56, 12, -51, -89, 63, 26, -50, 5, 0, 26, 13, 25, 52, 61, -30, 6, 10, -26, 82, 20, -85, 40, 18, -65, 43, -84, -92, -13, 33, 45, 61, 65, 41, -88, 68, 49, 17, 1, -75, 42, -29, 12},
  {9, 40, 5, -1, -60, 10, 14, 48, -51, 46, 38, 54, 13, -40, 7, -36, -29, -25, -61, -40, -4, 11, -41, -37, -22, -54, 85, 25, -104, 28, -30, 5, -49, -8, -22, 83, -93, 10, -1, -5, -45, 21, 102, 0, 41, -3, -19, -54, -21, 4, -16, 66, -38, 13, 19, 4, 8, -35, 85, 9, -40, 3, 20, -45},
  {-22, 19, 14, -87, 83, -28, -72, 13, -64, 33, 37, 1, -34, -32, -22, -60, -102, 16, 28, -42, -19, -42, -57, -87, 47, -2, 41, -31, -72, -33, -36, -44, -23, 37, 85, -45, 14, 15, -22, -25, 22, -113, -50, -37, 8, 43, -34, 32, 9, -40, -25, -70, 32, -27, 14, 3, -23, 2, -76, 27, 39, -13, 5, 19},
  {18, -7, 3, -56, 77, -25, 7, 17, 12, 33, -16, -2, 23, -10, 20, 12, 35, 29, -2, 43, 5, -4, 8, 23, 4, -11, 4, 8, 50, -62, -34, 40, 21, 47, 11, -17, 22, 27, 12, -19, -9, 14, -13, -13, -14, 13, 35, 16, 10, -7, -10, 63, -93, -7, -2, -1, 54, 39, -4, 121, -56, -8, -62, -29},
  {41, -17, -18, 81, -90, 31, 33, 13, -8, 39, -3, -24, 46, -16, 49, 41, 46, -20, -8, 13, 14, 23, -5, 7, -19, -28, 21, 58, 41, -3, -19, 7, 21, 18, 10, -8, -102, 10, 10, 8, 28, 19, -19, -36, 9, 6, 25, -2, -6, 7, 6, -25, 86, -15, -23, -19, 119, -119, -3, -11, 27, -13, -67, 12},
  {51, 1, 3, -61, -14, 56, 43, 8, -15, -7, 2, 29, -18, -11, 23, 38, -66, 27, 9, 11, 16, 24, 19, 16, 20, -44, 24, 42, -110, -12, 19, 50, 9, 28, 22, 2, 12, 18, -9, -19, 57, 25, 41, -52, 15, -14, 1, -6, 0, -11, -14, -71, -36, -13, -22, -12, -54, 7, 8, -11, -54, -7, -7, -47},
  {46, -7, -23, -63, 4, 35, 28, 15, -14, -1, 52, 23, 7, 15, 9, 0, -72, 42, -7, -7, 2, 4, -14, -3, -29, -33, 26, -26, 81, -66, -3, 13, -5, 16, 33, 7, -66, 16, -6, -21, -97, 71, -88, -11, -55, -85, -7, -5, -101, -19, -19, -51, -23, -19, 3, -61, -8, -83, -5, -36, 6, 8, -18, -109},
  {45, 42, 8, -41, -41, 31, 51, 0, -37, -49, 32, -12, 46, -15, 2, -4, 52, -55, -18, 0, -6, 27, 5, -7, 11, -56, 2, 4, 41, 3, -39, -19, 36, 12, 20, 33, 82, 22, -12, -51, -31, 67, 8, -10, -44, -38, 8, -12, 15, 13, -11, -94, 73, 5, 18, -79, 14, -1, 7, -42, 31, -9, 27, -7},
  {-10, -20, -96, -18, 7, 6, 80, 65, -21, 18, 63, 36, 48, -9, 47, -4, -54, -36, -21, 64, 12, 12, -59, -3, -13, 8, 22, 52, -62, -24, 78, -10, -50, -11, 16, 91, 67, 38, -22, -7, -102, 10, -69, 54, 43, 40, -50, -24, -37, 17, -30, -98, -41, 23, 2, 21, 54, -55, 56, 54, -25, -7, 3, -34},
  {44, 42, 43, -12, 3, 44, 11, -13, -12, -1, 13, -12, -12, -4, -25, 10, -31, -54, -22, -25, -7, -1, 47, 41, -7, -2, -53, 53, 9, 4, -33, -23, -1, 26, -12, 39, -78, -41, -14, -1, 32, 21, 74, -2, 4, -17, -10, 5, 12, 19, 22, -43, 31, -17, 21, 36, -9, -21, -37, 56, -57, 35, 28, -1},
  {12, 52, 4, 13, 33, 11, -3, -13, -22, -42, -27, -22, 2, -30, -46, 74, 45, -75, 4, 47, -25, 27, 33, 24, 3, 6, 14, 4, -63, 28, 21, 60, 29, -24, -41, -22, -12, 3, -11, 19, 50, -66, 65, -20, 57, 9, -4, -4, 1, 11, 47, 101, -72, -9, 2, -59, -38, 114, 14, -28, -21, 5, 12, 31},
  {13, 39, 33, -72, 88, 29, -8, -20, 12, -21, -9, -14, 22, -33, -27, 58, -77, -15, 5, 36, -13, 24, 18, 5, 10, 0, -14, -18, 0, -5, -27, 60, 11, 5, -22, 6, 71, -13, -2, 2, 98, -30, 59, -21, 60, 24, -15, 22, 5, 3, 26, -34, -82, -13, 5, -36, 77, -4, -10, 17, -36, -5, 2, -8},
  {1, 3, 45, -10, 5, 14, -39, -18, 16, 12, 3, -15, 36, -16, -11, -14, 70, -31, 9, 28, 14, 20, -1, -13, 0, 9, -27, -37, 9, -8, 18, 27, 2, 1, -1, -2, -61, 12, 8, 1, 25, -15, 35, 1, 50, 57, 8, 7, 19, -9, 7, -65, 63, 16, 6, 3, 37, -9, -4, -68, -91, -4, 7, -8},
  {9, 2, 9, -85, 45, -27, -16, 4, 28, 12, 14, -2, 32, -18, 4, -25, 30, 0, 6, -7, 9, 8, -6, 3, 9, 3, -32, -14, -128, -26, 19, -124, 5, -4, 6, 31, -119, 28, 3, -14, -43, 27, 35, 11, -41, 38, -10, 7, 29, -4, -39, -52, 3, 21, 14, 12, 33, -111, -20, 72, -49, -4, 8, -13},
  {1, 9, -4, -50, -80, -29, 26, -5, 16, 2, 1, -9, 1, -38, 16, -17, 21, 10, 28, 4, -11, -3, -3, 49, 23, 16, -47, -10, -15, -29, 32, -68, 4, -2, 20, 16, -73, 33, -3, -4, 3, 41, -27, -1, -7, 5, -2, 6, 22, -6, -48, 48, -78, 13, 11, 18, -84, 75, -35, -14, 30, 5, 23, 15},
  {19, 2, -13, -38, 52, -27, 19, -8, -11, -34, 4, -19, -66, -31, 18, -10, -42, 28, 56, 18, 9, 4, 16, 47, 16, 9, -30, -28, 73, -44, -22, 5, 10, 26, 48, -21, 21, 29, -4, 9, -32, -18, 80, 15, 42, -41, 29, 17, 10, 12, -17, -48, 6, 27, 4, 17, -60, -92, -44, -13, -76, -17, -6, 43},
  {44, 11, 23, -5, 16, -20, 37, -18, 2, -17, -2, 17, -34, 3, -5, -61, 20, -20, -10, -6, 20, -20, -1, 29, -4, -13, -12, 15, 0, -4, -67, 58, 7, 4, 5, -2, -85, 2, -21, -3, 81, 30, 56, -19, -14, -15, 14, 12, 15, 12, 44, -76, 45, -16, 6, -6, -95, -51, -38, 1, -50, -10, -8, 19},
  {-10, -18, 13, -29, -62, 34, 25, -26, 25, 9, -5, 9, -20, 20, 10, -24, -75, -86, 7, 14, 23, -5, -8, 25, 0, 14, -10, -14, 14, -2, -42, -26, 13, 24, -4, -1, 104, -36, -16, 5, -26, 2, 37, -24, -37, -10, -9, 20, 27, -4, 22, -77, -28, -18, 0, -33, -103, 12, -14, 32, -53, 6, 2, 3},
  {-5, -18, 15, -93, -77, 24, 23, -9, 16, 12, 26, -1, 5, 1, 37, 1, 40, -50, -1, 16, 18, 9, -8, 29, -23, 1, -10, 7, -8, 2, -27, 67, -10, 16, 14, 32, -40, -39, -22, -22, -33, 24, -36, -5, -62, 26, 13, 18, 5, 4, -5, 29, -99, -18, 17, -24, -53, 18, 11, -4, -118, 22, 11, -30},
  {-13, -1, 9, -48, 76, 31, 50, 23, 19, 16, 7, -3, 31, -20, 28, 12, 16, -47, 10, 3, -9, -6, -35, 21, -5, 4, -10, 5, 36, -31, -20, -72, 11, 27, 50, 40, -105, 33, -23, -25, -111, 16, -12, -25, -78, 26, 6, 21, 8, -6, -46, 3, 18, -19, 10, -8, -50, -43, -14, -90, -29, 14, 19, -25},
  {-9, -2, -8, -26, 57, 2, 35, 19, 24, 15, -9, -16, 34, -25, 10, 18, -41, -21, 4, 29, -8, 14, -12, 6, 2, 20, -19, 26, 110, -29, -4, 45, 15, 14, 11, 17, -99, 47, -6, -13, -102, -23, -65, -9, -48, 42, 9, 23, 25, 7, -45, 47, -26, 11, 18, 18, -77, 24, -3, -14, -8, -8, 11, -13},
  {9, -15, 34, -9, -40, 13, -29, 9, 25, 26, -4, -12, -5, -17, 17, 16, 69, 35, 13, -5, 16, 2, -9, 34, -21, -18, -6, 13, -115, -35, -8, -13, 22, -10, 26, -39, -74, 15, 6, -1, 19, -13, 9, 16, -81, 38, 33, 17, 28, 4, -20, 24, -46, 6, 9, 19, 11, -8, -45, -104, 2, -10, -19, 7},
  {17, -5, -18, -51, -42, 11, -21, -2, -20, 1, 6, -5, -61, -33, -3, 19, 46, 35, -19, -8, 9, -30, -8, 20, -9, -16, -5, 1, -3, -4, -74, 10, 40, 0, 29, -21, 61, -22, 14, -12, 83, 15, -15, -21, -54, -15, 40, 5, 16, -2, -1, -19, -98, 34, 10, 15, -65, -117, -61, -45, 74, -8, -22, 37},
  {17, 14, 7, -70, -1, 53, -24, -16, 27, 4, -7, 38, -20, 0, 8, -28, 32, 16, 10, -67, 12, -33, 10, -5, 29, -35, -19, -25, 71, 22, -93, 72, 21, 19, 9, -26, 72, 17, -9, -35, 54, 55, 23, -41, 7, 9, 18, 51, -7, 11, 42, -50, -20, -45, -9, 16, -46, -90, -71, -106, -52, -8, -7, 31},
  {10, -12, 4, -29, -54, -6, -44, -27, -5, -47, -19, 18, 2, 5, -1, -24, 55, -1, 31, 17, 20, -27, 43, 27, 6, -33, -15, -4, -59, -7, -45, -56, 17, -39, 4, -36, 50, -10, 9, -23, -71, 42, -55, -48, -43, -22, 44, 12, -4, 9, 12, -49, -107, 22, 1, 0, -63, -89, -27, -46, -55, 12, 8, -20},
  {-50, -10, -22, -2, 104, -68, -15, -4, -6, -61, -16, -3, -21, 46, -2, -44, 41, 20, 19, 29, -10, -71, 18, 25, 29, 15, -6, -2, 39, -19, -40, -42, -14, -71, 24, -14, -62, -25, -19, -24, -46, 45, -80, -77, 11, -61, 12, 44, 1, -8, -10, -97, -70, 27, 21, 40, 25, -59, -16, -96, -65, 28, 24, 18},
  {-15, 6, 1, -25, -1, -11, -14, -4, 40, -20, -38, -1, -3, 38, 25, -45, -116, 22, -11, -33, -22, 17, -30, 34, 18, -21, 7, -6, -15, 1, -49, -29, 30, -30, 13, 1, -54, -1, -21, -25, -36, 42, 51, -76, -10, -1, 2, 32, 9, 2, 3, 55, 74, 2, 31, 59, 98, 41, 1, -10, -41, -4, 20, -14},
  {-59, 28, -61, -71, 46, 3, -63, -22, 31, -15, -39, -8, 19, 99, 6, -97, -18, 45, 1, -29, -34, 99, 41, 1, 15, 7, 12, 21, 22, 57, -34, -42, 0, -21, -106, -19, 7, -57, 10, 49, -12, 19, -63, -52, -23, -25, -59, -46, -70, 38, 60, -57, 18, 14, 5, 36, 34, -118, 61, 15, -55, 12, 26, -49},
  {-46, 22, -52, -73, -85, 26, -22, -16, 42, -19, -16, -25, 46, 41, 36, -73, -24, -50, 21, 7, -8, 79, 48, -14, -6, 6, 21, -3, 16, 34, 25, 98, -43, -13, -32, -5, -91, 24, 1, 14, 24, -11, -59, 14, 11, 29, -45, -31, -36, 36, 25, 54, -98, -42, -24, -57, 76, -113, -4, -116, -38, -30, -18, -35},
  {-3, 0, -61, 2, 23, 4, -34, -11, 32, -18, -7, 4, 30, 47, 15, -76, 47, 0, -7, -49, 6, 45, 57, -3, -20, 8, 13, -14, -30, 27, -18, -125, -21, -53, -58, 19, -100, 16, 15, 13, 93, -20, -91, -4, -71, 12, -28, -23, 5, 9, 25, -40, -117, -24, -21, -51, -31, -54, 2, -21, 77, -5, -35, -11},
  {51, -3, -2, -26, 79, 7, -57, 3, 4, -18, 3, 22, -25, -40, -6, -22, -18, -23, 12, -80, 19, 13, 9, -34, -19, -44, -16, 37, -13, 0, 45, 61, -6, -25, 15, -50, -84, -31, 22, -18, -77, 9, -86, 59, -37, -32, 9, 8, 50, -6, -13, -35, -61, 21, -7, 51, -85, 12, -88, -44, -49, -35, -38, -24},
  {17, -53, -2, -90, -2, -82, -66, -31, -17, -1, 13, -7, 17, 5, -46, -54, -94, -4, 80, -70, 21, -29, 21, 11, -48, 7, -5, 19, -96, 17, 88, 36, -32, -77, 17, -85, 34, -21, -5, -7, 71, -27, 66, 90, -24, -49, 54, -61, -18, -20, 20, -71, 71, 60, 42, 19, 39, 83, 27, 10, 20, 32, -16, 14},
  {6, -77, -32, -63, -71, -28, -18, -39, 25, 4, -22, 22, 20, 52, 9, -38, -91, -40, 68, -26, 27, -37, 66, 57, -81, -21, 13, 1, 85, 5, 81, -103, -8, -39, -26, -59, 35, -31, -5, 21, 26, 18, -14, 61, 48, -48, 60, -40, -34, 7, 60, -86, 14, 96, 38, 30, 120, -50, 67, -51, 12, 73, -29, -24},
  {4, 33, -42, -4, 107, 4, 6, 8, -37, 8, 27, -15, 40, 16, 4, -81, -35, -4, 41, -23, 7, -7, 54, -41, -36, 15, 59, 4, 33, 21, 121, -18, -64, -20, 5, -20, -102, 44, -11, 21, -23, -122, 49, 39, 79, 17, -22, -38, -86, -26, -13, -73, 88, 47, 22, -4, -109, -63, 72, 39, -37, 32, 32, -16},
  {15, 55, -62, -66, -109, 20, -42, 5, -51, -53, -7, -37, 23, 15, -2, -73, 38, -47, 80, -52, -24, 20, 49, -34, 46, -6, 34, 2, -115, 43, 22, 30, -16, -22, -19, -45, 50, 60, -11, 10, -50, -61, 55, -31, -18, 26, -18, -3, -22, -23, 32, 40, -118, -37, 7, -18, -109, -118, 26, 33, -61, -8, 42, 23},
  {39, 37, 6, -41, 80, 30, -42, -50, 31, -91, -1, -61, -30, 61, 22, -38, -76, -82, 86, 41, 26, 31, 44, 26, 43, 12, 29, -13, -27, 19, 84, -13, -57, -57, -25, -95, 12, -29, -27, 27, 16, 27, -108, -16, 20, 6, -19, -3, 25, -4, 12, -116, -45, -11, -1, -33, 20, -111, 42, 17, 58, 29, -8, 31},
  {25, 10, 48, -29, 1, -51, -29, -23, -29, -58, 25, -2, -26, 45, -8, 28, -48, 0, 12, 66, 11, 13, 19, -9, -55, 0, 58, -2, -99, 56, 42, 56, -125, -68, 9, -22, -5, -34, -14, -4, -114, 27, 91, 36, -22, -12, -3, -46, -3, -20, -33, -46, -5, -3, 42, -114, 42, -83, 57, -25, 10, 35, 40, 30},
  {24, -50, 19, 3, 81, -65, -26, 15, 13, 17, 13, 5, 8, 11, -44, -35, -1, -29, 12, -13, 1, 4, -11, 14, -34, 13, -29, -7, 41, 41, 38, -55, -44, 21, -23, -33, -101, 62, 25, 32, -3, -17, 109, 55, 1, 27, 13, -27, -4, 43, -60, 68, 48, 26, 16, 4, -51, 65, 21, -36, 35, -24, 13, 39},
  {65, -25, 55, -5, -48, -90, -41, 11, 8, 41, 57, -5, -28, 63, -3, -13, -78, -16, -15, -32, 47, 11, -56, 83, -75, 22, -17, -4, -93, 33, 64, 61, -96, 15, 9, -51, -38, 37, 25, -10, -19, 69, 4, 35, -54, 65, 22, -43, -1, -3, -49, 15, -72, 29, 32, -38, -123, -124, 74, 110, -99, 19, 16, -3},
  {18, 10, 27, -38, -108, -12, -9, 47, -9, -50, 49, -12, 26, 83, -10, 29, -12, -54, -15, -37, -2, 2, -27, 31, -31, 22, -23, -10, -60, -2, 29, 54, -77, -27, 1, -11, -72, 31, -1, -24, -6, 34, -61, -11, -5, 15, -12, -37, -23, -54, -40, 45, 40, 38, 25, -58, -42, 31, 49, -11, 32, 49, 11, -27},
  {-12, -21, -15, -37, -33, 29, -24, 45, -19, -119, 20, 36, 0, 10, 3, 27, -117, 47, -7, -54, 17, -32, -37, 4, 8, -17, -9, -46, 16, -22, -3, 63, -7, -77, -8, -36, 13, -12, 14, -2, -7, -39, -96, -50, -25, -40, 22, -11, -45, -50, -17, 10, 3, 48, -47, -49, 57, -25, 7, 81, -88, 35, -63, 16},
  {-24, -4, 3, 10, -4, -17, -36, -13, -19, -91, 22, 12, -29, 38, 9, 26, 71, 91, -14, 1, 20, -26, -23, -85, -1, -16, 0, 26, -113, 7, 3, 47, 2, -13, 26, -26, 104, -25, 15, 11, -65, -67, 40, -22, 4, -16, 24, 22, -26, -68, 1, -37, -99, 9, -1, 0, 43, 71, 40, -34, -37, 38, -2, -3},
  {-20, 17, -1, -33, -1, 14, -48, -23, -64, -66, 17, -53, 5, 21, -14, 52, 50, -23, -10, 10, 28, 5, 29, 17, -64, -102, -27, 24, 38, 38, 27, -109, 16, -78, -51, -40, 66, -12, 33, -17, -71, 41, 24, 45, -56, -23, 37, -80, -76, -20, 18, 67, -1, 61, -5, -110, -119, -59, 57, 45, 69, 65, -13, -24},
  {-17, 3, -14, -21, -15, -24, 9, 15, -56, 13, 31, -2, 25, -43, -6, 15, -43, -39, -7, 25, 0, -16, -26, -30, -64, -64, 3, 28, -94, 32, 5, -90, -8, -7, 7, 53, -104, 45, 2, -9, 88, 23, 99, 23, 51, 3, 23, -36, -43, -38, -30, -96, 96, 15, 17, -83, -30, -73, 44, 70, -24, 26, 3, 4},
  {3, -14, -17, -94, 70, -5, 23, -2, -24, 46, 12, -3, 53, 12, 26, 63, -86, -57, -12, -13, -14, -23, 15, 49, -50, -53, 1, 9, -101, -17, -79, -9, 26, -12, 15, 26, -27, 22, -14, -70, 93, 121, -66, -9, -34, -8, 56, -6, -46, -9, -7, 15, 30, 27, 19, -18, -63, -77, -3, 44, -77, 48, -39, -28},
  {-29, 33, -9, -1, -108, -45, 3, 20, -39, 25, 71, -8, -23, 83, 42, 26, 36, -14, -37, -23, 25, -24, -57, -21, -11, 37, 25, -14, -71, -40, -7, -70, -71, -12, 49, 28, -84, -16, -14, -25, 36, 38, 79, -50, 17, 41, -28, 11, -33, -28, -26, 6, -13, 26, 33, -10, 70, -29, 21, 0, -53, 40, 39, -2},
  {-107, 27, -3, -96, -46, -45, 40, 47, -44, -19, 61, 51, -58, 63, 14, 26, 8, -16, -62, -39, 8, -29, -58, -117, -6, 43, 47, -54, 9, -53, -24, -68, -43, -12, 25, 61, -83, -34, 4, -3, 76, -38, -103, -86, -39, 30, -37, -21, -32, -29, 29, -69, -22, 35, 19, -32, -78, 38, 50, 28, -16, 30, 23, 19},
  {-125, 22, 8, -94, 43, -14, 29, 20, -62, -29, 22, 22, -11, 36, -32, -46, 15, 1, -36, -24, -14, -2, -17, -75, -1, 28, 67, -66, -77, -12, 22, 7, -11, -14, -33, 42, -8, -22, 14, 27, -63, -32, -8, -42, -18, -11, -26, -40, -17, -6, -9, 15, 80, 51, 4, -38, -120, 43, 32, 43, 27, 16, 26, -6},
  {-37, 13, 23, -30, -41, -19, 62, 24, -63, -75, 3, -33, 57, 0, -39, 17, 53, 52, -3, -43, -14, 5, 2, -54, -12, 11, 34, -20, 91, 46, -42, -3, 11, 13, -9, 44, -71, 25, 21, -4, -51, 30, -49, -56, 28, -27, 14, -18, -30, -7, -45, 7, 65, 39, 14, -40, -53, 0, 1, -25, -40, 52, 18, 10},
  {31, -28, -68, -97, -37, -105, 56, 8, -91, -95, 13, -2, 59, -1, -40, 1, 66, 25, 18, -55, -26, -91, -74, 7, -48, -21, 109, 9, 73, 66, 2, -18, -5, -113, 3, -15, -54, 6, 18, -40, -94, 15, -116, 12, 17, -113, 79, -78, -106, -28, 43, 14, 90, 106, 37, -49, 46, 35, 31, -30, 19, 79, -77, 39},
  {48, -4, -106, -5, 48, 16, 59, 17, -68, 6, 26, 65, 46, -57, 9, 19, 3, -51, 13, -70, -2, -44, -59, -3, 37, -38, 80, 24, 76, -27, 28, 69, -4, -17, 32, 22, -29, -11, 23, 38, 0, -29, -52, 12, 4, -41, 8, -48, -53, -19, 86, -42, 39, 37, 7, -3, 74, 26, 11, 4, -53, 48, -52, 36},
  {-36, 25, 5, -99, -57, -50, -44, 48, -80, 56, 52, -29, -13, -61, -30, -32, 51, 34, -18, -35, -31, -87, -55, -74, 75, -5, 68, -30, -116, -43, -80, -117, -29, 33, 127, 25, 51, 24, -41, -44, -41, -107, 18, -84, 2, 55, -20, 32, 2, -59, -53, -35, -13, -54, 9, -30, -81, 55, -83, 3, -15, -38, -45, 17},
  {-73, 14, -12, -76, 35, -9, -36, -20, -35, 18, 81, -13, -32, 10, 0, -117, -67, 68, 40, -50, 3, -93, -53, -37, 87, 37, 51, -11, 89, -91, -93, 5, -18, 10, 109, -16, -18, -29, -27, -45, 70, -106, -58, -99, -7, 34, -59, 102, -41, -64, 4, 55, 0, -3, 62, 47, 55, 48, -51, 66, -58, 13, -39, -38},
  {-127, 63, 5, -87, -23, -89, -33, -35, 12, -85, 68, -64, -69, 64, -37, -44, -85, -16, 67, 70, -3, -37, -44, -46, 72, 127, -50, -53, 41, -11, -9, 47, -94, -39, 81, -8, -71, 29, -24, -6, 22, -120, -61, -64, -4, 80, -128, 127, -14, -38, 7, 34, 93, 5, 122, 127, -37, 62, 21, 8, 78, 14, 89, -36},
  {-94, 103, -21, -23, 4, -84, 28, 38, -10, -105, 51, -11, -24, 59, -44, -31, 41, 23, 26, 96, 1, 11, -94, -77, 7, 103, 17, -48, 90, 25, 19, -63, -128, -96, 12, 1, 20, 76, -13, -22, -34, -80, -6, -5, -33, 62, -96, -8, -37, -2, -32, -73, -19, 64, 74, -8, 55, 82, 81, 97, -55, -9, 109, 82},
  {-62, -6, -33, 46, -59, -128, 103, 49, -22, -89, -25, 44, 71, -9, -43, 20, -49, -32, 52, 116, -20, 44, -98, -57, 7, 81, 20, -44, -9, 59, 67, 12, -88, -69, 13, 18, 2, 63, 8, 29, 7, -74, 34, 36, -41, 25, 10, -74, -25, 8, -50, -102, 33, 30, 41, -52, 12, -95, 74, 81, -20, -30, 57, 85},
  {-60, -78, -25, 22, -80, -33, 100, -5, 42, -57, -18, 62, 55, 67, 26, 28, -29, -40, 49, 64, -15, 17, -128, 28, -23, 33, 25, 9, -9, 8, -50, -7, 27, -69, -24, 9, -48, 38, -25, 28, -99, 88, -94, -92, -61, 2, -16, 2, 17, 4, -12, 55, -33, 23, 61, 11, -93, -49, 84, -60, -71, 43, 18, 31},
  {-31, -25, -2, -36, 88, 28, 78, -15, 35, -40, -12, 82, 67, 50, 31, 31, 95, 0, 9, 19, -1, 7, -44, 14, -36, 7, 21, -45, 91, -1, -44, 92, 63, -17, -11, 9, 88, -15, -12, -14, 30, 102, 56, -113, 10, -59, 19, -10, -52, -30, 24, -57, -13, 14, 36, -17, 60, 48, 30, 20, -40, -17, 2, -17},
  {21, 23, 15, -23, -61, -2, 0, 1, -8, -9, -7, -16, -31, 2, -26, 21, -8, -19, -7, -2, -7, 5, 23, 26, 3, 9, -6, 22, -100, 7, -17, -116, -3, 5, -20, 6, 29, -24, 1, 18, 68, -32, 9, 17, -42, -1, -5, -9, 23, -3, 17, -29, -89, -11, 6, 5, -16, 29, -2, 5, -16, 4, 14, 2},
  {-14, 25, 13, -1, 70, 14, -18, 2, 1, -18, -11, -22, 0, -1, -21, 1, -34, -59, 14, 16, -12, 11, 12, -32, 3, 10, 12, -44, 62, 15, 5, -67, 6, 23, -16, -14, 25, -6, 1, 17, 43, -57, -45, 3, 15, 38, -32, 20, 16, 7, 28, -90, -48, -20, -9, -43, -106, -71, 21, -63, -73, -7, 10, 11},
  {18, 20, 19, -33, 46, -5, -6, -2, -17, 0, 5, -14, -10, 7, -25, 2, -33, -58, -37, 12, -11, 12, 15, -26, -11, -8, -2, -1, 76, 18, -42, -77, 13, -10, -42, 24, -56, -9, 17, -2, -53, -3, 77, -5, 30, 8, -8, -17, 15, 13, 1, 57, 78, -2, -3, -12, 71, 47, 10, -93, -65, 17, 7, -24},
  {-27, 32, -9, -22, -3, -11, -6, -15, -1, -19, -7, -15, 27, -5, -4, -8, -56, -67, 12, 2, -16, 9, 24, 7, 20, 8, -12, -31, 5, 34, 16, -73, -6, 3, -14, -1, -114, 16, 1, 10, -48, -12, 37, 13, -2, -1, -8, -5, -4, -11, -9, -76, -29, 26, 6, 3, 64, -64, 3, 78, -70, -12, 32, 19},
  {-4, 18, -6, -30, 44, -18, 4, -15, 10, -4, 7, -16, -7, -1, -17, 1, 59, -43, -1, -2, 0, 7, -1, 8, 6, 8, -7, -8, 58, 10, 6, 2, -15, -16, -17, -9, -44, -11, 16, 10, -72, 4, -3, -7, -54, -2, -24, 2, -4, 6, 5, -9, 87, 9, 2, -2, -57, -98, 13, 37, 54, -8, 24, 3},
  {14, 22, -2, -11, -14, -2, -38, -10, -10, 11, -5, -7, -3, 1, -30, -31, 96, 5, -7, -18, -4, 24, 28, 2, 1, -20, -9, 12, -28, 14, 3, -55, -6, 13, -19, -33, 23, -3, 26, 9, 16, -36, 71, 6, 23, 13, -9, -10, 2, 25, 16, 68, -78, 0, -16, -8, 63, 4, 17, 99, 7, -24, 15, 16},
  {-2, 13, -18, 8, -30, 2, -22, -5, 7, -17, -3, -19, 17, -5, -14, -10, 37, 20, 4, -8, -8, 27, 5, 10, 2, -21, -11, 12, 77, 22, -1, 20, -10, -1, -24, -15, -5, -1, 4, 4, 70, 2, 14, 13, 41, 9, -11, -2, 5, 21, 5, -42, -12, -11, -17, -6, -104, -3, 12, 62, -76, -18, 11, 4},
  {18, 11, -13, -20, 69, -4, 9, 8, 18, 7, 16, 8, -35, 27, -8, -30, -34, -23, -12, 14, 20, -26, 10, 10, 29, 19, 7, 7, 56, -18, 14, -28, -24, 13, -12, -5, -27, 2, -2, 1, -36, -20, -63, 15, -11, -9, -17, 8, 5, -9, 25, 12, -102, -13, -7, -8, -87, -58, -8, 44, -19, -8, 6, -7},
  {8, -6, 14, -58, -108, 19, -10, 3, 26, -6, 3, 17, -16, 16, 4, -4, -80, -39, -30, 5, 16, 15, 10, 3, -1, 11, 27, -34, 36, -6, 6, -21, -8, 11, -30, 3, 82, -21, 2, 7, -94, -12, 95, 30, 21, 26, -11, 4, 30, 18, 12, 13, -20, 14, -4, -20, -50, 6, 31, -12, 9, 3, -25, 5},
  {35, -13, 13, -19, 76, -7, -17, 18, 8, 6, 20, -2, 11, -4, 29, 10, -27, -57, -6, -17, 12, 14, -20, 11, -23, 14, 17, -33, -83, 12, 7, 44, -4, 9, -8, -8, -62, 7, 3, -14, -28, 25, 16, 58, 1, 18, 13, -17, -10, 13, 22, -5, 64, -8, 7, -38, 30, -46, 22, -66, 39, 20, -30, 4},
  {2, -19, 28, 0, -123, -50, 8, 18, -9, 5, 24, 11, -25, -3, -16, -29, -21, -29, -50, 15, 17, -30, -20, -18, -22, 68, -7, -12, 52, 30, -47, -66, 1, 20, 16, 35, 56, 4, -9, -12, 35, 13, -60, 21, -64, -5, 4, 25, 10, -10, 11, -54, 30, -4, 29, 14, -102, -6, -13, -76, -28, 35, 9, 28},
  {2, 4, 4, 6, -23, 2, 16, 21, -16, -9, 2, 40, 37, -29, -7, -28, -38, 6, -20, 9, -12, -25, 19, 3, 6, 12, -21, 9, -37, 14, 6, 2, 0, -11, 17, 9, 75, 30, -6, -16, -1, 17, -25, 10, -35, 35, 2, -4, 5, -12, -11, -40, -1, 27, 13, 11, -37, -20, -31, -105, 41, -4, 15, 31},
  {-3, -10, 8, 6, 14, 2, 5, 14, -19, -4, 20, -22, 38, -14, -13, -2, -99, -24, 30, 21, -9, 2, -29, 14, -4, 17, -8, -42, 57, 7, 35, -100, -2, -3, 48, -5, 8, 36, -20, -18, -5, 31, -82, 24, 18, -3, 31, -18, -7, -22, -11, -36, 71, 21, 44, -6, 54, 36, 8, 54, -24, 3, 12, 24},
  {19, -23, -2, 51, -53, -4, -19, 21, 17, 24, 19, 6, -47, 5, -11, -21, 61, 74, -41, 8, 35, -7, -19, 39, -20, 18, 15, 1, -89, -6, -34, -71, 15, -5, -26, -18, 109, -30, 32, 1, -3, -1, 59, 6, 42, 8, 7, -16, 2, 38, -8, 69, -28, 15, -30, 19, 85, -76, -11, -2, -65, -14, -7, 13},
  {24, 9, -64, -11, -96, -1, -6, 0, 9, 11, 4, 10, -14, 28, -10, -32, 8, -15, -27, 4, 5, -3, 1, -15, 17, 0, 3, 3, 47, -6, 2, -111, 17, 21, -8, -3, 71, -1, -5, 20, -20, 16, -96, 0, -1, -27, -1, 11, -10, 2, 30, -76, -60, -6, -27, 4, -82, 29, 3, 10, 55, -25, -30, 15},
  {11, -35, -55, -44, 6, 35, 11, -4, 32, -5, 10, 24, 5, 21, 37, -6, 18, -26, -4, 4, 18, 31, -5, 26, -1, -5, 13, -9, -83, -6, -4, -47, 27, -22, -17, -11, -81, -25, -6, 13, 61, 42, -55, 2, 12, -25, -8, 4, -10, 12, 31, -65, -95, -22, -17, -13, 29, -109, 23, 24, 50, -19, -33, -15},
  {6, -47, -10, -39, -98, 24, 13, 18, 60, 13, 11, 17, 1, 30, 42, 31, 60, -59, 1, -4, 17, 30, -23, 50, -22, 15, 7, 4, 35, -9, -35, -50, 23, -6, -39, 13, -76, -1, 6, 18, -54, 58, -38, 4, -11, 14, 2, -16, -7, 13, 37, -41, -35, -52, -26, -11, -122, -48, 0, -71, -33, -26, -43, -7},
  {-2, -38, 3, -66, -122, -15, 56, 7, 35, 17, 27, -18, -38, 35, 10, 22, 13, -99, -41, 8, 11, -30, -20, 17, -28, 42, 30, 29, -62, 19, -44, 60, 15, 11, 4, 33, -32, -19, -10, -25, -11, 84, 27, 3, -80, -37, 6, 26, 19, -13, 18, -63, -16, -20, 14, 57, 6, 67, -36, -95, -38, 23, 28, 8},
  {-30, -44, -11, -47, 20, 1, 52, 15, 4, 7, 14, 52, -14, 25, -4, -4, 11, -16, -38, 19, 24, -65, -36, -7, 4, 35, 19, 47, -47, -39, 2, -117, 22, 10, -10, 6, -111, -24, 6, 20, -91, 15, -5, -9, -66, -61, 0, 1, -15, -26, 9, -48, -21, 32, -9, 34, -79, 4, -25, 0, -1, 28, -1, -2},
  {-16, -29, -10, -13, 21, -10, -18, -11, 0, -18, -6, 14, -22, 22, 0, -17, -101, 64, 45, 5, 10, -10, -9, 4, 3, 25, -19, -24, -14, -10, 6, -43, 5, 3, -1, 9, 27, -4, 10, 15, -51, -21, 88, 11, -108, -18, 23, -2, -11, 4, 19, 29, -117, 39, 1, 49, 57, 30, -35, 1, -31, 19, -12, 21},
  {-11, -7, 18, -35, 1, 1, 13, 6, 8, -6, 6, 16, 2, 14, 4, 18, -10, -12, -15, 22, 25, 0, -11, -40, -14, 41, 12, 0, -106, 20, 23, -17, -32, 11, -8, 26, 37, 6, 12, 14, -66, -14, -68, 58, -63, 43, -25, -29, -17, 9, -25, -24, 14, -29, -13, -4, 71, 55, 13, -1, 75, -9, 4, 46},
  {4, 5, -63, -19, -123, 32, 7, -19, -18, -4, -19, 30, -9, -10, -14, -13, -74, -43, -11, -9, -10, -19, 7, -37, 42, -42, 1, 6, -24, -15, 17, 22, 48, 3, -16, 2, 14, -12, 6, 19, -26, 3, -80, -10, 3, -60, 17, 10, 12, 15, 31, -90, -4, -21, -32, 22, -30, -3, -59, -84, -73, -38, -34, 23},
  {8, 11, -37, -76, 70, 40, -8, 4, 19, 4, -5, 40, 21, -4, 24, 18, 18, -48, -14, -10, -1, 18, -12, -3, 14, -23, -5, 3, 51, -9, -11, -110, 66, 16, -16, 4, 5, 15, 7, 32, -83, 4, 41, -28, -14, -6, -13, 10, 10, 14, -2, -16, 26, -40, -27, 17, -69, -105, -23, -39, -19, -62, 6, 14},
  {-7, 52, -10, -39, 51, 16, 0, 22, -5, 24, -1, -28, 19, -4, 22, -8, -86, -32, 1, 6, -4, 9, 4, -41, 33, 30, 8, 17, -28, 37, -32, -111, 6, 36, 0, 14, 20, 14, -2, 41, 15, -28, -61, -8, -21, 44, -34, 16, 1, 10, 3, -44, 18, -21, -12, -5, -103, -97, -33, -68, -12, -37, 25, 38},
  {32, 35, 46, 20, 51, 15, -15, 12, -10, 30, 43, -21, -25, -1, 17, 6, -49, -127, -10, -43, 2, 23, -3, -14, -14, 0, 21, 31, -63, 16, -21, 16, -29, 21, 31, 16, -94, 13, -22, -32, -52, 10, -93, 29, -78, 24, -19, 5, 19, 3, -8, 26, 6, -13, 20, 32, -60, 40, -32, 26, -104, -2, 57, -15},
  {-13, -7, 16, 41, -59, 47, 44, 35, 41, 18, 29, 38, 21, 1, 29, -17, 61, -91, -32, -51, -4, -1, -21, -15, -6, 8, 49, 24, -82, -27, 1, 67, -4, 21, 12, 44, -44, 19, -24, -24, 59, 4, 45, -21, -77, 24, -25, 24, 16, -23, -27, -72, 31, -28, 8, 26, -3, -123, -17, -118, 14, -21, 23, -44},
  {-2, -31, 7, -46, 40, 1, -10, -18, 32, 4, -4, 14, -70, 34, 41, -28, -39, -56, 4, -17, 37, -28, -37, -17, 22, 37, 10, -20, -66, -29, -55, -106, 24, 6, 42, -7, -43, -48, 2, 4, -57, -39, -7, 7, -4, 10, 24, 59, 27, -16, 20, -65, -80, -22, 13, 59, 47, -36, -52, -66, -127, 9, -54, 8},
  {3, 3, -8, 43, -18, 19, 3, -13, 7, 14, -9, 0, 36, -25, -7, -12, -92, -60, -7, -28, -30, 0, 7, 5, -9, -14, -25, -27, -13, 52, -41, 29, 17, -10, 0, 37, 47, 27, 0, -18, -100, 3, -69, 17, -38, 15, 14, -6, 25, 3, 5, 52, -79, -7, 5, -13, 68, -43, -28, -112, -9, 3, 6, 9},
  {0, -2, -14, -15, -88, 13, -8, 12, -3, -12, -10, 31, 12, -14, -18, -4, 66, -48, 6, -8, -16, -14, 33, -42, 6, -33, -10, -3, 58, 7, -3, 65, 22, -23, -9, -2, 18, 19, 3, 1, -107, -21, -31, 28, -44, -36, 13, -18, 3, -1, 1, 7, 15, -21, -17, 15, -95, -90, -29, -25, 42, -30, -14, 2},
  {-28, -5, -8, -3, 76, 15, 8, -2, -25, -32, 2, 56, 47, -35, -11, 28, 29, -41, 15, -16, -26, -30, 20, -42, 19, -6, -3, 17, -115, -2, 17, 41, 13, 2, -7, 13, 13, 32, -3, 22, 27, -31, -37, -14, -20, -29, 0, -17, 1, -6, -41, -56, -32, 3, 4, 43, 48, 14, -42, -50, 1, -27, -12, 26},
  {1, 20, -34, -9, 86, -18, -13, -14, -15, -13, -23, -7, 20, -45, -21, 4, -124, -34, 22, 4, -50, 3, 44, 3, 22, -10, 2, 30, 2, 56, -1, 48, 38, 7, -14, -1, -63, 24, 3, 39, 51, -24, 64, 23, -86, -29, 0, -15, -9, 12, 1, -99, -49, 23, -1, 24, 66, -39, -39, -80, 43, -8, 3, 47},
  {17, 33, -41, -12, -122, 34, -30, -1, -16, 2, 1, -31, 2, -1, -5, -47, -67, -100, 28, -21, -19, 10, 49, -16, 21, -14, 24, 11, 17, 24, 35, 60, -17, 13, 14, -7, 82, 6, -22, 0, -101, -4, 55, 31, -99, -8, -25, -8, -39, -2, 32, -91, -74, 15, 20, 14, -71, 46, -45, -58, -126, 19, 26, -15},
  {26, 49, -18, -44, 29, 28, -40, 8, -22, 0, 18, 7, -14, -3, -7, -14, 16, 1, 13, -31, -5, 3, 41, 15, 22, -12, -8, -26, 40, 11, 64, 4, -38, -2, 1, -35, -79, -11, -3, 10, -83, -10, 62, 18, -42, -1, -17, -32, 10, 2, 31, 23, -119, -4, 6, -6, 22, 59, -10, -113, -43, 24, 17, -17},
  {15, -25, 4, -40, 8, -17, -49, -28, 4, -19, 12, -10, -44, 27, 10, -14, -113, -58, 19, -23, 25, -23, -6, -2, -10, 18, 10, -30, 66, -1, 0, 42, -22, -23, 47, -39, 56, 2, -1, -12, -28, -25, -4, 17, -26, 11, 21, 2, 3, -23, 5, -72, -73, -1, 17, 33, 37, -81, -16, -34, -47, 22, -24, 21},
  {-6, -26, -18, 45, 37, -15, -16, 14, 31, 2, -23, 26, -1, 14, -34, 16, -28, -21, -8, 5, -14, -31, -22, -29, 7, 18, -10, -10, 12, 7, -44, 56, 19, -36, -4, 11, -96, -6, 16, 26, -73, -45, 61, -11, -5, -13, 17, 29, 8, 16, -2, -5, 30, -32, -13, -23, -73, 37, -20, 86, 9, -20, -39, 32},
  {-5, -39, -12, 17, 44, -26, 18, 27, 0, 10, -5, -4, -5, -17, -7, 13, -94, -15, -8, 32, -15, -6, -11, -3, -14, -16, -27, -26, 54, 2, -2, 32, 25, -13, -11, 7, -54, 34, 2, 10, 93, 15, 76, 44, -20, 5, 27, -7, -6, -1, -17, -52, -70, 8, 6, 9, -36, 57, 2, 92, 18, -13, -7, 34},
  {-6, -11, 5, 3, 75, -68, 26, 21, -10, -5, -2, -6, 14, 9, -32, -1, -49, 0, -1, 53, -17, -31, -12, 0, -1, 23, 0, -1, -28, -1, 9, -49, 11, -28, -28, 4, -90, 29, 2, 19, -114, 2, 2, -2, -18, -39, 7, -21, -8, -7, -8, -52, -64, 49, 19, 25, 66, -99, -3, 2, 26, 21, 5, 28},
  {12, -1, -32, -60, -104, -22, 2, 5, 4, -30, -12, 8, 46, 10, -24, 18, -27, -75, 22, 62, -20, 5, 26, 7, 5, 19, -9, 20, 72, 17, 22, -60, -12, -28, -13, -7, -127, 27, 2, 43, 15, -24, 40, 35, -33, -4, 4, -27, -6, 3, -9, -12, -70, 26, -10, 25, -5, -9, -12, -67, -100, 11, 10, 51},
  {15, 4, -40, -36, -38, 19, -28, -6, 1, -23, -10, 4, -2, 2, -26, -32, 77, -48, 16, 18, 1, 0, 28, 25, 15, 4, -19, 18, 70, 7, 0, 19, -15, -33, -25, -35, -125, 8, 5, 26, -41, 9, -12, 41, -60, -11, -10, -25, -2, 17, 25, 21, -33, 13, -9, 10, 26, 46, -21, -4, -79, 24, 4, 19},
  {-8, 9, -21, -30, -111, -10, -2, 8, 10, -5, -6, 6, -5, -14, -21, -45, -81, -30, 4, 26, -4, -10, 16, 38, 24, 27, -41, -15, 70, 0, 1, -68, -3, -31, -45, -12, -123, 9, 1, 14, -49, -15, 58, 23, -51, 5, 5, -20, -8, 23, 15, -24, -94, -8, -2, -5, -103, -18, 2, 23, -95, 15, -2, 11},
  {8, -5, -2, -55, -15, -32, 4, -8, -24, -35, -7, -23, -5, 0, -13, 21, -65, -3, 1, 26, -28, -7, -1, 7, -1, -3, 2, 10, 0, 37, -19, -37, 16, -19, -14, 18, -112, 16, 2, 8, -57, -7, -73, 42, 44, -14, 30, -17, 16, 11, 6, -94, 69, 39, 15, -17, 41, -106, 20, 112, -14, 19, 15, 49},
  {7, -39, -11, -58, -68, -9, 10, 36, 13, -9, 8, -11, -3, -15, 2, 13, 78, -17, -6, 44, -5, -21, -12, -11, -25, 7, -21, 20, 45, 9, -1, -102, 22, -34, 13, 47, -15, 30, -13, -7, 75, -10, 126, 33, 58, 2, 26, 31, 8, 2, -50, 4, 78, 11, 8, -18, -67, 65, -11, 14, -70, 8, 2, 40},
  {-11, -11, 12, -47, -43, -37, -6, 9, -15, 10, -12, -3, -3, -16, -19, 37, 32, -2, 3, -5, -22, -24, -8, 6, 3, -7, -4, -16, -11, 9, 2, -8, 23, 13, 32, -11, -85, 18, -4, -14, -6, -30, -42, -7, 5, 7, 42, 11, -1, -10, 4, -36, 81, 19, 10, 2, 76, 78, 0, 79, 17, -1, -8, 26},
  {13, -21, 25, 28, 45, -26, -2, -1, 3, 24, -4, 4, -9, 6, 10, 7, -14, -7, 12, 8, 7, -11, -3, 27, -12, 0, -7, 0, 4, -7, -38, -73, 17, 0, 10, 0, 25, -14, -7, -15, 36, 40, 80, 12, -37, 7, 31, 23, 7, -10, -16, -15, 44, 22, 9, 11, -21, -84, -2, 2, -13, 14, -8, -10},
  {-33, -25, -26, -35, -38, 1, 23, -5, 41, 0, 12, -8, 20, 38, 22, -11, 8, -10, 2, 43, 2, 7, -26, 4, -4, 31, -11, 17, -13, -20, 13, -22, 16, -16, -11, 32, -46, -10, 4, 21, -107, 11, 60, 9, -36, 14, -29, 17, -5, -7, 17, -58, -14, -2, 7, 31, 80, -6, -8, -120, -14, 3, 13, -25},
  {15, -10, 4, -14, -125, -18, -3, -7, 3, -14, 10, -4, 5, -6, -1, -23, -98, -50, 17, 48, 5, 9, -16, 17, -8, 32, -20, -39, -80, 4, 5, -121, -16, -28, 0, 11, -54, 24, 15, 29, 81, -28, 54, 29, -38, 34, -11, 3, -3, 26, 11, -57, -57, -6, -2, -12, 71, -12, 6, -47, -125, -5, 11, 10},
  {12, -22, 2, 8, 74, 13, 3, 1, -1, 19, 12, 15, -32, -2, 5, -2, 36, 29, -21, 29, 10, 4, 17, 8, -25, -3, -17, 16, 16, -4, 4, -28, 9, -6, -17, 4, -116, -9, 3, 13, -108, 33, -115, -6, -73, 3, -3, -17, -27, 20, -4, 44, -84, 25, -23, 3, -114, 38, 15, -80, -24, 2, -1, -15},
  {24, 5, 36, -47, -73, -20, 2, 20, -9, -8, 1, 12, 1, -37, 8, 20, -57, 48, 16, 0, 10, -9, -38, -30, -18, -15, -5, 14, -38, -14, -2, 74, 5, 56, 20, 34, -20, -11, 11, -4, -90, -20, -42, 10, -29, 29, -3, 15, -12, -9, -23, 32, 78, -30, -15, -2, 8, 23, -37, 61, 25, -13, -15, 0},
  {-4, -18, 4, 19, 3, 1, 22, 19, -5, 6, 20, 15, 27, 2, 13, -2, 55, -12, 17, -10, 14, -13, -43, 6, -8, -8, 28, -21, -90, -24, 8, 45, -22, 0, 23, 11, -55, -8, -4, -24, -5, -23, -37, 8, -37, 27, -10, 10, -17, -6, 0, 66, -31, 6, 23, -19, 12, -21, -10, 105, 19, 3, 7, -18},
  {-23, -9, 31, -26, -101, -38, 48, 21, 74, 5, -8, -14, 18, 40, -18, 23, -115, 11, 46, 50, -45, -36, -14, -24, 11, 76, -46, -27, 1, -14, 34, 84, -6, -24, 22, 19, 6, -11, -32, 49, 36, -6, 19, 43, -37, 25, -33, 69, 3, -11, 10, 81, -45, -17, 37, 40, -47, 95, -35, 37, 29, 12, 49, 46},
  {-69, -48, 2, 37, -123, -47, 3, -64, 56, -48, -39, -50, 21, 89, -32, 51, 67, 27, 52, 61, -21, -67, -27, 4, -4, 100, 6, 33, -6, 8, 26, 73, 36, -41, 12, -20, -57, -42, -79, -14, 49, -59, -56, 21, 41, -68, 24, 77, -1, -46, 48, -37, 64, 12, 65, 66, 116, 59, -1, -78, -76, 58, -26, 49},
  {-90, -65, -22, 15, -66, -57, -17, -127, 24, -36, -57, -86, 53, 59, -18, 48, -21, 18, 86, 13, -74, -61, 19, 61, -16, 52, 14, -5, 25, 48, 6, 55, 73, -87, 27, -89, -76, 41, -99, -9, 68, -80, -24, 25, 26, -30, 65, 34, -6, -90, 127, -55, -49, 89, 74, 63, -55, -65, -12, 2, 8, 93, 17, 75},
  {-64, -32, -19, 17, 82, -4, 15, -90, 65, -4, -88, -119, 96, 48, -22, -44, 104, -51, 85, -8, -128, -24, 92, 127, 14, 30, -65, 11, -65, 72, 59, 21, 55, -41, 63, -62, 69, 96, -128, -17, -78, -29, 35, 38, 41, 40, 69, 87, 5, -28, 100, -68, -101, 78, 127, 50, 39, 10, -16, -39, -27, 70, 49, 78},
  {-34, 45, -108, -16, 52, 71, 15, -40, -8, -41, -57, -64, 77, 0, -36, -69, 13, 30, 82, -11, -94, 73, 72, -1, 71, -9, -44, 8, -33, 53, 21, -44, 37, -10, -3, -91, -44, 85, -25, 69, 76, -63, -80, -36, 66, 35, -40, 18, -34, 13, 109, 74, -97, 23, -8, 59, 51, 66, 108, -34, 25, -9, 61, 50},
  {-44, 103, -58, -16, 78, 61, 17, -51, -12, -47, -62, -53, 122, 15, 2, -56, 48, -56, 60, -13, -83, 125, 95, -33, 84, -40, -32, 49, 54, 127, 68, 25, 51, 3, -126, -6, 33, 99, 11, 108, -92, -53, -92, -98, 2, 45, -100, -35, 1, 28, 82, 9, -88, 6, -48, -5, -20, 72, 127, 4, -68, -34, 115, -29},
  {-41, 84, -42, 30, -105, 68, 28, -52, -16, -23, -105, -27, 112, -35, -40, -27, -78, 21, 127, -30, -97, 117, 121, -38, 94, -65, -42, 79, -90, 112, -39, 118, 70, -13, -122, -21, 21, 122, 5, 127, -109, 17, -101, -88, 43, -9, -62, -31, 41, 106, 44, 67, -15, 56, -71, 38, -90, -68, 46, -74, 70, -40, 38, 9},
  {35, 16, 41, -47, 10, 1, 5, 4, -11, -9, 1, 29, -32, -6, -18, -19, -40, -37, -3, -9, 7, -1, 13, 29, -17, -4, -46, 59, -64, 21, -25, -101, -7, 3, -9, -23, -34, -60, 12, 7, 66, -19, 22, 1, 2, -23, 9, -10, 22, 20, 15, 60, -98, -39, 10, 15, 35, -99, -6, 73, -84, 18, 16, 15},
  {-8, -29, 17, -101, -66, 8, -8, -17, -7, 12, 9, 18, 1, 10, 3, 15, -3, -37, 62, -3, 10, -24, -13, 56, -34, -19, 27, 29, 55, 22, 19, -95, 20, -68, 21, -84, -118, -63, -34, -25, 69, 0, 90, -31, 13, -19, 34, -7, -7, -9, 65, -70, 81, 7, 26, 11, -109, -75, -3, -57, 4, 57, -26, -8},
  {-39, -23, 33, -7, -106, 26, 31, -69, 21, 12, 1, -17, 28, 16, 17, -21, -23, -17, 31, 26, -13, -14, -28, 37, 0, -2, -39, -7, -87, -17, -24, 11, 11, -15, 31, 4, -4, -23, -66, -38, 13, 26, 110, -62, 34, 10, 8, 33, -11, -27, -19, -7, 28, 49, 34, 28, 65, -73, -10, -7, 31, 40, 28, -38},
  {-65, -35, -33, -55, -17, 19, 68, -45, 57, -24, -8, 13, 85, 27, 26, -7, -30, -40, 50, 50, -36, 24, -17, 12, 26, 4, -35, -34, 42, -36, 30, 34, 3, -64, -22, 38, 75, 50, -63, -6, 59, 19, 82, -54, 45, 10, 4, 17, -4, -36, -90, -66, 64, 57, 16, 31, -72, 52, 47, 19, 12, 30, 45, -43},
  {-26, 9, -58, 8, -128, 18, 13, -12, 44, -23, 13, -24, 89, 58, 24, -68, -40, -8, 35, 10, -34, 49, -23, -20, 14, 36, -3, 0, -75, -36, 79, 65, -15, -79, -46, 42, -70, 53, -12, 20, 7, -23, -19, -35, -15, 32, -64, -7, -28, -20, -73, 32, 9, -7, 23, 40, 55, 56, 79, 71, -94, -15, 92, -34},
  {2, 6, -58, 48, 23, 9, -8, 21, 6, -17, 25, -37, -46, 28, 21, -88, 69, 22, -10, -41, 21, 41, 7, 3, -24, 24, 28, 52, -29, -18, 55, -31, -21, 6, -41, 7, -87, -13, 21, 30, 71, -15, -37, 20, 50, 29, -50, -54, -25, 9, -46, 78, -12, -8, -14, 24, 33, 75, 29, -23, -32, -6, 17, -35},
  {30, 23, 3, 44, -80, 12, -38, 20, -5, -32, 11, -30, -40, -15, 22, -33, -16, 29, 15, -16, 32, 40, 36, 37, -26, -5, 30, 38, -31, -1, -28, 64, -6, 26, 3, -14, -12, 19, 19, 24, -51, -45, -25, 49, 95, 18, -6, -34, -9, 26, -57, 15, -66, 0, -47, -31, -61, -59, -12, 67, -1, -38, -26, 23},
  {21, -5, 22, -61, -8, -36, 21, -10, 20, -30, 16, 47, -38, 28, 4, -71, 39, -27, 37, -17, 33, -22, -22, 27, -4, 2, 22, -1, 40, 7, -21, -120, -21, -8, -8, -16, -17, 28, -22, -4, 13, -4, 101, 26, -31, 17, 25, 19, -10, 2, 57, 35, 36, -33, 19, -54, -25, 84, -25, 32, 19, -9, -7, 36},
  {-18, -47, 34, -42, -35, -34, -2, -44, 24, -10, 7, 19, -33, 46, 7, -24, 58, -15, 53, 7, 15, -23, 17, 51, -18, 21, -3, -12, -52, 2, -36, -85, -40, 6, 31, -50, 5, -40, -32, -25, -103, 21, -3, -20, 32, -30, 33, 15, 5, -17, 17, 13, 28, 52, 43, -33, -128, 118, -13, -100, 17, 47, -10, 18},
  {-15, -35, -3, -22, -2, -27, -17, -36, 37, -11, -2, -7, -30, 64, 3, -28, -13, -31, 22, -2, 2, -24, 3, 81, -15, 21, -3, -23, -69, -7, -23, -59, -13, -16, 30, -36, -1, -16, -25, -43, -110, 52, 61, -7, -77, -16, 28, 29, -14, -14, 8, -9, 38, 51, 67, -26, -17, -39, 9, -29, 7, 74, 23, 15},
  {-45, -43, -75, -31, -105, 6, 36, 0, 55, -45, 5, 44, 24, 81, 28, -24, 77, -63, 37, 45, 8, 0, -58, 41, 2, 48, -18, 7, 2, -15, 7, 18, -17, -51, -5, -10, 82, 4, -21, -21, 57, 17, 51, -41, -33, -4, 6, 25, -48, -8, 3, -43, 38, 34, 36, 34, 62, -81, 62, -34, -2, 46, 15, 18},
  {-30, -3, -41, 7, -65, 31, -10, 16, 58, -61, 8, 9, 16, 67, 44, -51, 12, 34, 47, 48, 7, 75, -27, 6, -16, 43, -20, 58, -45, 14, -12, -119, -43, -53, -59, 15, 99, 32, -32, 5, 52, -6, -62, 3, -80, 77, -64, -2, -32, 4, -16, -43, -22, 35, 23, 49, -99, -24, 68, -63, -118, -4, 49, -2},
  {-3, -42, 53, -42, -112, 59, -51, 15, 85, -15, 11, -3, -10, 37, 48, 3, 79, 74, 6, 19, 57, 73, -14, 5, -49, 37, -3, 53, -11, -3, 5, -107, -54, -48, -54, -22, 6, 24, 6, 20, 15, 45, -83, 24, -9, 116, -45, 1, 10, 24, -48, -62, -72, -36, -8, 41, -126, -112, 8, -86, 38, -11, 26, -1},
  {-13, -23, -10, -58, 53, -7, -29, 3, 24, -22, 10, 4, -80, 17, 10, 31, 71, 31, -11, 17, 37, -10, -23, 2, -17, 28, 2, 13, 21, -4, -29, 16, -31, -16, 2, -27, 20, -10, 15, 16, -40, 8, 85, 0, -70, 33, -3, -17, -14, 10, -23, -13, 19, -2, 18, -10, -3, -31, 3, 38, -75, 12, 9, 59},
  {21, 14, -49, -94, -103, -1, -41, -3, 14, -18, 15, 24, -14, 16, 12, -2, -48, -13, 6, -58, 22, 9, -25, -19, -2, -8, -7, -6, -61, 22, 16, -115, -40, -2, 14, -10, -31, 29, 0, -13, 25, 44, -70, -1, -62, 12, -25, 16, -22, -10, 15, -51, -69, -29, 2, 4, -119, 48, -33, -78, 26, -45, -20, 4},
  {19, -54, -2, -76, -8, -77, -28, -6, 6, -24, 49, 13, 2, 30, 18, -73, 45, 12, 28, -5, 32, 0, 3, 16, -66, 30, -7, 0, -45, -2, 40, -86, -75, -4, 38, 7, -10, 12, -5, -4, 48, 16, -59, 83, 0, 73, 13, -12, -33, 18, -9, -35, 57, 22, 22, 0, -21, -116, 61, -71, 1, 0, -2, -30},
  {28, -48, 38, 7, -8, -22, -19, 26, -18, -28, 35, 42, 15, 18, 50, -95, 74, 25, 33, -47, 20, -3, -4, -38, -63, 6, 33, -7, 25, -48, 50, -21, -30, 9, 55, 11, -6, 2, -10, -35, -120, -17, 43, 60, 0, 44, 30, -11, -37, -19, -22, 24, -3, 36, 30, 1, 78, -57, -4, -52, -7, 18, 0, -60},
  {-5, 9, 31, -34, -123, -11, 0, 22, 7, -28, 19, 40, -2, 9, 48, -53, -20, 24, 31, -59, 6, 9, -10, -79, -1, 24, 57, 8, -4, -61, 66, 71, -24, 43, 45, 9, 11, 18, -25, -18, -15, -56, -45, 1, -15, 33, -25, 44, -4, -24, -37, 40, -44, -16, 23, 32, -79, -108, -24, 49, -4, -24, 42, -48},
  {-3, 28, 52, -28, -19, -32, -53, -28, 53, 8, -17, -48, -12, 27, 44, 17, -49, -47, 42, 12, 11, 62, 76, -5, -4, -1, 41, -38, -2, 6, 98, -10, -24, 22, 29, -6, 35, 52, -17, -2, 22, -29, -67, 42, 18, 98, -22, 64, 28, 20, -7, -91, 6, -79, 12, -6, -25, -79, -25, -85, -20, -35, 43, -11},
  {34, -42, 13, -5, -13, -18, -32, 13, 15, 21, -27, -12, -2, -19, 13, -26, -29, -33, -2, -10, 3, 19, 14, 50, -18, -11, 20, -27, -10, 7, -46, -55, 11, 0, 12, -7, 100, 9, 32, -14, -90, -11, 73, 30, -44, 44, 27, 5, 15, 26, -24, -66, -63, -5, -4, 6, -39, -105, -69, -94, -41, -12, -57, 7},
  {23, -43, -79, 33, 4, -20, 24, 9, -29, 5, 6, 32, 82, -28, 14, -35, 48, -45, -59, -74, -18, 45, -1, 15, -60, -58, 12, -14, -49, 34, -46, -1, -43, -49, 3, 116, -45, 16, 2, -24, -69, 26, 18, 10, -67, 2, 16, -66, -11, 14, -39, -78, -15, 7, 9, -33, 105, -14, -36, -4, -16, -26, -56, -2},
  {27, -9, -25, -66, -117, 9, -22, 26, 13, 3, 1, 32, -11, -41, -2, -37, 13, -12, -15, -11, 13, 33, 2, -69, 16, -20, -23, 26, 110, 8, 48, -45, -33, 5, 1, -35, -18, 48, 21, 37, 66, -7, -67, 40, -42, -41, -7, -20, 18, 18, -16, 57, 103, -11, -21, 18, 27, 35, -71, 46, 13, -44, -24, 24},
  {21, -14, -55, -39, 78, -1, 2, 30, 31, -23, 28, 36, -7, 21, 12, -52, 35, -32, -31, 2, 27, -3, -36, -10, 19, 2, 10, 15, 69, -33, 13, 61, -13, -10, 8, -21, 80, 13, 2, 31, 19, 28, -116, 7, -75, -31, -7, -4, -10, -8, -8, 20, 108, -14, 1, 4, -79, 54, 23, -13, 58, -4, -11, -4},
  {-4, 11, -39, -51, 46, 4, 4, 36, 11, -9, 46, 8, -2, 34, 39, -61, 22, -79, -7, 8, 19, -29, -37, -40, 15, 33, 62, -4, -40, -34, 28, 54, -43, 20, 55, 9, -91, 21, -5, 27, -107, -24, -18, 22, -84, -10, -1, 28, -43, -59, -13, -16, -115, -3, 19, -10, 11, -111, -18, -89, -48, -3, 14, 7},
  {-26, 48, 4, -100, -85, 15, -34, 44, -5, 1, 13, -33, 8, 15, 22, -30, 51, -45, 51, -26, -2, -9, 35, -30, 44, 20, 53, -9, -105, 12, 107, -72, -45, 34, 62, -8, -31, 54, -10, 29, -96, -49, -34, 32, -57, 33, -5, 15, -19, -49, -6, -39, -5, -2, 21, -2, 73, 49, -37, 28, -41, 14, 18, 44},
  {19, 8, -19, -52, 15, 25, -80, -9, -29, -20, -7, -23, -2, 26, 1, 39, 53, -123, 49, -95, 11, -17, 50, 11, 18, -21, 36, -2, -85, 9, 34, 3, -21, 27, 36, -64, -52, -2, 6, 21, -23, -60, 60, -15, -32, -15, 28, -4, -40, -37, 5, 59, 20, 17, 1, 9, -34, 61, -24, 19, -72, 24, -36, 33},
  {29, -33, -29, -47, 78, 20, -34, -12, 3, -4, -35, 31, 24, 24, -8, 28, -67, -71, 35, -67, 7, -25, 33, 51, -11, -44, 15, -5, -52, 4, 1, -24, 27, -58, 1, -75, 85, -12, 8, 17, 22, -14, -69, -45, -37, -71, 62, -7, -27, -12, 32, -86, 73, -1, 2, 2, 71, 27, -44, -48, -105, 14, -67, 39},
  {-11, -3, 5, -27, -123, -41, -4, 20, -46, 13, -27, 17, 37, 3, -56, 30, 55, -27, -12, 5, -22, -49, -2, -31, -5, -10, -6, 4, -119, 46, -41, 51, -10, -24, 8, -40, 50, 29, 0, 30, -23, -18, 66, -26, -5, -99, 50, -13, 18, -50, 45, -68, 9, -19, 20, -63, 6, 89, 15, -67, -29, 23, -28, 50},
  {13, 12, 12, -24, -15, 10, -23, -9, -11, 13, 5, 1, 1, 9, -42, -27, -33, -47, -11, 7, -16, 9, 49, -1, -12, 5, -27, 10, -105, 9, 23, -29, 2, 6, -18, 15, 61, 37, 10, -2, 70, 30, 100, 21, -14, 16, 12, -19, -6, 22, -39, 9, -70, -3, 2, -2, -2, 58, 18, -33, 31, -17, 17, -2},
  {-6, 17, -3, 49, 50, -33, 10, 21, -20, -5, 39, 6, -27, 1, -5, 21, 36, -29, -44, -37, 26, -4, -31, -42, -17, 24, -18, -15, -36, -5, -2, -18, -31, -38, 5, 20, -6, 16, 10, -23, -66, -9, -10, -5, -15, 14, -11, 2, -4, -6, -99, -48, -9, 6, 10, 2, -49, 1, 10, -2, -124, -16, 27, -14},
  {-38, 20, -21, -56, -31, -28, 14, 15, -21, -11, 0, -4, -3, -6, -25, 8, -52, -2, -24, -34, -18, -17, -20, -26, 16, 34, -35, -32, 39, 13, -16, -63, -16, -40, -21, 15, -90, 0, 4, -3, -121, -10, 4, -10, 27, 3, -12, -14, 10, 6, -36, 0, -120, 14, 21, 1, -22, 14, 13, -8, -95, -1, 27, 41},
  {-33, 6, -4, -9, -112, -36, -2, -13, 2, -10, -5, -7, 7, -7, -33, -13, -90, 30, -14, -16, -10, 11, -5, -5, 12, 31, -42, -27, 7, 12, 20, 62, -19, -18, -28, -1, -63, -5, 2, 15, -7, 2, 46, -20, -32, -3, -6, -10, 17, 3, -2, 21, -82, 17, 19, 8, 11, -25, 12, 22, 16, 17, 28, 40},
  {-24, 22, 5, -95, -70, -12, 9, -9, 12, -24, -11, -35, 29, -11, -39, 13, -23, 11, 21, 6, -25, 17, 9, -9, 15, -11, -41, -51, -22, 26, -9, 25, 15, -9, -34, 24, -79, 23, 4, -2, -14, 22, -89, 2, 30, 2, 20, -10, 7, 10, 22, -15, -31, -1, 2, -30, -10, -38, -13, 53, -11, -4, 6, 26},
  {20, 4, -1, -29, 70, -12, -37, -6, -24, -39, -7, -18, 12, -47, -30, 10, -76, -32, -10, 14, -4, -2, 5, 1, -7, -48, -62, -5, 65, 20, -36, -95, 36, -11, -10, 28, -105, 7, 7, -11, -54, 48, -105, 42, -30, -1, 37, 3, 34, 2, -3, 42, -97, 20, 2, -76, -23, -118, -15, 77, -17, 12, 0, 11},
  {3, 7, -41, -61, -7, -10, 8, 39, -31, -29, 15, 6, 39, -38, 12, 0, 74, -16, 6, 43, -12, 7, -27, -40, -15, -25, -23, 37, -119, 25, 22, -86, 19, -4, 10, 52, -78, 58, 4, -5, -4, 7, 44, 30, -1, 12, -5, -21, -7, -21, -41, -63, 26, 29, 2, -37, 42, 75, 23, 109, -53, 3, 13, 16},
  {-23, -33, 1, -19, 63, -8, -38, 36, 43, 63, 18, -37, -16, 3, 34, 57, -90, -9, 12, 28, 7, 2, 2, -24, -8, 6, -11, 12, -84, -70, -26, -55, -10, -3, 19, -10, 25, -10, 3, -4, 72, -31, 4, -22, -6, 58, -29, 35, 1, 2, -6, 32, 89, -44, -32, -9, 80, -47, -25, 0, -3, -13, -16, -26},
  {-27, 10, 10, 56, 19, -27, 31, 28, -1, 22, 2, -4, 0, 17, 12, 40, 15, 26, -22, 33, -13, 0, -19, -19, -2, 25, 10, 9, -109, -26, 26, -16, 4, -15, 12, 37, 42, 17, 14, 3, 45, 2, 18, -17, -52, 32, -10, 3, -1, -3, -13, 0, -77, 3, -8, 16, -52, -32, -36, 26, -110, 6, 8, 23},
  {-6, 7, 17, 4, -105, -24, 22, -4, -11, 12, 1, 4, -7, -6, -9, 27, 24, -12, -8, 54, -5, -12, 4, 0, -3, 33, -21, 8, 17, 18, 7, 17, -23, -29, 28, 19, 44, 27, 4, -1, -82, -29, 10, -32, -29, 1, 8, -3, 8, 14, 18, -50, -126, 25, -1, 22, -4, 57, -29, 35, 26, 18, 0, 15},
  {23, 22, 12, -20, 85, -23, 21, 9, -19, -12, -9, 21, -14, -7, -12, 31, -26, -39, 20, 50, 8, 10, 19, 32, 0, -2, -19, -2, -84, -5, 4, -24, -16, 6, 6, -6, -35, 25, 0, 15, -36, -31, 22, -25, -23, 27, 3, -6, 10, 5, 19, 50, 43, -11, -16, 0, -88, -64, -17, 43, 32, 1, 7, -9},
  {21, 34, -12, -34, -71, -17, 26, 13, -20, -5, -7, 14, -25, -11, -31, 27, -108, 18, 29, 68, -13, -1, 13, 13, 15, -1, -12, -21, -27, 4, -2, 16, -25, 1, -8, -23, -119, 2, 10, 5, -48, -38, -46, 10, -31, 19, -7, 5, -30, -3, -10, 34, -66, -6, -32, -7, 1, -46, 1, -16, -49, -23, 7, -9},
  {27, 17, 19, -67, -43, -12, 2, -7, -13, -23, -13, -17, 18, -16, -16, 24, -18, -8, 32, 35, -2, 1, -8, 21, 16, -19, -1, -52, 45, 10, -4, 67, 19, 9, 16, -6, -7, 20, 4, -14, 52, -9, 47, 8, -79, 23, 14, 11, -26, 3, -8, -58, -32, -38, -3, -14, 30, -4, -9, 35, 11, -29, 6, 6},
  {9, -7, -1, -22, 92, 9, 23, 11, -12, -2, 34, -1, 8, -10, 28, 61, -30, -63, 6, -12, 22, 17, -37, 16, 5, -44, 30, -11, 11, 5, 0, -41, 3, 4, 33, 32, 102, 28, -6, -16, -31, -34, 89, 27, -9, 55, -16, 13, 22, 11, 10, -78, -93, -30, 28, 5, -87, 52, -24, 110, 34, -24, 34, 13},
};

    float fc_requant_mult1[64] = {0.000438352f, 0.000421153f, 0.000383174f, 0.000073634f, 0.000036215f, 0.000359181f, 0.000335009f, 0.000441128f, 0.000384555f, 0.000465078f, 0.000594315f, 0.000427911f, 0.000305164f, 0.000373321f, 0.000504366f, 0.00027054f, 0.000037975f, 0.000179232f, 0.000432447f, 0.000391367f, 0.000503953f, 0.000425419f, 0.000417308f, 0.000399135f, 0.000482278f, 0.000379702f, 0.000455956f, 0.000488105f, 0.000038154f, 0.000421938f, 0.000274664f, 0.000038327f, 0.000367679f, 0.000365257f, 0.000397796f, 0.000337327f, 0.000037158215641561986f, 0.00031798274519537504f, 0.0006585022370188498f, 0.0004341399574553242f, 0.000037202041581138f, 0.0002861715845269202f, 0.00003995327179721882f, 0.00031624516686168657f, 0.00008866873127969523f, 0.00032958458922875455f, 0.00044385265024776173f, 0.00036416602673436176f, 0.00038471327894709385f, 0.0005268627284791369f, 0.00041508001923409816f, 0.00004526421109910374f, 0.00003603342943510234f, 0.00042886543707370975f, 0.000472995787077283f, 0.00038363546730862583f, 0.00003829327682907055f, 0.000038191981968802076f, 0.0003370749439536464f, 0.00004826316675310157f, 0.00004636815054652473f, 0.0004564631135197546f, 0.0003812879841657741f, 0.000352099011565219f};


// uint8_t *fc1_output = linear_layer_q(
//     flattened_output,
//     fc1_in,
//     fc1_out,
//     fc1_weights,
//     fc1_bias,
//     fc_requant_mult1,
//     output_zero_pointfc1
// );
    for (int n = 0; n < 64; n++) {
        int32_t sum = 0;
        for (int i = 0; i < 784; i++) {
            sum += (int32_t)FC1[i][n] * ((int32_t)flat[i] - 0);
        }

        // Requantize
        float yf= sum * fc_requant_mult1[n];
        int32_t y = (int32_t)(yf + (yf >= 0 ? 0.5f : -0.5f));
        // round half to even
        // int32_t y;
        // int32_t fx = (int32_t)x; 
        // float diff = x - (float)fx;

        // if (diff < 0.5f) {
        //     y = fx;
        // }
        // else if (diff > 0.5f) {
        //     y = fx + 1;
        // }
        // else { // diff == 0.5
        //     if ((fx & 1) == 0)
        //         y = fx;     // even
        //     else
        //         y = fx + 1; // odd → up
        // }

        // add output zero point
        y += output_zero_pointfc1;

        // ReLU 
        if (y < output_zero_pointfc1)
            y = output_zero_pointfc1;

        // clamp to uint8
        if (y < 0)   y = 0;
        if (y > 255) y = 255;

        fc1_output[n] = (uint8_t)y;
    }
    printf(" FC1 Output (size=%d):\n", fc1_out);
    for(int i=0;i<fc1_out;i++){
        printf("%4d ", fc1_output[i]);
    }
    printf("\n");

    //-----------------------------------------------------------------------------//
    //
    // --------------------------- FULLY CONNECTED --------------------------------//
    //                             LAYER 5 64->10
    // ----------------------------------------------------------------------------//
    
    int fc2_in = fc1_out; // 64
    int fc2_out = 10;
    int32_t* fc2_bias = (int32_t*)malloc(fc2_out*sizeof(int32_t));

    // Initialize weights/bias
     for(int i=0;i<fc2_out;i++) fc2_bias[i] = 0;
    // int* fc2_output = linear_layer(fc1_output, fc2_in, fc2_out, fc2_weights, fc2_bias);
    int8_t* fc2_output = (int8_t*)malloc(10 * sizeof(int8_t));
     float fc_requant_mult2[10] = {
      0.004487273878639151, 0.004021438359785049, 0.0046898233097664044, 0.0038485844794009846, 0.005124463586200433, 0.0041220704115096, 0.004204309689617205, 0.0036782430250047278, 0.005489125006414414, 0.004708096199389385};
    int output_zero_pointfc2 =73;
    int8_t FC2[64][10] = {
        {-91, 87, 12, -128, 57, -128, -19, 14, 48, -1},
        {-36, 34, -41, 13, 3, -24, 59, -128, -3, 8},
        {-81, 8, -40, 57, 43, -36, -119, -66, 50, 36},
        {2, -32, -26, -8, 6, 19, -35, -21, 6, 6},
        {-7, 32, 23, 23, -18, -28, -13, -39, 12, 29},
        {-3, -35, -53, -34, 16, -47, 58, 60, 17, 47},
        {36, -19, 37, -19, -114, -8, -37, -3, -17, 34},
        {19, -69, 44, -84, 21, 29, -7, -77, 6, 41},
        {23, -87, -38, 13, -51, 38, -10, 26, 23, -9},
        {46, 63, -128, -27, 13, 0, -54, 3, 17, 15},
        {-128, 52, -7, 35, 42, 47, -100, 44, -23, 37},
        {34, -72, 66, -93, 39, -2, -92, 76, -31, 38},
        {-2, -127, 35, -49, -44, 11, 36, -9, 24, 39},
        {10, 43, -9, 42, -69, 36, -67, 54, -75, -23},
        {-95, -118, -39, -19, 12, 13, -103, 58, -7, 40},
        {6, -27, 27, -61, 24, -29, -42, -23, 16, 32},
        {20, -16, 20, -29, -16, -25, -10, 24, 0, -32},
        {-77, 4, 9, 0, -4, 21, -4, 0, -11, 12},
        {-5, -93, 38, 60, 5, 16, 51, 31, 22, 7},
        {35, -55, 27, 22, -128, 31, 14, -45, -2, -61},
        {-53, -5, -66, 20, 56, 21, -76, 68, -22, -16},
        {-45, -23, -83, -109, -3, 40, 56, 2, 31, -2},
        {-49, 18, 32, -14, -28, -51, 54, 12, 23, -128},
        {12, -18, 36, -30, -20, -75, -66, 62, 46, -104},
        {36, -72, -79, 30, -1, -45, 37, -82, -50, 23},
        {31, -15, -11, 62, -83, 56, -33, -55, -43, 4},
        {8, 27, 23, 7, 61, 26, 2, 53, -127, 22},
        {-38, 87, -83, -100, -10, -12, 8, 31, -44, 67},
        {6, -21, -3, 6, 27, -29, -31, 30, -2, 27},
        {-16, 51, 47, -24, -13, 1, 29, -87, 30, -41},
        {-109, -34, 11, 25, 14, 42, 25, 20, -19, -45},
        {-25, -12, -8, -29, 17, -19, 25, -23, -4, 4},
        {61, 23, 15, -64, -19, -84, 12, 65, 22, 19},
        {55, 2, -81, 10, 25, -98, -30, -40, 22, 40},
        {-36, -67, 21, 62, 19, -60, -128, -9, 10, 34},
        {17, 29, -7, -58, -39, 18, -26, -79, -1, 53},
        {0, -26, -10, -31, -18, 7, -27, -6, 5, 30},
        {-2, -86, 35, -4, -4, 20, 33, -93, 30, 20},
        {43, 17, -43, -100, 67, 24, 27, 1, -16, -32},
        {46, -81, -6, -24, 24, 46, 49, -59, -12, -13},
        {-11, 10, 23, 34, 26, 17, -35, 19, 9, -29},
        {-32, 33, 27, -57, -42, -26, -95, 50, 9, 5},
        {12, -21, 25, 36, -29, 10, -11, 21, 14, -23},
        {-73, -11, 58, -15, 19, 36, -4, -114, 26, -67},
        {3, -48, -15, -40, -8, -4, 10, -16, -1, -10},
        {-97, -103, -106, 18, 29, 58, -3, -92, 24, 18},
        {20, -20, 68, -38, 16, -107, -75, 20, 27, -35},
        {41, -59, -38, 57, -27, -11, -19, -14, 21, 23},
        {33, -42, -52, 20, -54, -19, -6, -101, 28, -24},
        {37, 56, -28, -95, 6, 12, 31, 6, 29, -69},
        {5, -5, 34, 26, 1, -125, 35, 30, -56, -56},
        {30, -8, 2, -13, 25, -34, 16, -17, -13, 19},
        {18, -27, 4, 3, -22, -31, 10, -1, 7, -4},
        {35, 80, 51, 4, -34, -97, -3, 61, -56, -103},
        {-81, 38, 33, 66, -45, -10, -99, 12, 7, 19},
        {50, 23, -67, 65, -117, 7, -34, 29, -17, 8},
        {-31, 31, -32, -15, 7, 24, 12, 8, -11, 21},
        {-18, -15, 4, 12, 24, -33, -21, 37, 6, 23},
        {-50, 70, 14, 4, -70, 72, 24, 13, -56, -23},
        {-28, -25, 16, 20, 16, -6, 25, 0, 21, -28},
        {22, 19, -28, 1, 3, -6, 5, -24, 23, -27},
        {-75, 49, 31, 26, -19, -65, -79, 73, -48, -47},
        {-53, 44, -32, 47, -74, 20, 21, -60, 14, 2},
        {42, -49, 21, 9, 16, 9, 13, -111, 8, -43},
        };
        // uint8_t *fc2_output = linear_layer_q(
        //     fc1_output,
        //     fc2_in,
        //     fc2_out,
        //     fc2_weights,
        //     fc2_bias,
        //     fc_requant_mult2,
        //     output_zero_pointfc2
        // );
         for (int n = 0; n < 10; n++) {
            int32_t sum = 0;
            for (int i = 0; i < 64; i++) {
                sum += (int32_t)FC2[i][n] * ((int32_t)fc1_output[i] - 70);
            }
            // Requantize
            float yf= sum * fc_requant_mult2[n];
            int32_t y = (int32_t)(yf + (yf >= 0 ? 0.5f : -0.5f));
            
            // add output zero point
            y += output_zero_pointfc2;
            // clamp to uint8
            if (y < 0)   y = 0;
            if (y > 255) y = 255;

            fc2_output[n] = (uint8_t)y;
        }
    
    printf("Final FC Output (size=%d):\n", fc2_out);
    for(int i=0;i<fc2_out;i++){
        printf("%4d ", fc2_output[i]);
    }
    printf("\n");

    // int pred = argmax(fc2_output, 10);
    // printf("Predicted class = %d\n", pred);

    // int prob[10];
    // softmax_fixed(fc2_output, prob, 10);

    // for (int i = 0; i < 10; i++)
    //     printf("Class %d: %d%%\n", i, prob[i]);

   int *base_conv = Base_conv(inside, c_in1, c_out1, input_matrix, filter1,stride1, padding1);
    //int *base_conv1 = Base_conv(oside1, c_in2, c_out2, out1_i8, filter2, stride2, padding2);


    printf("Convolution Output Matrices (c_out1=%d, oside1=%d):\n", c_out1, conv_out1);

    for (int oc = 0; oc < 1; oc++) {
        printf("Channel %d:\n", oc);
        for (int i = 0; i < conv_out1; i++) {
            for (int j = 0; j < conv_out1; j++) {
                int idx = oc * (conv_out1 * conv_out1) + i * conv_out1 + j;
                printf("%4d ", (int)base_conv[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }
    //printf("Convolution Output Matrices (c_out1=%d, oside1=%d):\n", c_out1, conv_out1);

    // for (int oc = 0; oc < 1; oc++) {
    //     printf("Channel %d:\n", oc);
    //     for (int i = 0; i < conv_out2; i++) {
    //         for (int j = 0; j < conv_out2; j++) {
    //             int idx = oc * (conv_out2 * conv_out2) + i * conv_out2 + j;
    //             printf("%6d", base_conv1[idx]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // free memory
    free(input_matrix);
    free(filter1);
    free(filter2);
    free(out1);
    free(out2);
    // free(flat_out2);
    // Free memory
    
    free(fc1_output);
    free(fc2_output);
    return 0;
}