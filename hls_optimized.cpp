#include "ap_fixed.h"
#include "hls_stream.h"

typedef ap_fixed<16,8> data_t; // Q8.8 fixed-point

void conv_layer(
    hls::stream<data_t> &in_stream,
    hls::stream<data_t> &out_stream,
    const data_t kernel[1][5][5]
){
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return
    
    #pragma HLS ARRAY_PARTITION variable=kernel complete dim=1
    #pragma HLS PIPELINE II=1
    
    static data_t line_buffer[4][28];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
    
    data_t window[5][5];
    
    // Sliding window processing
    CONV_LOOP: for(int i = 0; i < 24; i++) {
        for(int j = 0; j < 24; j++) {
            #pragma HLS LOOP_FLATTEN
            
            // Update window
            for(int wi = 0; wi < 5; wi++) {
                for(int wj = 0; wj < 5; wj++) {
                    if(wi < 4) {
                        window[wi][wj] = line_buffer[wi][j+wj];
                    } else {
                        window[wi][wj] = in_stream.read();
                    }
                }
            }
            
            // Convolution computation
            data_t sum = 0;
            for(int ki = 0; ki < 5; ki++) {
                for(int kj = 0; kj < 5; kj++) {
                    #pragma HLS UNROLL factor=4
                    sum += window[ki][kj] * kernel[0][ki][kj];
                }
            }
            
            // ReLU and output
            out_stream.write((sum > 0) ? sum : 0);
            
            // Shift line buffer
            for(int k = 0; k < 4; k++) {
                line_buffer[k][j] = line_buffer[k+1][j];
            }
            line_buffer[4][j] = window[4][0];
        }
    }
}

void max_pool(
    hls::stream<data_t> &in_stream,
    hls::stream<data_t> &out_stream
){
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS PIPELINE II=1
    
    POOL_LOOP: for(int i = 0; i < 12; i++) {
        for(int j = 0; j < 12; j++) {
            #pragma HLS LOOP_FLATTEN
            
            data_t max_val = -999;
            for(int pi = 0; pi < 2; pi++) {
                for(int pj = 0; pj < 2; pj++) {
                    #pragma HLS UNROLL
                    data_t val = in_stream.read();
                    if(val > max_val) max_val = val;
                }
            }
            out_stream.write(max_val);
        }
    }
}
