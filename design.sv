`timescale 1ns / 1ps

module cnn_accelerator (
    input wire clk,
    input wire reset,
    input wire start,
    output reg signed [15:0] predictions [0:9],
    output reg done
);

// Fixed-point configuration
parameter DATA_WIDTH = 16;
parameter FRAC_WIDTH = 8;

// Network parameters
parameter INPUT_CH = 1;
parameter INPUT_H = 28;
parameter INPUT_W = 28;
parameter CONV_KERNEL = 5;
parameter CONV_OUT_H = 24;
parameter CONV_OUT_W = 24;
parameter POOL_OUT_H = 12;
parameter POOL_OUT_W = 12;
parameter NUM_CLASSES = 10;

// State machine states
typedef enum logic [2:0] {
    IDLE,
    CONV,
    POOL,
    FC,
    DONE
} state_t;

state_t current_state, next_state;

// Performance counters
reg [31:0] conv_cycles;
reg [31:0] pool_cycles;
reg [31:0] fc_cycles;
reg [31:0] total_cycles;

// Operation counters
reg [15:0] conv_counter;
reg [15:0] pool_counter;
reg [15:0] fc_counter;

// Test image and weights (same as your original)
reg signed [DATA_WIDTH-1:0] test_image [0:27][0:27];
reg signed [DATA_WIDTH-1:0] conv_weights [0:0][0:4][0:4];
reg signed [DATA_WIDTH-1:0] fc_weights [0:9][0:0][0:11][0:11];
reg signed [DATA_WIDTH-1:0] fc_biases [0:9];

// Initialize weights and test image (same as your original)
initial begin
    // [Keep your existing initialization code]
end

// State machine
always @(posedge clk) begin
    if (reset) begin
        current_state <= IDLE;
        conv_cycles <= 0;
        pool_cycles <= 0;
        fc_cycles <= 0;
        total_cycles <= 0;
        conv_counter <= 0;
        pool_counter <= 0;
        fc_counter <= 0;
        done <= 0;
    end else begin
        current_state <= next_state;
        
        // Update performance counters
        case (current_state)
            CONV: conv_cycles <= conv_cycles + 1;
            POOL: pool_cycles <= pool_cycles + 1;
            FC: fc_cycles <= fc_cycles + 1;
        endcase
        
        // Update operation counters
        if (current_state == CONV) conv_counter <= conv_counter + 1;
        if (current_state == POOL) pool_counter <= pool_counter + 1;
        if (current_state == FC) fc_counter <= fc_counter + 1;
        
        // Done signal
        done <= (current_state == DONE);
    end
end

// Next state logic
always_comb begin
    next_state = current_state;
    case (current_state)
        IDLE: if (start) next_state = CONV;
        CONV: if (conv_counter == CONV_OUT_H*CONV_OUT_W-1) next_state = POOL;
        POOL: if (pool_counter == POOL_OUT_H*POOL_OUT_W-1) next_state = FC;
        FC: if (fc_counter == NUM_CLASSES-1) next_state = DONE;
        DONE: next_state = IDLE;
    endcase
end

// [Add your actual convolution, pooling, and FC implementation here]
// For now just dummy outputs
always @(posedge clk) begin
    if (current_state == FC) begin
        for (int i = 0; i < NUM_CLASSES; i++) begin
            predictions[i] <= i * 16'h0100 + fc_counter;
        end
    end
end

endmodule
