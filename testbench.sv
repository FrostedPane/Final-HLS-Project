`timescale 1ns / 1ps

module tb_cnn_accelerator;

// Parameters
parameter DATA_WIDTH = 16;
parameter NUM_CLASSES = 10;

// Testbench signals
reg clk;
reg reset;
reg start;
wire signed [DATA_WIDTH-1:0] predictions [0:NUM_CLASSES-1];
wire done;

// Instantiate DUT
cnn_accelerator dut (
    .clk(clk),
    .reset(reset),
    .start(start),
    .predictions(predictions),
    .done(done)
);

// Clock generation
always #5 clk = ~clk;

// Fixed-point to float conversion
function real fixed_to_float;
    input [15:0] fixed;
    begin
        fixed_to_float = $signed(fixed) / 256.0;
    end
endfunction

// Test stimulus
initial begin
    // Initialize signals
    clk = 0;
    reset = 1;
    start = 0;
    
    // Dump VCD file for waveform viewing
    $dumpfile("dump.vcd");
    $dumpvars(0, tb_cnn_accelerator);
    
    // Reset
    #20;
    reset = 0;
    #10;
    
    // Start processing
    $display("[%0t] Starting simulation", $time);
    start = 1;
    #10;
    start = 0;
    
    // Wait for completion
    fork
        begin
            wait(done == 1);
            $display("\n[%0t] Simulation completed successfully", $time);
            
            // Performance report
            $display("\nPerformance Analysis");
            $display("====================");
            $display("Module              Latency(cycles)");
            $display("Convolution Layer   %0d", dut.conv_cycles);
            $display("Pooling Layer       %0d", dut.pool_cycles);
            $display("FC Layer            %0d", dut.fc_cycles);
            $display("Total               %0d", dut.conv_cycles + dut.pool_cycles + dut.fc_cycles);
        end
        begin
            #500000; // 500us timeout
            $display("\n[%0t] Simulation timed out", $time);
            $display("Current state: %0d", dut.current_state);
            $display("Counters: Conv=%0d Pool=%0d FC=%0d", 
                    dut.conv_counter, dut.pool_counter, dut.fc_counter);
            $finish;
        end
    join_any
    
    // Display raw logits (fixed-point values)
    $display("\nRaw Logits (Fixed-Point):");
    $display("----------------------------");
    for (int i = 0; i < NUM_CLASSES; i++) begin
        $display("Class %0d: %0d (hex: %h)", i, predictions[i], predictions[i]);
    end
    
    // Display converted logits (floating-point values)
    $display("\nConverted Logits (Floating-Point):");
    $display("------------------------------------");
    for (int i = 0; i < NUM_CLASSES; i++) begin
        $display("Class %0d: %f", i, fixed_to_float(predictions[i]));
    end
    
    // Determine and display predicted class
    begin
        integer predicted_class = 0;
        real max_score = fixed_to_float(predictions[0]);
        real current_score;
        
        $display("\nDetermining predicted class...");
        
        for (int i = 1; i < NUM_CLASSES; i++) begin
            current_score = fixed_to_float(predictions[i]);
            if (current_score > max_score) begin
                max_score = current_score;
                predicted_class = i;
                $display("New max found: Class %0d with score %f", i, current_score);
            end
        end
        
        $display("\nFinal Prediction:");
        $display("----------------");
        $display("Predicted Class: %0d", predicted_class);
        $display("Confidence Score: %f", max_score);
        
        // Print runner-up classes
        $display("\nTop 3 Predictions:");
        begin
            real scores [0:9];
            integer indices [0:9];
            
            // Initialize arrays
            for (int i = 0; i < NUM_CLASSES; i++) begin
                scores[i] = fixed_to_float(predictions[i]);
                indices[i] = i;
            end
            
            // Simple bubble sort (for small NUM_CLASSES)
            for (int i = 0; i < NUM_CLASSES-1; i++) begin
                for (int j = i+1; j < NUM_CLASSES; j++) begin
                    if (scores[i] < scores[j]) begin
                        real temp_score = scores[i];
                        int temp_index = indices[i];
                        scores[i] = scores[j];
                        indices[i] = indices[j];
                        scores[j] = temp_score;
                        indices[j] = temp_index;
                    end
                end
            end
            
            // Display top 3
            for (int i = 0; i < 3; i++) begin
                $display("%0d. Class %0d: %f", i+1, indices[i], scores[i]);
            end
        end
    end
    
    // Finish simulation
    #100;
    $finish;
end

endmodule
