# SQuAD-task
Machine Reading Comprehension task


## Model: DCN Encoder + Answer Pointer Approach
#### Hyper parameter values (final) and Results
  * Hyper-Parameters:
    batch_size = 40, embed_size = 100, lstm_units = 100, dropout = 0.07, learning_rate = 0.001, Optimiser = AdamOptimizer    
  * Results:
    Train-Data : Exact-Match = 57.433 %, F1 = 71.077 %    
    Dev-Data   : Exact-Match = 59.0909 %, F1 = 71.467 %        
