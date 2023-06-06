# Lane-Change-Classification-with-LSTM
## Method
Long Short Term Memory Networks are RNN based methods designed for process sequence data. LSTM networks can learn long distance context dependent information and store context history information. Therefore, it can be used to learn sequence data, such as lane change clues. This method employed LSTM Networks as classifier and vehicle bounding box coordinates as input data for lane change classification.

## Architecture of the LSTM method. 

<img width="530" alt="Screenshot 2023-06-06 at 11 55 04" src="https://github.com/kailliang/Lane-Change-Classification-with-LSTM/assets/56094206/1187e990-568e-4fb1-8739-30de4ebbaa2c">

a) input data: bounding box coordinates of 60 frames. 

b) LSTM layer. 

c) linear layer. 

d) output: left lane change, right lane change and lane keeping.
