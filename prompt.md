Riemann Zeta Predictor Prompt


Description:

Write a series of python programs that train a decoder only transformer model to predict the real part of the non trivial zeros of the “Riemann Zeta function” using PyTorch to run on a Mac mini m4. 

The training task involves time-series style prediction of “the real part of the non trivial zeros of the Riemann Zeta function”, formulated as next token prediction with the 103 billion zeros provided as training, validation, and test data.

You should write the following scripts

- Running hyper parameter tuning using the validation split.
- Training a decoder only transformer model in PyTorch on the training split.
- Evaluating on the test split, by averaging over 10 runs of nucleus sampling until <eoa> token reached, on every input. Assume you have access to an already trained model saved locally.


You should write the following scripts

- Running hyper parameter tuning using the validation split to identify the best model size (number of layers, number of heads, hidden dimension size) and training hyperparameters (learning rate, batch size, etc.)
- Training a decoder only transformer model in PyTorch on the training split. this script will train the transformer model on a next token prediction task where the input is the i-th zero and the output is the imaginary part of the i-th zero, rounded to 4 decimal places.
- Evaluating on the test split, by averaging over 10 runs of nucleus sampling until <eoa> token reached, on every input. Assume you have access to an already trained model saved locally.


Background on the Riemann Zeta function:

The Riemann Zeta function is defined as:
zeta(s) = sum(1/n^s) for n=1 to infinity

The non trivial zeros of the Riemann Zeta function are the complex numbers s where zeta(s) = 0 and Re(s) = 0.5
For the purposes of this project, we will only be considering the imaginary part of the non trivial zeros, as the real part is always 0.5.



System spec and related requirements: 

This project will primarily run on a Mac Mini M4.

The specs of the Mac Mini M4 are as follows:
Apple M4 chip
* 10-core CPU with 4 performance cores and 6 efficiency cores
* 10-core GPU
* Hardware-accelerated ray tracing
* 16-core Neural Engine
* 120GB/s memory bandwidth
Memory: 16GB
Storage: 256GB SSD (not all available to program, assume you have 64GB available for storage)
MacOS: 15.3.1


DATA FORMAT: 
where is data located, what is the format, how to split into train val test

Make sure training test and validation splits come in chronological order. Training comes first, then validation, then test.

Zeros available here in .DAT format: https://beta.lmfdb.org/riemann-zeta-zeros/data/

Load zeros only to 4 decimal points

This dataset contains the first 103,800,788,359 zeros of the Riemann zeta function, available in many smaller .DAT files on demand

There is one zero in each line.

The order of the zeros are important and the program must be able to predict the i-th zero when prompted, for any I values


Tokenization and vocabulary:

The vocabulary is as follows:
vocabulary: list[str] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ':', 'b', 'e', ' ', 'p']

'b' is the beginning of the sequence token
'e' is the end of the sequence token
'p' is the pad token

':' is used to separate the i-th index from the imaginary part of the zero
'.' is used to separate the integer part of the imaginary part from the decimal part of the imaginary part
for example: 4935:3847.2344



Training:

The training task is next token prediction on a small vocabulary. The vocabulary is as follows:

Similar to language modeling. Use a decoder only transformer model. 



Specifics of hyperparameter tuning, training, and evaluation



Validation is about finding the right size and dimensions of the transformer model that can be run easily on m4 Mac mini

Following guidelines must be followed:

Code should be modular, clean, and well documented
The shared functionality among the three scripts should be refactored into a utils.py file that the scripts import locally
Requirements should be listed under requirements.txt in the root of the project
Training Code should save progress and checkpoints at reaspnable intervals, and training should be easily paused and resumed without losing much progress.
All code must be usable on Google collab GPU and TPUs, but also should be runnable locally on Mac m4 mini. This includes checkpoints that are being saved as well, they should be portable across different runtimes