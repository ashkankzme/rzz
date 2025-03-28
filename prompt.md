Description
Write a collection of python programs, as described in great detail in the rest of this prompt, that train a decoder only transformer model to predict the imaginary part of the non trivial zeros of the “Riemann Zeta function” using PyTorch to run on a Mac mini m4. 

The training task involves time-series style prediction of “the imaginary part of the non trivial Riemann Zeta zeros” (rzz), formulated as “next token prediction” with the zeros provided to be used as training, validation, and test data through a python function interface.

At a high level, the program should support the following core functionality:
- Running hyper parameter tuning to identify the optimal model (number of layers, number of heads, hidden dimension size) and training hyperparameters (learning rate, batch size, etc)
- Training a decoder only transformer model in PyTorch on the time-series training data. This script will train the transformer model on a next token prediction task where the input is the i-th zero index and the model is expected to predict the i-th rzz to 4 decimal places token by token.
- Evaluating model performance on the test split, by averaging over 10 runs of nucleus sampling until “end” token reached using the model trained in previous steps. 
Background on the Riemann Zeta function
The Riemann Zeta function is defined as:
zeta(s) = sum(1/n^s) for n=1 to infinity

The non trivial zeros of the Riemann Zeta function are the complex numbers s where zeta(s) = 0 and Re(s) = 0.5 (ie. real part) For the purposes of this project, we will only be considering the imaginary part of the non trivial zeros, as the real part is always 0.5.
System spec and related requirements
This project will primarily run on a Mac Mini M4.
The specs of the Mac Mini M4 are as follows:
Apple M4 chip
* 10-core CPU with 4 performance cores and 6 efficiency cores
* 10-core GPU
* Hardware-accelerated ray tracing
* 16-core Neural Engine
* 120GB/s memory bandwidth
Memory: 16GB
Storage: 256GB SSD (not all available to program, assume you have 64GB available for storage of checkpoints and training data)
MacOS: 15.3.1

Although this code is primarily going to be run on an M4 Mac mini, I am considering using more powerful GPUs provided by Google collab and other cloud GPU providers. Therefore all code must be usable on Google collab GPU and TPUs, while continuing to run locally on Mac m4 mini. This includes checkpoints that are being saved as well, they should be portable across different runtimes.

Project file structure
The project should roughly follow this structure:

rzz/
src/
data_utils.py
transformer.py
tune_hyper_parameters.py
train.py
evaluate.py
config.json
data/
checkpoints/
logs/
requirements.txt
readme.md

If you think having additional files will be necessary for modularity and readability of the code, feel free to add them, but do not remove or modify any of the existing directory and file structure.
Data
Write a PyTorch dataset interface that reads from the database of imaginary parts of riemann zeta zeros (rzz) and returns the requested zeros to training, test, and hyperparameter tuning scripts. The data fetching module and all usable utilities should be stored under data_utils.py as specified in the project structure, and the code should be reusable for the training and other scripts.

Assume you have access to a local script under “rzz/src/zeros_db.py”, and you can import the function “zeros_starting_at_N(N, number_of_zeros)” from it. The zeros_starting_at_N() function takes N (index starting from 0) and number_of_zeros (count of zeros from start to return), and return the count of zeros requested starting from the N parameter. This interface should be enough to load data into memory and use in core training and evaluation scripts. The function returns data in the following format:

> zeros_starting_at_N(10, 3)
> for z in zeros:
>     print(z)
(10, mpf('49.7738324776723021819167846785638367196'))
(11, mpf('52.9703214777144606441472966088808215661'))
(12, mpf('56.4462476970633948043677594767060321269'))

The first value in each item is the i-th index, starting with an offset of 10 in this example.
The second value is the imaginary part of the i-th riemann zeta zero with high precision in mpf format, generated by the “mpmath” library. 

After fetching the right interval of zeros, you must first keep only the first 4 decimal places of the i-th rzz and discard the rest. For our project’s purposes, learning to predict only the first 4 decimal places of the answer is sufficient. To become usable by the transformer model, each rzz training sample should be reformatted in the following way:
“b<i>:<rzz>epp…p” (number of padding tokens determined based on remaining context from the 32 token context window size). For example, this is what every datapoint printed above should be reformatted to:

“b10:49.7738epppppppppppppppppppp”
“b11:52.9703epppppppppppppppppppp”
“b12:56.4462epppppppppppppppppppp”

Since we are dealing with time-series data, the train and validation splits are t-th and (t+1)-th million zeros, for all t in range of 1 to K. For example, at t = 2, the training set is zeros from 2,000,000 to 2,999,999, and the validation set is a fixed randomly selected 1% subset of zeros from 3,000,000 to 3,999,999. At the next iteration of training (t = 3), the training zeros roll over to the 3,000,000 to 3,999,999 range, and similar dynamics applies to the validation set as well. At every time t, the current millionth interval I_t is used in training, and a fixed random 1% subset of the I_t+1 interval (draw a sample once, store, and use the same sample for validation loss until t+1) will be used for measuring validation loss. After the training round t is concluded, t = t+1, and the training and validation splits increment by one, each moving to the next file in the directory. 

Assume that the training split ends after going through K million zeros, and K is set as a configurable parameter in config.json, and is set to 10 as default, which means training will be done on the first 10 million rzzs. Zeros (K+1)-million and forward will be treated as test data and used in evaluating the fully trained model.

The order of the zeros are important and the program must be able to predict the i-th zero when prompted, for any i values. This means that random shuffling of the input is prohibited and training data zeros must go in sorted order of occurrence.
Tokenization and vocabulary
The vocabulary is as follows:
vocabulary: list[str] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ':', 'b', 'e', ' ', 'p']

'b' is the beginning of the sequence token
'e' is the end of the sequence token
'p' is the pad token
':' is used to separate the i-th index from the imaginary part of the zero
'.' is used to separate the integer part of the imaginary part from the decimal part of the imaginary part

Example input: 4935:3847.2344
Tokenized input: [‘b’, ‘4’, ‘9’, ‘3’, ‘5’, ‘:’, ‘3’, ‘8’, ‘4’, ‘7’, ‘.’, ‘2’, ‘3’, ‘4’, ‘4’, ‘e’, ‘p’, …, ‘p’]
(Padding is repeated to fill context window size of 32)

Assume there is no whitespace, and tokenize character by character. Token ID lookup map should be small and simple to implement.
Training
The training task is next token prediction (similar to language modeling) on a small vocabulary as defined above.
In training, each rzz will be predicted as a few next token prediction tasks starting from the “b<i>:” point, until the ‘e’ token has been reached.
That means that the following fake input “b1:23.4567e” will be transformed into the following training trajectories:
in: “b1:”, target: ”2”
in: “b1:2”, target: “3”
in: “b1:23”, target: “.”
in: “b1:23.”, target: “4”
in: “b1:23.4”, target: “5”
in: “b1:23.45”, target: “6”
in: “b1:23.456”, target: “7”
in: “b1:23.4567”, target: “e”

*(padding will be added to each example before data feeding into the model)

These trajectories are fed into the model during training, in the order they occur in time as demonstrated in above example, to prevent leakage from future inputs into current targets. Note that trajectories generated from an input always start after the “b<i>:” input, as there is no point in predicting “<i>”, and you can assume that i-th index will always be provided in the input.

Use cross entropy loss for training, and the loss will be calculated over the vocabulary of possible next tokens. For logging, debugging, and understanding purposes, also log the MSE loss at each interval for both training and validation sets. The details on how to calculate this loss are under section “Inference and evaluation on test set.”

Use GELU for the feed-forward activation layers. Use a decoder only transformer model, specified with the same optimal hyper parameters discovered in the validation round. Assume the context window size is 32 tokens, and any input that is shorter than 32 will be padded using the ‘p’ token.

Write the transformer code using PyTorch from basic building blocks and do not import a full transformer decoder from any libraries. This reusable code should be under a separate file called transformer.py and the training, validation and test scripts will import them from this file.

It’s important to be able to pause and resume training. Upon resuming, script should always load from the latest stored checkpoint and continue training exactly where it was left off to the accuracy of which data point, training step, and epoch. Save the last checkpoint (updated on every save) as well as a checkpoint at the end of each training round, i.e. at the end of every 1 million zeros. Update the last model checkpoint every few training iterations, and make it a configurable parameter in config.json.
Hyperparameter tuning
This is the first script that is run in the pipeline, and its goal is to determine the hyperparameters used in training and other scripts further down the line.

You can use the first 10000 zeros to do grid search on the hyperparameters and choose the best performing one.
The following parameters need to be grid searched:
Number of epochs: range from 1-16
Learning rate: use a sensible range for this type of scenario
Batch size: [16, 32, 64, 128]
Transformer-dependent parameters: Model Dimensionality (d_model), Number of Attention Heads (num_heads), Number of Layers, Attention Head Dimensionality (d_k, d_v), different Dropout Rates and anything else appropriate

Assume the context window size is 32 and you do not have to do any hyperparameter tuning for the context window.

Feel free to add any other necessary parameters not listed here that cannot be intuitively set and would need hyperparameter tuning. However not every hyperparameter needs to be searched, only the impactful and important ones, since we have limited resources to dedicate to hyperparameter searching.

It’s important that part of this hyperparameter tuning involves choosing the right model dimensions so that training works well and fast enough to be run on a Mac mini M4 as specified above. That means that the hyperparameter tuning code should also take into consideration resource usage on the host machine and try to optimize or rule out values for the hyperparameters. Make a best effort on this but it does not have to be super complex, you can make assumptions based on the system spec and not test for actual usage.
Inference and evaluation on test set
In inference time and on the test set, the goal is to have the model predict zeros it has not seen before, and should be scored based on how well it can predict the actual number, making a mean squared error (MSE) loss appropriate as a measure of success, which is different from the cross entropy loss used in training.

The test set is the first N zeros (N=1000 as default) from the (K+1)-th file in the remote directory. K is defined earlier under the data section and refers to the last 1 million zeros used in training. Both K and N must be configurable in implementation, and that should be reflected in data_utils.py.

For each data point in the test set, the last trained checkpoint must be used to predict the next token on the i-th zero. The starting point for each input will look like the following string: “b<i>:pp…p” After each token is predicted, the new token will be appended to the input, and this will be repeated in a loop until the ‘e’ token is produced by the model.The MSE loss will then be accumulated by transforming the predicted i-th rzz into a float, and comparing it with the actual i-th rzz over the span of the test set.

Nucleus sampling should be used to generate next tokens, with p = 0.9. Print the result of evaluation in standard output.
General guidelines
Pay close attention to all the details of the requirement provided in this prompt, and fill in the blanks as best as you can by taking and interpolating on existing context in the prompt.
Order of execution is: hyperparameter tuning → training → evaluation on test set
It’s important to have clear logging, so include enough details in error logs so they can be traced back to the line of code that caused them. Logs should be stored locally on disk. This includes a separate class of logs dedicated to the training loop and include details such as training loss at step N, and so on.
Code should be modular, clean, and well documented
The shared functionality among the scripts should be refactored into a “*_utils.py”, “transformer.py”,  or similar files that they all import, to reduce duplicate code and improve reusability.
Project requirements should be listed under requirements.txt in the root of the project
Training Code should save progress regularly, and training should be easily paused and resumed without losing significant progress.
All code must be usable on Google collab GPU and TPUs, but also should be runnable locally on Mac m4 mini. This includes checkpoints that are being saved as well, they should be portable across different runtimes

