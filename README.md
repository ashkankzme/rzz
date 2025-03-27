# RZZ Transformer Project

This project trains a decoder-only transformer model in PyTorch to predict the imaginary part of the non-trivial zeros of the Riemann Zeta function (rzz). The data is provided via a function interface that returns zeros in high-precision format. For training the model, we work on a next token prediction task over a small vocabulary.

## Execution Order

1. **Hyperparameter Tuning**  
   Run `python src/tune_hyper_parameters.py` to grid search over a subset (first 10,000 zeros) and select the best hyperparameters.

2. **Training**  
   Run `python src/train.py` to train the model on successive million-zero intervals. The model saves checkpoints (which can be resumed).

3. **Evaluation**  
   Run `python src/evaluate.py` to evaluate the fully trained model on the test set using nucleus sampling (p = 0.9) and report the mean squared error.

## Notes

- The data is loaded via the function `zeros_starting_at_N(N, number_of_zeros)` from `zeros_db.py`.
- The tokens are defined over the vocabulary:  
  `['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ':', 'b', 'e', ' ', 'p']`
- The context window is fixed to 32 tokens. Data shorter than 32 tokens is padded with `'p'`.
- The transformer is implemented from basic building blocks in PyTorch.

Feel free to customize the code further to suit your hardware and runtime (Mac mini M4 or cloud GPUs/TPUs).
