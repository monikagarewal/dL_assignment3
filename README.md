# dL_assignment3

Assignment 3 

Hindi dataset was used from Google's Dakshina for training Seq2Seq models.

-main.py
1. It contains multilayered seq2seq network with all the required functionality.
2. train() function is the start point of code.
3. Make is_wandb_active = False to train model on best hyperparameter and evaluate as required.
4. Making is_wandb_active = True will start the sweep and will begin searching for best hyper param.
5. Comments are added above all sections of code to identify which part does what.
6. predictions_vanilla contains prediction on test data set.

-attn_main.py
1. It contains single layered network with all the required functionality.
2. run() function is the start point of code.
3. train() function is the training loop per step.
3. Make is_wandb_active = False to train model on best hyperparameter and evaluate as required.
4. Making is_wandb_active = True will start the sweep and will begin searching for best hyper param.
5. Comments are added above all sections of code to identify which part does what.
6. predictions_attention contains prediction on test data set.


