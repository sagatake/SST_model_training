# Directory structure

- 0_feature_calculation
Feature calculation directory.
You need to place video/audio/text files into src_xx directory.
Please follow the instruction bellow for detailed procedure.

- 1_train_regression.py
Training code for score estimation models.
Since this code uses cross validation for evaluation, you need to train the finalized models by using whole dataset with 2_save_trained_model.py afterwards.
If the model's performance is high enough, you can move on to the next step to save the finalized models

- 2_save_trained_model.py
Code to train and save the finalized model.

- aligned_fature_par
Feature files directory

- aligned_subj_score_par
Label files directory

- feature_label_example
Example data of features and labels to train models

# Get started

0. Create and activate conda environment using .yml file in env directory
1. Place videos/audios/texts in 0_feature_calculation/src_video, src_audio, src_text.
2. In command prompt, run "cd 0_feature_calculation" and "python eval_pipeline/multipipeline.py"
3. Data preparation phase. place your features and labels as same as the feature_label_example directory
3-1. Copy data/features.csv into aligned_feature_par
3-2. Place csv file of corresponding score labels in aligned_subj_score_par
4. Train regression models by running "python 1_train_regression.py" in command prompt
5. If the score is high enough, run "python 2_save_trained_model.py" to finalize your models
