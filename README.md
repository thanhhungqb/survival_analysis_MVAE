# Multiple Task Variational Autoencoder for survival analysis
This repos contains source code for [paper](https://doi.org/10.3390/electronics10121396):

Vo, Thanh-Hung, Guee-Sang Lee, Hyung-Jeong Yang, In-Jae Oh, Soo-Hyung Kim, and Sae-Ryung Kang. 2021. "**Survival Prediction of Lung Cancer Using Small-Size Clinical Data with a Multiple Task Variational Autoencoder**" Electronics 10, no. 12: 1396. https://doi.org/10.3390/electronics10121396


# Data
- The form of data mostly come from CSV/XLSX
- Predefined "fold" column, e.g., randomly 1-5, or 0-4, default fold 1 is for the test (can customizable in configure file, test_fold)

# Configure:
The important step to configure is in *.json, where most configuring data, model, etc., are placed.
There are some important notes:
- meta_df: where XLSX data file
- cat_names, cont_names, and y_names should be defined
- model, loss_func: model and loss function definition
- process_pipeline_1: you may need custom cut points if y different

For *.sh, the bash shell to run experiments.
The command-line to run is "python -m prlab.cli run" (in a bash script, *.sh), change to "python -m prlab.cli k_fold" if want to run k-fold cross-validation (*k* should be defined, *k_start* default is 0)

# Run
- setup environment (virtualenv is recommend, "pip install --upgrade pip wheel setuptools" may be needed)
- pip install -r requirements.txt
- setup and login for [wandb](https://wandb.ai/site) to save logs
- custom [run.sh](run.sh) to the path of configuring and some other configure
- ./run.sh
- The report will be in models/*/reports*.txt.



Source code is provided 'as-is' WITHOUT any WARRANTY or SUPPORT. Using this script is at YOUR OWN RISK.

