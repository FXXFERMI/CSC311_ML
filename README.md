# CSC311_ML
Keep a journal and record the experiments that we perform during CSC311 ML Challenge

## Project Structure
- `data/`: Store our data clean result & the final pred_new.py(we store the pred.py here because we want to simulate the MarkUs test cases)
    - `Q1_4_analysis_data/`:
    - `Q5_analysis_data/`:
    - `Q6_9_analysis_data/`:
    - `Q10_analysis_data/`
    - `raw_data/`:
        - Where the raw dataset from quercus stores
    - `pred_data`:
      - `final`:
        - `pred_new.py`: our final pred version on MarkUs
        - `estimators.csv`: the estimators for our final model
        - `final_vocab.csv`: the vocab list for our model
        - all the rest are just test cases
      - `matrix`: our data matrix there


- `models/`:Store our model scripts
  - `Q1_4_models/`
  - `Q5_models/`
  - `Q6_9_models/`
  - `Q10_models/`
           

- `scripts/`:
  - `Q1_4_code/`:
      - `Q1_4_data_cleaning.py`
  - `Q5_code/`:
      - `Q5_data_cleaning.py`
  - `Q6_9_code/`:
      - `Q6_9_data_cleaning.py`
  - `Q10_code/`:
      - `Q10_data_cleaning.py`

- `pred`: where our final model scripts stored
  - `data_cleaning.py`: data cleaning for the whole dataset
  - `new_model_logistic_newest.py`: the final model scripts, where we train our final model
  - all the rest are the models we wrote to compare


- `repot/`:
  - `final_report.qmd`: where to write our report.
  - `final_report.pdf`: the pdf version of our report
        
# How to Use
1. Clone or download this repository to your local machine.
2. Ensure you have python 3.10 and necessary packages installed(sklearn, numpy, pandas)
3. Run the scripts in the pred/ directory to reproduce the data cleaning process, analysis, and model fitting.
4. Explore the data/pred_data/final directory to view our predictions.

# License
This project is licensed under the MIT License. See the LICENSE file in the repository root for more information.

# Contact
For any queries regarding this study, please contact Siqi Fei at fermi.fei@mail.utoronto.ca. 
Further materials and updates can be found at this GitHub repository.