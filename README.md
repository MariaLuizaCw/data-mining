
# Data Mining Course - PESC UFRJ

Welcome to the repository for the Data Mining course at the Program Systems Engineering and Computer Science (PESC), Federal University of Rio de Janeiro (UFRJ). This repository contains Jupyter notebooks showcasing various data mining techniques and projects completed during the course.

## Projects

### Project 1: Obesity Level Classification

- **Folder**: `project1`
- **Objective**: Classify the level of obesity in individuals based on their eating habits and physical condition.
- **Data Source**: This project uses the "Obesity Levels" dataset from Kaggle.
  - `teste.csv`: Test sample downloaded from Kaggle
  - `train.csv`: Train sample downloaded from Kaggle
  - `sample_submission.csv`: Submission example downloaded from Kaggle
  - `model_output.csv`: Classification made by the developed model using Test sample
- **Notebooks**:
  - `obesity.ipynb`: Notebook containing both analysis and model developtment for the classification task.
  - `KaggleScores.png`: Print with the scores achieved in Kaggle submission.

### Project 2: Store Sales - Time Series Forecasting

- **Folder**: `project2`
- **Objective**: Forecasting store sales using time series data to predict future sales accurately.
- **Data Source**: This project uses the "Store Sales - Time Series Forecasting" dataset from Kaggle.
  - `holidays_events.csv`: Contains information about holidays and events.
  - `oil.csv`: Contains historical oil price data.
  - `sample_submission.csv`: Submission example downloaded from Kaggle.
  - `stores.csv`: Contains information about different stores.
  - `test.csv`: Test sample downloaded from Kaggle.
  - `train.csv`: Train sample downloaded from Kaggle.
  - `transactions.csv`: Contains transaction data for each store.

- **Notebooks**:
  - The general strategy was to create a model for each store, adding information to these models over the steps.
  - `explore.ipynb`: First look and all data samples.
  - `step1.ipynb`: Forecasting using only the train dataset.
  - `step2.ipynb`: Adding holidays dataset to the model.
  - `step3.ipynb`: Incorporating oil price information into the model.
  - `step4.ipynb`: Including employee payment information  to enhance the model.

- **Data Subfolders**:
  - Inside the `data` folder, there is a subfolder for each step (`step1`, `step2`, `step3`, `step4`) containing:
    - Auxiliary CSVs created during the execution of the respective notebook.
    - Results obtained for the variations generated using two algorithms: Random Forest Regressor and Gradient Boosting Regressor `results.csv`.

- **Submissions Folder**:
  - The `submissions` folder contains:
    - `submission_step_1.csv`
    - `submission_step_2.csv`
    - `submission_step_3.csv`
    - `submission_step_4.csv`
  - These files contain the preparation of the submission for the best results from the four steps.


## Installation

To run the notebooks, you will need to install the required Python packages. You can install all required packages using the following command:

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file in the root of your repository if it doesn't exist and list all necessary packages.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with any enhancements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Thanks to the professor Zimbrão.
- Kaggle for providing the datasets used in our projects.
