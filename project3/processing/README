### Project3: Bus Location Prediction in Rio de Janeiro

#### Objective

The project aims to predict bus locations in two ways:
1. Predicting the latitude and longitude where the bus will be given elapsed time.
2. Predicting the elapsed time given the latitude and longitude.

Data from 50 bus routes in Rio de Janeiro were used, employing PostGIS and Python with Jupyter Notebooks for implementation.

#### Project Structure

- **Folder `db`**: Contains the notebook `load_db.ipynb` used to load data into PostGIS tables.

- **Folder `processing`**:
  - **Folder `modules`**: Contains Python modules used in the project.
    - `calculate_mean_speed_modules.py`: Functions for calculating bus speed models.
    - `predict_modules.py`: Classes and methods for performing both types of bus location predictions.
  - **Folder `notebooks`**: Contains Jupyter notebooks used for analysis and development.
    - `analyse_trajectory.ipynb`: Analysis and testing of methods to map bus trajectories.
    - `start_model.ipynb`: Development and testing of methods to create bus speed models.
    - `first_test_model.ipynb`: Testing bus speed models for both prediction types on a single bus line.
    - `evaluate_model_results.ipynb`: Evaluation and discussion of predictive model results applied to the 50 bus routes in the training data.

- **Folder `scripts`**: Contains Python scripts and a subfolder `routes`.
  - **Subfolder `routes`**: Contains HTML files created with Folium showing modeled bus routes for Monday, Saturday, and Sunday.
  - `calculate_mean_speed_points.py`: Calculates the speed model for a given bus route and inserts it into the `bus_speed_model` table in the database.
  - `calculate_predictions.py`: Predicts time, latitude, and longitude for bus routes in the training data and adds them to the `vehicle_tracking_pred_latlong` and `vehicle_tracking_pred_datahora` tables.
  - `calculate_predictions_teste.py`: Predicts time, latitude, and longitude for bus routes in the test data and adds them to the `vehicle_tracking_pred_latlong_teste` and `vehicle_tracking_pred_datahora_teste` tables.
  - `calculate_traj_paralel.py`: Calculates bus trajectories for all 50 lines in parallel using multithreading for Monday, Saturday, and Sunday.
  - `calculate_end_points.py`: Calculates initial and final points for 50 bus routes using heuristics explored in `analyse_trajectory.ipynb` and adds them to the `bus_end_points` table.
  - `paralel_script_execution.py`: Parallelizes script execution using multithreading across bus routes.
  
- **Folder `api`**: Contains a notebook for API interaction.
  - `request_evaluation.ipynb`: This notebook makes requests to the API designated by the professor using the username "Maria Luiza" and the password "120040005". It evaluates the test results.

#### Notes

- Monday was chosen as the representative weekday for modeling bus routes during weekdays.
- The outcome was not as good as expected, but many lessons were learned during the work, such as: handling large volumes of data, manipulating geographic data, using spatial databases, manipulating APIs and others.