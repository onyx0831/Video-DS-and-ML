from feature_engineering import run_feature_engineering
from model_training import run_training
from model_predict import run_predict

def main():
    experiment_name = "iris_experiment"
    run_feature_engineering(experiment_name=experiment_name)
    run_training(experiment_name=experiment_name)
    run_predict(experiment_name=experiment_name)

if __name__ == "__main__":
    main()
