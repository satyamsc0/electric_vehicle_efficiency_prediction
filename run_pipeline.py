from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path=r"D:\Jio_Institute\electric_vehicle_efficiency_prediction\data\Cheapestelectriccars-EVDatabase 2023.csv")
# source MLOps/bin/activate
# "file:/home/dhruba/.config/zenml/local_stores/d637b3fd-029c-4462-92f5-92a843a8616e/mlruns"
# zenml integration install mlflow -y
 #zenml experiment-tracker register mlflow_tracker_customer --flavor=mlflow
#  zenml model-deployer register mlflow_customer --flavor=mlflow
# zenml stack register mlflow_stack_customer -a default -o default -d mlflow_customer -e mlflow_tracker_customer --set
#  mlflow ui --backend-store-uri 'file:/home/dhruba/.config/zenml/local_stores/d637b3fd-029c-4462-92f5-92a843a8616e/mlruns'

#  python3 run_deployment.py --config deploy
#  python3 run_deployment.py --config predict 

#new stack
# zenml experiment-tracker register mlflow_tracker_newone --flavor=mlflow
# zenml model-deployer register mlflow_newone --flavor=mlflow
# zenml stack register mlflow_tracker_newone -a default -o default -d mlflow_newone -e mlflow_tracker_newone --set


#  zenml experiment-tracker register mlflow_prediction  --flavor=mlflow
#  zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_prediction  --set

# streamlit run streamlit_app.py