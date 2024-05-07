import sys
sys.path.append('d:/Jio_Institute/electric_vehicle_efficiency_prediction')

from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configurations"""

    model_name: str = "linear_regression"
    fine_tuning: bool = False
  