from azureml.core.model import Model
from azureml.core import Workspace, Dataset
from utility.utils import *
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
import joblib
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import LocalWebservice
from recommender.non_personalized import *
import pickle
import logging 

FORMAT = '%(asctime)-15s - %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("main.py")
logger.setLevel("INFO")

FOLDER_NAME = "models"
WHERE_TO_DEPLOY = get_credentials()["environment"]

logger.info("Inizialing script...")
URM_train, URM_test = import_data(0.8)
logger.info("Finished importing data")

logger.info("Inizialing a random model...")
randomRecommender = RandomRecommender()
randomRecommender.fit(URM_train)
logger.info("Training finished")

logger.info("Saving the model...")
MODEL_NAME = "local_model"
f = f"{FOLDER_NAME}/{MODEL_NAME}.pkl"
joblib.dump(randomRecommender, f)
logger.info("Best model saved!")

ws = get_workspace()

model = Model.register(workspace=ws,
                       model_path=f,
                       model_name=MODEL_NAME,
                       tags={"version": "1"},
                       description="movielens_10M")
logger.info(Model.get_model_path(MODEL_NAME, _workspace=ws))

logger.info("Model registered")

aciconfig = AciWebservice.deploy_configuration(
            cpu_cores=1,
            memory_gb=1,
            tags={"data":"Random Recommender",
            "Gestore": "Alessandro Artoni",
            "Owner": "Alessandro Artoni",
            "Environment": "dev",
            "Progetto": "Random recommender example"},
            description='Example on how to deploy a random recommender',
            )

logger.info("ACI Deployed")

env = Environment('custom')
env.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'azureml-defaults',
    'joblib',
    'numpy'
])

inference_config = InferenceConfig(entry_script="score.py", source_directory="recommender", environment=env)

logger.info("Inference config setted")

if(WHERE_TO_DEPLOY=="LOCAL"):
    deployment_config = LocalWebservice.deploy_configuration(port=8890)
    # Deploy the service
    service = Model.deploy(
        ws, "localmodel", [model], inference_config, deployment_config)
    # Wait for the deployment to complete
    service.wait_for_deployment(True)
    # Display the port that the web service is available on
    logger.info(service.port)
    logger.info(service.scoring_uri)

if(WHERE_TO_DEPLOY=="REMOTE"):
    logger.info("Deploying the model remotely")
    service = Model.deploy(workspace=ws,
                    name='aci-local',
                    models=[model],
                    inference_config=inference_config,
                    deployment_config=aciconfig,
                    overwrite = True)
    service.wait_for_deployment(show_output=True)

    logger.info("Model deployed")

    url = service.scoring_uri
    logger.info(url)
