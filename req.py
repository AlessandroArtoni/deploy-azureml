import requests
from utility.utils import *
import json

headers = {"Content-Type":"application/json"}

# this is my ACI url
ACI_url = "http://bd689cc1-2442-4337-baea-3a81e8cbced5.westeurope.azurecontainer.io/score"

r = requests.post(ACI_url, data=json.dumps({"data":1}), headers=headers)
print(r.status_code)
print(r.json())
