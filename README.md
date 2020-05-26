# Damage Prediction

The objective is predict the isurance cost or damage cost based on a number of categorial and numeric data. This github repository provides python training and testing code. It also provides methods for processing different types of data for machine learning modelling.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.
Below are the libraries used in Train script
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
import plotly.offline as py
py.init_notebook_mode(connected=True)

color = sns.color_palette()
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import seaborn as seabornInstance
%matplotlib inline
import json
```
Below is the list of libraries used in test script.

```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import pickle
import numpy as np
```

## Usage
### Machine Learning Modelling
The src folder contains the training and testing code
1. Train.ipynb is the jupyter notebook used for training and saving the Model. This files also highlights the steps.
2. Test.ipynb is the test jupyter notebook. It loads the model and necessary details needed for prediction.

### Docker Container for loading dataset
1. The CSV is uploaded in the docker container  
2. Below command creates the docker postgres container
   docker run -p 5432:5432 -d -e POSTGRES_USER="objectrocket" -e POSTGRES_PASSWORD="1234" -e POSTGRES_DB="some_db" -v ${PWD}/pg-data:/var/lib/postgresql/data --name pg-container postgres # Docker image
3. To build the image run the below command
   docker-compose up --build

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
