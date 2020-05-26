Shivam Sharma


#############################################################
Machine Learning Model
#############################################################
The src folder contains the training and testing code
1. Train.ipynb is the jupyter notebook used for training and saving the Model. This files also highlights the steps.
2. Test.ipynb is the test jupyter notebook. It loads the model and necessary details needed for prediction.

#############################################################
DOCKER CONTAINER
#############################################################
1. The CSV is uploaded in the docker container  
2. Below command creates the docker postgres container
   docker run -p 5432:5432 -d -e POSTGRES_USER="objectrocket" -e POSTGRES_PASSWORD="1234" -e POSTGRES_DB="some_db" -v ${PWD}/pg-data:/var/lib/postgresql/data --name pg-container postgres # Docker image
3. To build the image run the below command
   docker-compose up --build
