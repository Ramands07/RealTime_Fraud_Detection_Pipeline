## END to END REALTIME FRAUD DETECTION 

Steps followed in creating this project

1. Created the repository
2. created the environment
3. done the setups which includes setup.py (connection file), source as src in this created - logger.py to get logs , exception.py to handle exception and utility as utlis.py
4. 







Realtime_fraud_detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebook/
│
├── src/
│   ├── component/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_training.py
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── models/
├── reports/
├── tests/
├── mlruns/
│
├── .gitignore
├── .dockerignore
├── Dockerfile
├── README.md
├── requirements.txt
├── setup.py
└── .env




