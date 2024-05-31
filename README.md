1. Install dependencies  pipenv install -r requirements.txt
2. Chạy lần lượt các bước sau\
       python3 01_api_data_request.py\
       python3 02_cleaning_stats_data.py\
       python3 03_feature_engineering.py\
       python3 04_feature_engineering_data_visualisation.py\
       cd ml_model_build_random_forest && python3 random_forest_model_build.py\
       cd ../predictions && python3 predictions.py\
3. Khởi chạy webserver\
       python3 web_server/server.py
