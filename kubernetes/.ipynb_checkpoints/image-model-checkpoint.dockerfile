FROM tensorflow/serving:2.7.0

COPY models/converted_model models/converted_model/1
ENV MODEL_NAME="converted_model"