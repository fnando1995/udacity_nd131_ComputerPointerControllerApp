import pandas as pd
import logging
import cv2
from src.input_feeder import InputFeeder
from src.model_integrator import Integrator
from src.mouse_controller import MouseController
import time


df = pd.DataFrame([],columns=['FACIAL_PRES','HEAD_PRES','LANDMARK_PRES','GAZE_PRES','LOAD_TIME','INF_TIME','APP_FPS'])

# VARIABLES
FACIAL_MODELS = {
    'FP32-INT1':'intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
}
HEAD_MODELS={
    'FP16':'intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001',
    'FP32':'intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001',
    'FP32-INT8':'intel/head-pose-estimation-adas-0001/FP32-INT8/head-pose-estimation-adas-0001'
}
LANDMARK_MODELS={
    'FP16':'intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009',
    'FP32':'intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009',
    'FP32-INT8':'intel/landmarks-regression-retail-0009/FP32-INT8/landmarks-regression-retail-0009'
}
GAZE_MODELS={
    'FP16':'intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002',
    'FP32':'intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002',
    'FP32-INT8':'intel/gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002'
}

for facial in FACIAL_MODELS:
    for head in HEAD_MODELS:
        for landmark in LANDMARK_MODELS:
            for gaze in GAZE_MODELS:
                models_dict = {'face_det_model': FACIAL_MODELS[facial],
                               'head_pose_estimation_model': HEAD_MODELS[head],
                               'facial_landmarks_detection_model': LANDMARK_MODELS[landmark],
                               'gaze_estimation_model': GAZE_MODELS[gaze]
                               }
                FLAGS = {'face_det_model_show': False,
                         'head_pose_estimation_model_show': False,
                         'facial_landmarks_detection_model_show': False,
                         'gaze_estimation_model_show': False
                         }

                type_input = "video"
                stream_input = "bin/demo.mp4"
                input_feeder = InputFeeder(type_input, stream_input)
                input_feeder.load_data()
                counter_stop = 100

                integrator = Integrator(models_dict, FLAGS, 'CPU', None, 0.7)

                inf_time=[]
                FPS_A =time.time()
                counter=0
                while counter < counter_stop:
                    image = input_feeder.next_batch()
                    if image is None:
                        logging.info('image from feeder is None, video finished or input stream was interrumped!')
                        break
                    inf_a=time.time()
                    integrator.process_image(image)
                    inf_time.append(time.time()-inf_a)
                    counter+=1
                    # cv2.imshow("win",image)
                    # cv2.waitKey(1)
                FPS_B = time.time()

                new_row = {     'FACIAL_PRES': facial
                            ,   'HEAD_PRES': head
                            ,   'LANDMARK_PRES': landmark
                            ,   'GAZE_PRES': gaze
                            ,   'LOAD_TIME': integrator.loading_time
                            ,   'INF_TIME': sum(inf_time)/counter_stop
                            ,   'APP_FPS': counter_stop/(FPS_B-FPS_A)
                         }
                df = df.append(new_row, ignore_index=True)

                input_feeder.close()
df.to_csv("benchmark.csv",index=False)
df = pd.read_csv("benchmark.csv")
df = df.sort_values(by=['INF_TIME'],axis=0,ascending=True)
print(df)
