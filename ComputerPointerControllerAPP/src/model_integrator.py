import logging
import time
from src.models.head_pose_estimation import Model_X as hpe
from src.models.face_detection import Model_X as fd
from src.models.facial_landmarks_detection import Model_X as fl
from src.models.gaze_estimation import Model_X as ge

class Integrator():
    def __init__(self,models_dict,FLAGS,device='CPU',extensions=None,prob=0.7):
        self.model_fd = fd(models_dict['face_det_model'],device,extensions,prob)
        self.fd_flag = FLAGS['face_det_model_show']
        self.model_hpe = hpe(models_dict['head_pose_estimation_model'],device,extensions)
        self.hpe_flag = FLAGS['head_pose_estimation_model_show']
        self.model_fl = fl(models_dict['facial_landmarks_detection_model'],device,extensions)
        self.fl_flag = FLAGS['facial_landmarks_detection_model_show']
        self.model_ge = ge(models_dict['gaze_estimation_model'], device,extensions)
        self.ge_flag = FLAGS['gaze_estimation_model_show']
        self.loading_time = time.time()
        self.model_fd.load_model()
        self.model_hpe.load_model()
        self.model_fl.load_model()
        self.model_ge.load_model()
        self.loading_time = time.time()-self.loading_time

    def process_image(self,orig_image):
        mouse_x,mouse_y=None,None
        roi_faces = self.model_fd.predict(orig_image,self.fd_flag)
        if len(roi_faces)==0:
            logging.info('no faces detected')
            return None
        first = True
        for roi in roi_faces:
            x1,y1,x2,y2 = roi[:4]
            roi = orig_image[y1:y2,x1:x2,:]
            angles = self.model_hpe.predict(roi,self.hpe_flag)
            (roi_left_eye_orig_image,roi_right_eye_orig_image)  = self.model_fl.predict(roi,self.fl_flag)
            if (roi_left_eye_orig_image is None) or (roi_right_eye_orig_image is None):
                logging.info('One or both eyes not detected')
                return None
            # Only first face detected
            if first:
                first=False
                mouse_x,mouse_y = self.model_ge.predict(roi_left_eye_orig_image,roi_right_eye_orig_image,angles)
        if (mouse_x is None) or (mouse_y is None):
            logging.info('something wrong happened with mouse coords in model_ge predict')
            return None
        if self.ge_flag:
            print(mouse_x, mouse_y)
        return mouse_x,mouse_y


