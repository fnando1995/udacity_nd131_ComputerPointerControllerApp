from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
import math

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.extensions = extensions
        self.device = device
        self.model_xml = model_name + '.xml'
        self.model_bin = model_name + '.bin'


    def load_model(self):
        self.plugin = IECore()
        if self.extensions:
            try:
                self.plugin.add_extension(self.extensions, self.device)
            except Exception as e:
                raise ValueError("Problem at extensions loading, is the extension path for the device {} ok?".format(self.device))
        try:
            self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        except Exception as e:
            raise ValueError("Net would not initialize, is the model path ok?")
        supported_layers = self.plugin.query_network(network=self.net, device_name="CPU")
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        self.exec_net = self.plugin.load_network(self.net, "CPU")
        self.input_names = [i for i in self.net.inputs.keys()]
        self.eyes_input_shape = self.net.inputs[self.input_names[1]].shape
        self.eyes_height,self.eyes_width = self.eyes_input_shape[-2:]


    def predict(self, left_eye_orig_image,right_eye_orig_image,angles):
        left_eye_proc_image, \
        right_eye_proc_image = self.preprocess_input(np.copy(left_eye_orig_image),
                                                     np.copy(right_eye_orig_image))
        outputs = self.exec_net.infer({'head_pose_angles': angles,
                                       'left_eye_image': left_eye_proc_image,
                                       'right_eye_image': right_eye_proc_image})
        return self.preprocess_output(outputs, angles)


    def preprocess_input(self, left_eye_image,right_eye_img):
        try:
            self.left_eye_image = cv2.resize(left_eye_image, (self.eyes_width, self.eyes_height))
            self.right_eye_img = cv2.resize(right_eye_img, (self.eyes_width, self.eyes_height))
            self.left_eye_image = self.left_eye_image.transpose((2, 0, 1))
            self.right_eye_img = self.right_eye_img.transpose((2, 0, 1))
            self.left_eye_image = self.left_eye_image.reshape(1, 3, self.eyes_height, self.eyes_width)
            self.right_eye_img = self.right_eye_img.reshape(1, 3, self.eyes_height, self.eyes_width)
            return self.left_eye_image, self.right_eye_img
        except:
            print('error at gaze preprocess')
            print(left_eye_image.shape,right_eye_img.shape)


    def preprocess_output(self, outputs, angle):
        gaze_vector = outputs['gaze_vector'][0]
        roll = angle[2]
        val_cos = math.cos(roll * math.pi / 180.0)
        val_sin = math.sin(roll * math.pi / 180.0)
        mouse_x = gaze_vector[0] * val_cos + gaze_vector[1] * val_sin
        mouse_y = -gaze_vector[0] * val_sin + gaze_vector[1] * val_cos
        return mouse_x, mouse_y
