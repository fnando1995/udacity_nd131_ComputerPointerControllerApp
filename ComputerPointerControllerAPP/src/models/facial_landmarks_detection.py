from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.extensions = extensions
        self.device = device
        self.model_xml = model_name + '.xml'
        self.model_bin = model_name + '.bin'
        self.landmark_aug=None


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
            exit()
        self.exec_net = self.plugin.load_network(self.net, "CPU")
        self.input_blob = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_blob].shape
        self.height,self.width= self.input_shape[-2:]

    def predict(self, orig_image,flag):
        if self.landmark_aug is None:
            self.landmark_aug=min(orig_image.shape[0],orig_image.shape[1])//10
        preprocessed_image = self.preprocess_input(np.copy(orig_image))
        output = self.exec_net.infer({self.input_blob: preprocessed_image})
        return self.preprocess_output(output,orig_image,flag)

    def preprocess_input(self, image):
        image = cv2.resize(image, (self.width, self.height),interpolation=cv2.INTER_CUBIC)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, self.height, self.width)
        return image

    def preprocess_output(self, outputs,orig_image,flag):
        orig_image_h,orig_image_w=orig_image.shape[:2]
        output = outputs['95'][0]
        landmarks =[]
        for i in range(0, len(output), 2):
            xmin,ymin=( int(output[i][0][0]*orig_image_w)-self.landmark_aug
                       ,int(output[i+1][0][0]*orig_image_h)-self.landmark_aug)
            xmax,ymax=( int(output[i][0][0]*orig_image_w)+self.landmark_aug
                       ,int(output[i+1][0][0]*orig_image_h)+self.landmark_aug)
            xmin = 0 if xmin < 0 else xmin
            ymin = 0 if ymin < 0 else ymin
            xmax = 0 if xmax < 0 else xmax
            ymax = 0 if ymax < 0 else ymax
            landmarks.append([(xmin,ymin),(xmax,ymax)])
        for i in range(len(landmarks)):
            if flag:
                cv2.rectangle(orig_image, landmarks[i][0], landmarks[i][1], (0, 255, 0), 1)
            # changing coord for rois in orig image
            landmarks[i]=orig_image[landmarks[i][0][1]:landmarks[i][1][1],landmarks[i][0][0]:landmarks[i][1][0],:]
        return landmarks[:2]