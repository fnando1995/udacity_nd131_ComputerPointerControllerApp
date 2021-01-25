'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np


class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None,det_thresh=0.7):
        self.extensions = extensions
        self.device = device
        self.model_xml = model_name + '.xml'
        self.model_bin = model_name + '.bin'
        self.face_threshold = det_thresh

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
        self.input_blob = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_blob].shape
        self.height,self.width= self.input_shape[-2:]

    def predict(self, orig_image,flag):
        preprocessed_image = self.preprocess_input(np.copy(orig_image))
        output = self.exec_net.infer({self.input_blob: preprocessed_image})
        return self.preprocess_output(output,orig_image,flag)


    def preprocess_input(self, image):
        #[BxCxHxW]
        #[1x3x384x672]
        image = cv2.resize(image, (self.width, self.height),interpolation=cv2.INTER_CUBIC)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, self.height, self.width)
        return image


    def preprocess_output(self, outputs,orig_image,flag):
        orig_image_height,orig_image_width = orig_image.shape[:2]
        roi_faces = []
        for d in outputs['detection_out'][0][0]:
            # d = [image_id, label, conf, x_min, y_min, x_max, y_max]
            if d[0]==-1:
                break
            if d[2]<self.face_threshold:
                continue
            # [x1,y1,x2,y2,conf]
            roi_faces.append([    int(d[3]*orig_image_width)
                            ,int(d[4]*orig_image_height)
                            ,int(d[5]*orig_image_width)
                            ,int(d[6]*orig_image_height)
                            ,d[2]])
        if flag:
            for det in roi_faces:
                x1, y1, x2, y2 = det[:4]
                cv2.rectangle(orig_image,(x1, y1), (x2, y2) , (0, 0, 255), 1)
        return roi_faces
