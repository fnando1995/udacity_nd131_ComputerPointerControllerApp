'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

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
        self.input_blob = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_blob].shape
        self.height,self.width= self.input_shape[-2:]

    def predict(self, orig_image,flag):
        preprocessed_image = self.preprocess_input(np.copy(orig_image))
        output = self.exec_net.infer({self.input_blob: preprocessed_image})
        return self.preprocess_output(output,orig_image,flag)


    def preprocess_input(self, image):
        #[1xCxHxW]
        #[1x3x60x60]
        image = cv2.resize(image, (self.width, self.height),interpolation=cv2.INTER_CUBIC)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, self.height, self.width)
        return image

    def preprocess_output(self, hpResults, orig_image,flag):
        yaw = hpResults['angle_y_fc'][0][0]
        pitch = hpResults['angle_p_fc'][0][0]
        roll = hpResults['angle_r_fc'][0][0]
        if flag:
            def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
                def build_camera_matrix(center_of_face, focal_length):
                    cx = int(center_of_face[0])
                    cy = int(center_of_face[1])
                    camera_matrix = np.zeros((3, 3), dtype='float32')
                    camera_matrix[0][0] = focal_length
                    camera_matrix[0][2] = cx
                    camera_matrix[1][1] = focal_length
                    camera_matrix[1][2] = cy
                    camera_matrix[2][2] = 1
                    return camera_matrix
                yaw *= np.pi / 180.0
                pitch *= np.pi / 180.0
                roll *= np.pi / 180.0
                cx = int(center_of_face[0])
                cy = int(center_of_face[1])
                Rx = np.array([[1, 0, 0],
                               [0, math.cos(pitch), -math.sin(pitch)],
                               [0, math.sin(pitch), math.cos(pitch)]])
                Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                               [0, 1, 0],
                               [math.sin(yaw), 0, math.cos(yaw)]])
                Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                               [math.sin(roll), math.cos(roll), 0],
                               [0, 0, 1]])
                # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
                R = Rz @ Ry @ Rx
                camera_matrix = build_camera_matrix(center_of_face, focal_length)
                xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
                yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
                zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
                zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
                o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
                o[2] = camera_matrix[0][0]
                xaxis = np.dot(R, xaxis) + o
                yaxis = np.dot(R, yaxis) + o
                zaxis = np.dot(R, zaxis) + o
                zaxis1 = np.dot(R, zaxis1) + o
                xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
                yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
                p2 = (int(xp2), int(yp2))
                cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
                xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
                yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
                p2 = (int(xp2), int(yp2))
                cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
                xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
                yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
                p1 = (int(xp1), int(yp1))
                xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
                yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
                p2 = (int(xp2), int(yp2))
                cv2.line(frame, p1, p2, (255, 0, 0), 2)
                cv2.circle(frame, p2, 3, (255, 0, 0), 2)
                return frame
            focal_length = 950.0
            scale = 300
            frame_h, frame_w = orig_image.shape[:2]
            center_of_face = (frame_w / 2, frame_h / 2, 0)
            draw_axes(orig_image, center_of_face, yaw, pitch, roll, scale, focal_length)
        return [yaw,pitch,roll]
