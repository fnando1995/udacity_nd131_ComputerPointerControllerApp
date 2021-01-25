import logging
import cv2
from argparse import ArgumentParser
from src.input_feeder import InputFeeder
from src.model_integrator import Integrator
from src.mouse_controller import MouseController


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("--facedetectionmodel", required=True, type=str
                        , default='intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
                        , help="path to face detection model file with no extension.")
    parser.add_argument("--facedetectionmodel_show", required=False, type=bool
                        , default=False
                        , help="FLAG for showing output from face detection model")
    parser.add_argument("--faciallandmarkmodel", required=True, type=str
                        , default ='intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
                        , help="path to facial landmark detection model file with no extension.")
    parser.add_argument("--faciallandmarkmodel_show", required=False, type=bool
                        , default=False
                        , help="FLAG for showing output from facial landmark detection model")
    parser.add_argument("--headposeestimationmodel", required=True, type=str
                        , default='intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'
                        , help="path to head pose estimation model file with no extension.")
    parser.add_argument("--headposeestimationmodel_show", required=False, type=bool
                        , default=False
                        , help="FLAG for showing output from head pose estimation model")
    parser.add_argument("--gazeestimationmodel", required=True, type=str
                        , default='intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002'
                        , help="path to gaze estimation model file with no extension.")
    parser.add_argument("--gazeestimationmodel_show", required=False, type=bool
                        , default=False
                        , help="FLAG for showing output from gaze estimation model")
    parser.add_argument("--input", required=True, type=str
                        , default = "bin/demo.mp4"
                        , help="path to video file, 'cam' for webcam-0")
    parser.add_argument("--counter", required=True, type=int,
                        default=10,
                        help="number of frames spared when for making the mouse movement")
    parser.add_argument("--cpu_extension", required=False, type=str,
                        default=None,
                        help="extesion full path if any, CPU only allowed")
    parser.add_argument("--device", required=False,default="CPU", type=str,
                        help="device to use for inference",choices=['CPU', 'GPU', 'FPGA', 'VPU'])
    parser.add_argument("--prob_threshold", required=False, type=float, default=0.7,
                        help="probability for model detections")
    parser.add_argument("--mouseprecision",required=True,type=str,default="high"
                        ,help="precision for the mouse",choices=['high', 'medium', 'low'])
    parser.add_argument("--mousespeed",required=True,type=str,default="fast"
                        ,help="Speed of the mouse when moving",choices=['fast', 'medium', 'slow'])
    return parser


def main():
    # Manage argsparser

    args = build_argparser().parse_args()
    models_dict = {'face_det_model':args.facedetectionmodel,
                   'head_pose_estimation_model':args.headposeestimationmodel,
                   'facial_landmarks_detection_model':args.faciallandmarkmodel,
                   'gaze_estimation_model':args.gazeestimationmodel
                    }
    FLAGS =       {'face_det_model_show':args.facedetectionmodel_show,
                   'head_pose_estimation_model_show':args.headposeestimationmodel_show,
                   'facial_landmarks_detection_model_show':args.faciallandmarkmodel_show,
                   'gaze_estimation_model_show':args.gazeestimationmodel_show
                    }
    integrator = Integrator(models_dict,FLAGS,args.device,args.cpu_extension,args.prob_threshold)
    mouse = MouseController(args.mouseprecision
                            ,args.mousespeed)
    type_input   = "cam" if args.input=="cam" else "video"
    stream_input = 0     if args.input=="cam" else args.input
    input_feeder = InputFeeder(type_input,stream_input)
    input_feeder.load_data()
    active_mouse_val = args.counter

    # start app

    counter=0
    while True:
        image = input_feeder.next_batch()
        if image is None:
            logging.info('image from feeder is None, video finished or input stream was interrumped!')
            break
        value = integrator.process_image(image)
        if value is None:
            logging.info('going to next batch')
            continue
        x,y=value
        counter+=1
        if counter==active_mouse_val:
            mouse.move(x,y)
            counter=0
        cv2.imshow("win", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Execution terminated by pressing 'q' key")
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
