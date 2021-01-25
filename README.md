# Computer Pointer Controller

This repo is about the third project for Intel's Edge AI Nano degree at Udacity.

The project refers to an application that given an input video/cam_stream 
it controls the mouse pointer of the local computer with the direction of the
head and eyes.

The action is done every `counter` number of frame, and has the direction of the
first face detected.

As the project involves 4 different models (see image below), I chose
to create a class that manage the 4 models called `Integrator`. This class holds 
the loading and prediction of each model and creates the necessary pipeline 
between outputs and inputs of each model.

![pipeline](bin/pipeline.png)

## Project Set Up and Installation

### Create and activate the environment.

To use the project, I recommend using a virtual environment. This could be done 
with the following commands:

```
$ cd $PATH
$ python3 -m venv environment
$ source environment/bin/activate
(environment) $ pip install -r requirements.txt
```

- `$PATH` refers to the full pat where the project has been unzipped.
- `venv` is a python module which is helpfull for creating virtual environments
. (sudo apt-get install python3.venv).
- `source` command help to activate the virtual environment. After this command 
you will notice the command-line has changed ,appearing the name of the virtual
 environment name on it.
- `pip` python module helfull for installing new packages.

### Downloading model networks

I recommend to use openvino downloader module:

```
cd $PATH
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py \ 
--name 'NAME_OF_MODEL'
```

here you can change `'NAME_OF_MODEL'` with the following model names:

- [face-detection-adas-binary-0001](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html):  
- [landmarks-regression-retail-0009](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [head-pose-estimation-adas-0001](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [gaze-estimation-adas-0002](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)


All models will download under the `$PATH/intel` directory. except if you add the 
`-o $NEWDIR` argument to the download command.

## Demo


### How to run

The project could has arguments that can be seen with `python main.py --help` command.

```
usage: main.py [-h] --facedetectionmodel FACEDETECTIONMODEL
               [--facedetectionmodel_show FACEDETECTIONMODEL_SHOW]
               --faciallandmarkmodel FACIALLANDMARKMODEL
               [--faciallandmarkmodel_show FACIALLANDMARKMODEL_SHOW]
               --headposeestimationmodel HEADPOSEESTIMATIONMODEL
               [--headposeestimationmodel_show HEADPOSEESTIMATIONMODEL_SHOW]
               --gazeestimationmodel GAZEESTIMATIONMODEL
               [--gazeestimationmodel_show GAZEESTIMATIONMODEL_SHOW] --input
               INPUT --counter COUNTER [--cpu_extension CPU_EXTENSION]
               [--device {CPU,GPU,FPGA,VPU}] [--prob_threshold PROB_THRESHOLD]
               --mouseprecision {high,medium,low} --mousespeed
               {fast,medium,slow}

optional arguments:
  -h, --help            show this help message and exit
  --facedetectionmodel FACEDETECTIONMODEL
                        path to face detection model file with no extension.
  --facedetectionmodel_show FACEDETECTIONMODEL_SHOW
                        FLAG for showing output from face detection model
  --faciallandmarkmodel FACIALLANDMARKMODEL
                        path to facial landmark detection model file with no
                        extension.
  --faciallandmarkmodel_show FACIALLANDMARKMODEL_SHOW
                        FLAG for showing output from facial landmark detection
                        model
  --headposeestimationmodel HEADPOSEESTIMATIONMODEL
                        path to head pose estimation model file with no
                        extension.
  --headposeestimationmodel_show HEADPOSEESTIMATIONMODEL_SHOW
                        FLAG for showing output from head pose estimation
                        model
  --gazeestimationmodel GAZEESTIMATIONMODEL
                        path to gaze estimation model file with no extension.
  --gazeestimationmodel_show GAZEESTIMATIONMODEL_SHOW
                        FLAG for showing output from gaze estimation model
  --input INPUT         path to video file, 'cam' for webcam-0
  --counter COUNTER     number of frames spared when for making the mouse
                        movement
  --cpu_extension CPU_EXTENSION
                        extesion full path if any, CPU only allowed
  --device {CPU,GPU,FPGA,VPU}
                        device to use for inference
  --prob_threshold PROB_THRESHOLD
                        probability for model detections
  --mouseprecision {high,medium,low}
                        precision for the mouse
  --mousespeed {fast,medium,slow}
                        Speed of the mouse when moving

```


This usage explain each argument used for the `main.py` file. I recommend the following
command just to test it on CPU with FP32 models.

```
python3 main.py \
--facedetectionmodel intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 \
--faciallandmarkmodel intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 \
--headposeestimationmodel intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 \
--gazeestimationmodel intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 \
--input bin/demo.mp4 \
--counter 10 \
--prob_threshold 0.7 \
--mouseprecision high \
--mousespeed fast \
--facedetectionmodel_show True \
--headposeestimationmodel_show True \
--faciallandmarkmodel_show True \
--gazeestimationmodel_show True
```

## Benchmarks

I create a file `benchmark.py` which benchmark all different model's precisions. below
there is a print of the final dataframe order ascending by column INF_TIME.

![benchmark](bin/benchmark.png)

This benchmark was made over the first `100` frames of de `bin/demo.mp4` video.

## Results

It is weird to see that FP16 precision is slower than FP32 in making a full inference (
this means over the 4 models), I suppose this is due to FP32 optimization for CPU's.

Loading time is as expected, FP16 is faster to load that FP32 and even faster than
 FP32-INT8.
 
`benchmark.py` code is time-measured mostly over the effective time. APP_FPS was 
considered only for app real pipeline (no loading time).

### Edge Cases
Face control seems to be really near to real effectiness. I can wait to see this uses cases:

- Manufacturing machines control with facial movements to avoid contact with surfaces.
- Facial movements to talk for voiceless people.

 
