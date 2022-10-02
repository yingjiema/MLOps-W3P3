<p align = "center" draggable=”false”
   ><img src="https://user-images.githubusercontent.com/37101144/161836199-fdb0219d-0361-4988-bf26-48b0fad160a3.png"
     width="200px"
     height="auto"/>
</p>



# <h1 align="center" id="heading">Part 3 - Deploying a Face Segmentation Background Changer on AWS EC2 using FastAPI</h1>


Image segmentation has a lot of amazing applications that solve different computer vision problems. Image segmentation is, essentially, a classification task in which we classify each pixel as belonging to one of the target classes. So when you pass an image through a segmentation model, it will give one label to each of the pixels that present in the image. For this assignment, we'll be checking out background segmentation and replacement of those pixels. Background segmentation is a process in which an algorithm removes the static background from an image. This allows only changing a section of the image. This process is important for motion detection or object tracking.

In today's session, we'll be taking a look at face detection and background changer algorithm. Our goal will be to detect the faces in an image and replace the background while retaining those faces. To do this, we'll be utilizing DeepLab!

DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., person, dog, cat and so on) to every pixel in the input image. By using this image segmentation, we can separate the face in the foreground from the background. You can find the code in `deeplab.py`. We will then use a crawling Google image search to replace the original background with a background that matches our query.

We are going to deploy a [pretrained image segmentation model](https://github.com/tensorflow/models/tree/master/research/deeplab).

## Create EC2 Instance

- Go to your EC2 console
- Create an EC2 instance
- Pick `Amazon Linux`
- Pick instance type: At least `t3.medium` up to `t2.xlarge`
- Create key-pair and download key (if you haven't already created one)
- Edit network to enable the following...
    - Enable public IPV4 address
    - Open HTTP, HTTPS, SSH, and port 8000. All should be open from anywhere
- Set drive size to 20GB
- Launch Instance

## Install Dependencies
- Get the IP address of the instance
- Change key permissions
- SSH into the machine
- Install git if needed (`sudo apt install git` for Ubuntu based distros, `sudo yum install git` for Amazon Linux)
- Install pip if needed (`sudo apt install python3-pip` for Ubuntu based distros, `sudo yum install python3-pip` for Amazon Linux)
- Clone the repo (`git clone ...`)
- If there's permission issues with gitlab, generate ssh keys (`ssh-keygen`) and add them to github account
- CD into the folder (`cd cloned-repo`)
- Install the requirements (`pip3 install -r requirements.txt`)

## Run API
- Run the API (`uvicorn app:app --host 0.0.0.0 --port 8000`)
- Create a request with docs (http://ec2.ip.address:8000/docs)

# Apple Silicone

## Run API locally

Read this section if you are on an M1 chip ( arm64 / silicon; in theory it should work on M2 as well, yet to be tested ) and would like to launch the API locally. This part has been tested on an Apple M1 chip, macOS Monterey 12.6 machine.

## Install Tensorflow for M1
Follow instructions step by step from [Metal: Getting Started with tensorflow-metal PluggableDevice](https://developer.apple.com/metal/tensorflow-plugin/), please read the NOTEs carefully and apply them accordingly.

After you install `miniforge3`, you can use it to create a new environment, say `tf-metal` and then install the `tensorflow-deps` to the environment.

After following these three steps, make sure in your terminal to check if tensorflow is installed correctly by running

```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

<details>
<summary> Example Output

    Metal device set to: Apple M1

    systemMemory: 8.00 GB
    maxCacheSize: 2.67 GB

    2022-09-28 20:46:17.797673: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
    2022-09-28 20:46:17.797813: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
    tf.Tensor(-692.91614, shape=(), dtype=float32)

</summary>
</details>

If this is NOT successful ( symptoms: your python crashes ), please reach out to your instructors.

## Install dependencies

1. install the dependencies from `requirements-notf.txt`

    ```
    python -m pip install -r requirements-notf.txt
    ```

    You will not need `tensorflow` here, since it has been installed in the previous step.

2. Troubleshooting:
    Fire up a Python kernel from terminal and simply import the following packages as shown in `deeplab.py` to make sure they are installed correctly.

    ```
    import cv2
    import tensorflow as tf
    ```

    a. Another difference you notice in `requirements-notf.txt` is that we are using `opencv-python` instead. This is to avoid a conflict in `numpy` versions: if you see an `ImportError: numpy.core.multiarray failed to import`, that's due to the conflict and installing the correct version of `opencv-python` should fix it.

    b. Another possible error you might see is `AssertionError: Duplicate registrations for type 'experimentalOptimizer'`, not sure about the exact problem, yet uninstall `keras` shall solve it:

        ```
        python -m pip uninstall keras
        ```
5. Run the API locally

    ```
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```

    Your API will be running at `http://localhost:8000/docs`
