import click
import cv2
import onnxruntime as rt
import time
from typing import Tuple, List

from pathlib import Path
import yaml
import numpy as np

from PUTDriver import PUTDriver, gstreamer_pipeline


class AI:
    def __init__(self, config: dict, forward_initial: float):
        self.path = config['model']['path']

        self.sess = rt.InferenceSession(self.path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

        self.buffer_size = config['robot']['buffer_size']
        self.buffer = np.zeros((self.buffer_size, 2))
        self.buffer[0] += forward_initial

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        return img

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        detections = np.clip(detections, -1.0, 1.0)
        return detections

    def update_buffer(self, steering_signal: Tuple[float, float]) -> None:
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = steering_signal

    def calculate_steering(self) -> Tuple[float, float]:
        # Calculate as the exponential moving average of the last buffer_size signals
        weights = np.exp(np.arange(self.buffer_size)) # Exponential weights, the latest signal has the highest weight
        weights /= weights.sum()
        weights = np.expand_dims(weights, 1)

        steering_signal = (self.buffer * weights).sum(axis=0)

        return steering_signal

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)

        assert inputs.dtype == np.float32
        assert inputs.shape == (1, 3, 224, 224)
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)
        if self.buffer_size > 1:
            self.update_buffer(outputs)
            steering_signal = self.calculate_steering().astype(np.float32)
        else:
            steering_signal = outputs[0]  # NOTE: batch dim.

        assert steering_signal.dtype == np.float32
        assert steering_signal.shape == (2,)
        assert steering_signal.max() <= 1.0
        assert steering_signal.min() >= -1.0

        return steering_signal


@click.command()
@click.option("--record", is_flag=True, help="TODO")
# prompt=True,  # click will ask interactively
@click.option("--start-forward", type=float, default=0.0, help="Forward value when robot starts racing.")
@click.option("-m", "--model", help="Override model name, ignoring config.yml")
@click.option("-b", "--bufsz", type=int, help="Override buffer size for momentum, ignoring config.yml")
def main(record: bool, start_forward: float, model, bufsz: int):
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # print("[debug]: args:", record, start_forward, model, bufsz)
    if model:
        if not Path(model).exists():
            print(f"[err] file {model} does not exist")
        else:
            config["model"]["path"] = model
            print(f"[ok] Using model {model}")

    if bufsz:
        config["robot"]["buffer_size"] = bufsz
        print(f"[ok] Using buffer size {bufsz}")

    # print(config)
    # return

    driver = PUTDriver(config=config)
    ai = AI(config=config, forward_initial=start_forward)

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, display_width=224, display_height=224,
        framerate=15,
    ), cv2.CAP_GSTREAMER)

    # model warm-up
    ret, image = video_capture.read()
    if not ret:
        print('No camera')
        return

    _ = ai.predict(image)

    # Longer camera and model warm-up
    WARMUP_FRAMES = 30
    print("Warming up...")
    warmup_tic = time.time()
    for _n in range(WARMUP_FRAMES):
        ret, image = video_capture.read()
        if not ret:
            print('No camera')
            return
        forward, left = ai.predict(image)
        # print(f" [warmup] Predicted {forward:.4f}\t{left:.4f}")
    warmup_tac = time.time()
    print(f" [ok] Took {warmup_tac - warmup_tic} seconds "
          f"({((warmup_tac - warmup_tic) * 1000 / WARMUP_FRAMES):.3f} ms per frame)")

    input('Robot is ready to ride. Press Enter to start...')

    report_times = True  # TODO: config (?)

    forward, left = 0.0, 0.0
    while True:
        print(f'Forward: {forward:.4f}\tLeft: {left:.4f}')
        driver.update(forward, left)

        tic = time.time()
        ret, image = video_capture.read()
        tac = time.time()
        if report_times and (tac - tic) > 0.001:
            # example: Frame capture took 0.000158 seconds
            print(f"Frame capture took {tac - tic} seconds")

        if not ret:
            print('No camera')
            break
        forward, left = ai.predict(image)

        tic2 = time.time()
        if report_times:
            print(f"Prediction took {tic2 - tac} seconds")


if __name__ == '__main__':
    main()
