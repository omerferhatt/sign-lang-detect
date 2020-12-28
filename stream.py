from flask_opencv_streamer.streamer import Streamer
from rasp_inference import TSLReader

port = 3030
require_login = False
streamer = Streamer(port, require_login)

reader = TSLReader(
    labels_txt='labels.txt',
    model_path='saved_models/model_mobilenetv3_aug_86.tflite',
    camera_device=0
)
# reader.set_bbox()

while True:
    frame = reader(stream=True, show=False, threshold=0.5)

    streamer.update_frame(frame)

    if not streamer.is_streaming:
        streamer.start_streaming()
