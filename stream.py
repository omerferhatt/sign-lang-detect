from flask_opencv_streamer.streamer import Streamer
from rasp_inference import TSLReader

port = 3030
require_login = False
streamer = Streamer(port, require_login, stream_res=(640, 480))
streamer.thread = 4

reader = TSLReader(
    labels_txt='data/labels.txt',
    model_path='saved_models/mobilenetv3_s_prune_quant_default_acc_90.tflite',
    camera_device=0
)
# reader.set_bbox()

while True:
    frame = reader(stream=True, show=False, threshold=0.35)

    streamer.update_frame(frame)

    if not streamer.is_streaming:
        streamer.start_streaming()
