import sys
from io import BytesIO
import os
import paddle
 
# First, delete torchcodec from sys.modules if it exists
if 'torchcodec' in sys.modules:
    del sys.modules['torchcodec']
 
# Enable torch proxy
paddle.compat.enable_torch_proxy(scope={'torchcodec'})
 
# Now try to import torchcodec
from torchcodec.decoders import VideoDecoder
 
video_path = '../utils/Shufflers/dataset_videos/draw.mp4'
 
with open(video_path, 'rb') as f:
    res = BytesIO(f.read())
 
decoder = VideoDecoder(res, num_ffmpeg_threads=0)
 
# Output decoder metadata
print('=== Decoder Metadata ===')
print(f'average_fps: {decoder.metadata.average_fps}')
 
video_fps = decoder.metadata.average_fps
idx = [0, 18, 36, 53, 71, 89, 107, 125, 143, 160, 178, 196]
video = decoder.get_frames_at(indices=idx).data
 
print(f'\n=== Video Result ===')
print(f'video shape: {video.shape}')
print(f'video dtype: {video.dtype}')
print(f'video device: {video.device}')
print(f'video: {video}')

import hashlib
import math

np_video = video.detach().cpu().numpy()
array_bytes = np_video.tobytes()
data_md5 = hashlib.md5(array_bytes).hexdigest()

print(f'video md5: {data_md5}')
 
paddle.compat.disable_torch_proxy()
print('\nTest completed successfully!')