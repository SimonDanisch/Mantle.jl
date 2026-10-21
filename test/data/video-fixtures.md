Synthetic `testsrc2`, 192×192, 12 fps, 24 frames, closed GOPs of eight frames.
The raw references are FFmpeg-decoded NV12; native tests compare every byte.

```sh
ffmpeg -f lavfi -i testsrc2=size=192x192:rate=12:duration=2 -c:v libx264 -pix_fmt yuv420p -x264-params threads=1:keyint=8:min-keyint=8:scenecut=0:open-gop=0:bframes=0:ref=3 -f h264 h264_type2_test.h264
ffmpeg -f lavfi -i testsrc2=size=192x192:rate=12:duration=2 -c:v libx265 -pix_fmt yuv420p -x265-params pools=1:frame-threads=1:keyint=8:min-keyint=8:scenecut=0:open-gop=0:bframes=2:ref=3 -f hevc h265_decode_test.h265
ffmpeg -i h264_type2_test.h264 -pix_fmt nv12 -fps_mode passthrough -f rawvideo h264_type2_test.nv12
ffmpeg -i h265_decode_test.h265 -pix_fmt nv12 -fps_mode passthrough -f rawvideo h265_decode_test.nv12
```
