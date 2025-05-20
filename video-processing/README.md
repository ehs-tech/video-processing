curl --location 'http://127.0.0.1:5000/blur_video' \
--header 'Content-Type: application/json' \
--data '{
    "video_url": "https://videos.pexels.com/video-files/4426909/4426909-hd_1280_720_50fps.mp4",
    "output_file_name": "output_video"
}'