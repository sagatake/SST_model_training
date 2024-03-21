#Updated on 2022年  2月 17日 木曜日 20:42:00 JST

#please run this on /home/takeshi-s using docker(alaebr/openface)
#To activate docker container, run "docker run -v /Users/takeshi-s:/home/takeshi-s -it algebr/openface:latest"
#Then please change directory (cd) to this place,
# run this by "bash ./openface_runner.sh"

#eval 'python3 ./video_crop/0_2_video_cropping.py'

echo 'now : '$1
eval '/home/openface-build/build/bin/FeatureExtraction -f '$1' -out_dir '$2
#eval '../openface-build/build/bin/FeatureExtraction -f ./Documents/my_research/experiment/1_visual_feature/video_crop/output.avi'
