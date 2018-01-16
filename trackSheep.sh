rm ~/Documents/throughFenceRL/*.*
python -b getCoordsVideo.py
python -b getTraj.py
p=`pwd`
cd ~/Documents/throughFenceRL
ffmpeg -framerate 10 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
cd `echo ${p}`