locfull.npy: getCoordsVideo.py
	python getCoordsVideo.py

traj: plottingScripts/getTraj.py locfull.npy
	python plottingScripts/getTraj.py True

speed: plottingScripts/plotSpeed.py locfull.npy velfull.npy
	python plottingScripts/plotSpeed.py True

trajOut.mp4: locfull.npy
	ffmpeg -framerate 10 -pattern_type glob -i '/data/b1033128/Tracking/throughFenceRL/traj/traj*.png'
	 -c:v libx264 -r 30 -pix_fmt yuv420p -vf scale=1200:1180 '/home/b1033128/Documents/throughFenceRL/traj/trajOut.mp4'

trimImages: /data/b1033128/Tracking/throughFenceRL/0000.png
	for a in [0-9]*.png; do convert -trim "$a" T"$a"; done


clean:
	rm *.pyc

all: clean track traj
