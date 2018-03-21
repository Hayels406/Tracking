locfull.npy: getCoordsVideo.py
	python getCoordsVideo.py

traj: plottingScripts/getTraj.py locfull.npy
	python plottingScripts/getTraj.py True

speed: plottingScripts/plotSpeed.py locfull.npy velfull.npy
	python plottingScripts/plotSpeed.py True

trajOut.mp4: locfull.npy
	ffmpeg -framerate 10 -pattern_type glob -i '/home/b1033128/Documents/throughFenceRL/traj*.png'
	 -c:v libx264 -r 30 -pix_fmt yuv420p -vf scale=1200:1180 '/home/b1033128/Documents/throughFenceRL/trajOut.mp4'

clean:
	rm *.pyc

all: clean track traj
