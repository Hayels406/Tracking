track: getCoordsVideo.py
	python getCoordsVideo.py

traj: plottingScripts/getTraj.py locfull.npy
	python plottingScripts/getTraj.py True

speed: plottingScripts/plotSpeed.py locfull.npy velfull.npy
	python plottingScripts/plotSpeed.py True

clean:
	rm *.pyc

all: clean track traj
