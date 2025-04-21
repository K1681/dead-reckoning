import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

def main():
    # File names
    infile_name = "raw.csv"
    outfile_name = "raw.gif"

    # Read the no. of lines in inflie
    with open(infile_name, "rb") as infile:
        num_lines = sum(1 for _ in infile)
        infile.close()

    # Read the infile
    positions = np.zeros((num_lines, 3))
    with open(infile_name, 'r') as infile:
        infile_csv = csv.reader(infile)
        for i in range(num_lines):
            line = next(infile_csv)
            positions[i][0] = float(line[3].strip())
            positions[i][1] = float(line[5].strip())
            positions[i][2] = float(line[7].strip())

    # Animation
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    mag = max(abs(np.min(positions)), abs(np.max(positions)))
    ax.set_xlim([-mag, mag])
    ax.set_ylim([-mag, mag])
    ax.set_zlim([-mag, mag])

    x, y, z = np.array([[5, 0, 0],[0, 5, 0],[0, 0, 5]])
    u, v, w = np.array([[mag, 0, 0],[0, mag, 0],[0, 0, mag]])
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")

    ax.view_init(10, -60)
    animated_plot, = ax.plot3D([], [], [], 'r')
    ax.set_title("Dead Reackoning")
    ax.set_xlabel("X", labelpad=20)
    ax.set_ylabel("Y", labelpad=20)
    ax.set_zlabel("Z", labelpad=20)
    ax.grid()
    def update(frame):
        animated_plot.set_data(positions[:, 0][:frame], positions[:, 1][:frame])
        animated_plot.set_3d_properties(positions[:, 2][:frame])
    animation = FuncAnimation(fig=fig, func=update, frames=range(0, num_lines+1, 10), interval=20)
    animation.save(outfile_name)
    plt.show()

if __name__ == "__main__":
    main()
