import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import csv
import os

def RotationMatrix(axis, theta):
    a = axis[0]
    b = axis[1]
    c = axis[2]
    st = math.sin(theta)
    ct = math.cos(theta)
    mag = math.sqrt(a*a+b*b+c*c)
    matrix = np.array([[a*a+b*b*ct+c*c*ct  , a*b*ct-a*b-c*st*mag , a*c*ct-a*c+b*st*mag ],
                       [a*b*ct-a*b+c*st*mag, a*a*ct+b*b+c*c*ct   , -b*c*ct+b*c+a*st*mag],
                       [a*c*ct-a*c-b*st*mag, -b*c*ct+b*c-a*st*mag, a*a*ct+b*b*ct+c*c   ]])
    return matrix/(a*a+b*b+c*c)

def RotateFrame(body_frame, theta):
    for i in [0, 1, 2]:  # Apply roll, then pitch, then yaw
        body_frame = body_frame@RotationMatrix(body_frame[:,0], theta[0])
    return body_frame

def DeadReckoning_raw(infile_name, outfile_name):
    with open(infile_name, 'r') as infile, open(outfile_name, 'w') as outfile:
        # easy to read csv files
        infile_csv = csv.reader(infile)
        next(infile_csv)  # skip the headings

        # Initialize states
        T = 0
        S = np.zeros(3)
        V = np.zeros(3)
        A = np.zeros(3)
        BF = np.identity(3)

        # Initialize the output file
        outfile.write(f"Time, {T}, Pos_X, {S[0]}, Pos_Y, {S[1]}, Pos_Z, {S[2]}, BFx, {BF[0][0]}, {BF[0][1]}, {BF[0][2]}, \
                        BFy, {BF[1][0]}, {BF[1][1]}, {BF[1][2]}, BFz, {BF[2][0]}, {BF[2][1]}, {BF[2][2]}\n")

        for line in infile_csv:
            # Read sensor inputs
            t = float(line[0].strip())
            a = np.array([float(line[1].strip()), float(line[2].strip()), float(line[3].strip())])
            w = np.array([float(line[4].strip()), float(line[5].strip()), float(line[6].strip())])

            # Temp variables
            dt = T - t

            # Update states
            T = t
            BF = RotateFrame(BF, w*dt)
            A = BF@a
            S += V*dt + (A*(dt**2))/2
            V += A*dt

            # Log results
            outfile.write(f"Time, {T}, Pos_X, {S[0]}, Pos_Y, {S[1]}, Pos_Z, {S[2]}, BFx, {BF[0][0]}, {BF[0][1]}, {BF[0][2]}, \
                            BFy, {BF[1][0]}, {BF[1][1]}, {BF[1][2]}, BFz, {BF[2][0]}, {BF[2][1]}, {BF[2][2]}\n")

def Visualize(infile_name, outfile_name):
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
            positions[0] = float(line[3].strip())
            positions[1] = float(line[5].strip())
            positions[2] = float(line[7].strip())

    # Animation
    # TODO fix animation
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    min, max = np.min(positions, axis=0), np.max(positions, axis=0)
    ax.set_xlim([min[0], max[0]])
    ax.set_ylim([min[1], max[1]])
    ax.set_zlim([min[2], max[2]])
    animated_plot, = ax.plot3D([], [], [], 'r')
    ax.set_title("Dead Reackoning")
    ax.set_xlabel("X", labelpad=20)
    ax.set_ylabel("Y", labelpad=20)
    ax.set_zlabel("Z", labelpad=20)
    ax.grid()
    def update(frame):
        animated_plot.set_data(positions[:, 0][:frame], positions[:, 1][:frame])
        animated_plot.set_3d_properties(positions[:, 2][:frame])
    animation = FuncAnimation(fig=fig, func=update, frames=num_lines, interval=25)
    animation.save(outfile_name)
    plt.show()

def main():
    indirectory_name = "../data/raw/raw_imu_data"
    outdirectory_name = "../result/raw/"
    for infile in os.listdir(indirectory_name):
        infile_name = os.path.join(indirectory_name, infile)
        outfile_name = os.path.splitext(os.path.join(outdirectory_name, infile))[0]+".csv"
        DeadReckoning_raw(infile_name, outfile_name)

if __name__ == "__main__":
    main()
