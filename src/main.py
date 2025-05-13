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

def DeadReckoning(infile_name, outfile_name):
    with open(infile_name, 'r') as infile, open(outfile_name, 'w') as outfile:
        # easy to read csv files
        infile_csv = csv.reader(infile)
        next(infile_csv)  # skip the headings

        # Initialize states
        T = 0
        S = np.zeros(3)
        V = np.zeros(3)
        A = np.zeros(3)
        BF = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) # defines initial rotation

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

def KalmanFilter(infile_name, outfile_name, sigma_m = 3, sigma_v = 1, sigma_a = 5):
    # Read the no. of lines in inflie
    with open(infile_name, "rb") as infile:
        num_lines = sum(1 for _ in infile)
        infile.close()

    # Read the infile
    with open(infile_name, 'r') as infile, open(outfile_name, 'w') as outfile:
        # easy to read csv files
        infile_csv = csv.reader(infile)

        #initialize KF variables
        dt = 0.02  # Time step
        Z = np.zeros((3, 1))  # I/P
        # X = np.zeros((9, 1))  # O/P
        A = np.array([[1, 0, 0, dt, 0,  0,  (dt*dt)/2, 0,         0        ],
                      [0, 1, 0, 0,  dt, 0,  0,         (dt*dt)/2, 0        ],
                      [0, 0, 1, 0,  0,  dt, 0,         0,         (dt*dt)/2],
                      [0, 0, 0, 1,  0,  0,  dt,        0,         0        ],
                      [0, 0, 0, 0,  1,  0,  0,         dt,        0        ],
                      [0, 0, 0, 0,  0,  1,  0,         0,         dt       ],
                      [0, 0, 0, 0,  0,  0,  1,         0,         0        ],
                      [0, 0, 0, 0,  0,  0,  0,         1,         0        ],
                      [0, 0, 0, 0,  0,  0,  0,         0,         1        ],])
        H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],])
        Q = np.zeros((9, 9))  # Zero process noise. Here take as zero.
        R = sigma_m*np.identity(3)  # Measurement noise.  Must be given by the sensor datasheet
        X = np.zeros((9, 1))  # Initial state
        P = np.diag([0, 0, 0, sigma_v, sigma_v, sigma_v, sigma_a, sigma_a, sigma_a])  # Initial error

        for _ in range(num_lines):
            # Take measurement
            line = next(infile_csv)
            Z[0] = float(line[3].strip())
            Z[1] = float(line[5].strip())
            Z[2] = float(line[7].strip())

            # Prediction step
            X_ = A@X
            P_ = A@P@A.T+Q

            # Kalman gain
            K = P_@H.T@np.linalg.inv(H@P_@H.T+R)

            # Estimation setp
            X = X_+K@(Z-H@X_)
            P = P_-K@H@P_

            # Log position
            outfile.write(f"Time, {float(line[1].strip())}, Pos_X, {X[0][0]}, Pos_Y, {X[1][0]}, Pos_Z, {X[2][0]} \n")

def VisualizeGif(infile_name, outfile_name, title):
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
    ax.view_init(10, -60)
    animated_plot, = ax.plot3D([], [], [], 'r')
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    ax.set_title(title)
    ax.set_xlabel("X", labelpad=20)
    ax.set_ylabel("Y", labelpad=20)
    ax.set_zlabel("Z", labelpad=20)
    ax.grid()
    def update(frame):
        animated_plot.set_data(positions[:, 0][:frame], positions[:, 1][:frame])
        animated_plot.set_3d_properties(positions[:, 2][:frame])
    animation = FuncAnimation(fig=fig, func=update, frames=range(0, num_lines+1, 10), interval=20)
    animation.save(outfile_name)

def VisualizeCompare(n):
    # Close any previously drawn figures
    plt.cla()
    plt.clf()
    plt.close("all")

    # Directory names
    DeadReckoning_indirectory_name = "../result/DeadReckoning/csv"
    KalmanFilter_indirectory_name = "../result/KalmanFilter/csv"

    # Selecting proper files
    DeadReckoning_files = os.listdir(DeadReckoning_indirectory_name)
    KalmanFilter_files = os.listdir(KalmanFilter_indirectory_name)
    DeadReckoning_infile_name = os.path.join(DeadReckoning_indirectory_name, DeadReckoning_files[n])
    KalmanFilter_infile_name = os.path.join(KalmanFilter_indirectory_name, KalmanFilter_files[n])

    # Dead Reckoning animation
    # Read the no. of lines in inflie
    with open(DeadReckoning_infile_name, "rb") as infile:
        num_lines = sum(1 for _ in infile)
        infile.close()

    # Read the infile
    positions = np.zeros((num_lines, 3))
    with open(DeadReckoning_infile_name, 'r') as infile:
        infile_csv = csv.reader(infile)
        for i in range(num_lines):
            line = next(infile_csv)
            positions[i][0] = float(line[3].strip())
            positions[i][1] = float(line[5].strip())
            positions[i][2] = float(line[7].strip())

    # Animation
    fig1 = plt.figure(1, figsize=(10, 10))
    ax1 = plt.axes(projection="3d")
    mag = max(abs(np.min(positions)), abs(np.max(positions)))
    ax1.set_xlim([-mag, mag])
    ax1.set_ylim([-mag, mag])
    ax1.set_zlim([-mag, mag])
    x, y, z = np.array([[5, 0, 0],[0, 5, 0],[0, 0, 5]])
    u, v, w = np.array([[mag, 0, 0],[0, mag, 0],[0, 0, mag]])
    ax1.view_init(10, -60)
    animated_plot1, = ax1.plot3D([], [], [], 'r')
    ax1.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    ax1.set_title(f"Dead Reckoning {n}")
    ax1.set_xlabel("X", labelpad=20)
    ax1.set_ylabel("Y", labelpad=20)
    ax1.set_zlabel("Z", labelpad=20)
    ax1.grid()
    def update1(frame):
        animated_plot1.set_data(positions[:, 0][:frame], positions[:, 1][:frame])
        animated_plot1.set_3d_properties(positions[:, 2][:frame])
    animation1 = FuncAnimation(fig=fig1, func=update1, frames=range(0, num_lines+1, 10), interval=20)


    # Kalman Filter animation
    # Read the no. of lines in inflie
    with open(KalmanFilter_infile_name, "rb") as infile:
        num_lines = sum(1 for _ in infile)
        infile.close()

    # Read the infile
    positions = np.zeros((num_lines, 3))
    with open(KalmanFilter_infile_name, 'r') as infile:
        infile_csv = csv.reader(infile)
        for i in range(num_lines):
            line = next(infile_csv)
            positions[i][0] = float(line[3].strip())
            positions[i][1] = float(line[5].strip())
            positions[i][2] = float(line[7].strip())

    # Animation
    fig2 = plt.figure(2, figsize=(10, 10))
    ax2 = plt.axes(projection="3d")
    mag = max(abs(np.min(positions)), abs(np.max(positions)))
    ax2.set_xlim([-mag, mag])
    ax2.set_ylim([-mag, mag])
    ax2.set_zlim([-mag, mag])
    x, y, z = np.array([[5, 0, 0],[0, 5, 0],[0, 0, 5]])
    u, v, w = np.array([[mag, 0, 0],[0, mag, 0],[0, 0, mag]])
    ax2.view_init(10, -60)
    animated_plot2, = ax2.plot3D([], [], [], 'r')
    ax2.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
    ax2.set_title(f"Kalman Filter {n}")
    ax2.set_xlabel("X", labelpad=20)
    ax2.set_ylabel("Y", labelpad=20)
    ax2.set_zlabel("Z", labelpad=20)
    ax2.grid()
    def update2(frame):
        animated_plot2.set_data(positions[:, 0][:frame], positions[:, 1][:frame])
        animated_plot2.set_3d_properties(positions[:, 2][:frame])
    animation2 = FuncAnimation(fig=fig2, func=update2, frames=range(0, num_lines+1, 10), interval=20)

    plt.show()

def apply_DeadReckoning():
    # directory names
    indirectory_name = "../data/raw_imu_data"
    outdirectory_name = "../result/DeadReckoning/csv"

    # apply dead reckoning
    for infile in os.listdir(indirectory_name):
        infile_name = os.path.join(indirectory_name, infile)
        outfile_name = os.path.splitext(os.path.join(outdirectory_name, infile))[0]+".csv"
        DeadReckoning(infile_name, outfile_name)

def apply_KalmanFilter():
    # directory names
    indirectory_name = "../result/DeadReckoning/csv"
    outdirectory_name = "../result/KalmanFilter/csv"

    # apply Kalman Filter
    for infile in os.listdir(indirectory_name):
        infile_name = os.path.join(indirectory_name, infile)
        outfile_name = os.path.splitext(os.path.join(outdirectory_name, infile))[0]+".csv"
        KalmanFilter(infile_name, outfile_name)

def apply_VisualizeGif(n):
    # directory names
    DeadReckoning_indirectory_name = "../result/DeadReckoning/csv"
    DeadReckoning_outdirectory_name = "../result/DeadReckoning/gif"
    KalmanFilter_indirectory_name = "../result/KalmanFilter/csv"
    KalmanFilter_outdirectory_name = "../result/KalmanFilter/gif"

    # visualize DeadReckoning
    files_to_visualize = os.listdir(DeadReckoning_indirectory_name)
    for i in range(n):
        infile_name = os.path.join(DeadReckoning_indirectory_name, files_to_visualize[i])
        outfile_name = os.path.splitext(os.path.join(DeadReckoning_outdirectory_name, files_to_visualize[i]))[0]+".gif"
        VisualizeGif(infile_name, outfile_name, "Dead Reckoning")

    # visualize KalmanFilter
    files_to_visualize = os.listdir(KalmanFilter_indirectory_name)
    for i in range(n):
        infile_name = os.path.join(KalmanFilter_indirectory_name, files_to_visualize[i])
        outfile_name = os.path.splitext(os.path.join(KalmanFilter_outdirectory_name, files_to_visualize[i]))[0]+".gif"
        VisualizeGif(infile_name, outfile_name, "Kalman Filter")

def main():
    apply_DeadReckoning()
    apply_KalmanFilter()
    apply_VisualizeGif(5)  # Number of gifs to ceate
    VisualizeCompare(0)  # Index of the raw data file to compare, after DR and KF

if __name__ == "__main__":
    main()

# TODO implement MultiVisualize
# TODO different initial positions and KF tuning
