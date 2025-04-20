from datetime import datetime
import csv
import math
import numpy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
Re = 6_371_000

def get_theta(lat):
    return (math.pi/2 - lat)

def get_phi(lon):
    if lon >= 0:
        return lon
    else:
        return 2*math.pi + lon

def polar_to_cartesian(r, theta, phi):
    return [r*math.sin(theta)*math.cos(phi), r*math.sin(theta)*math.sin(phi), r*math.cos(theta)]

def cartesian_to_polar(x, y, z):
    global Re
    r = math.pow(x*x + y*y + z*z, 0.5)

    if z >= 0:
        theta = math.atan((math.pow(x*x + y*y, 0.5))/z)
    else:
        theta = math.pi + math.atan((math.pow(x*x + y*y, 0.5))/z)

    if y >= 0:
        if x >= 0:
            phi = math.atan(y/x)
        else:
            phi = 2*math.pi + math.atan(y/x)
    else:
        phi = math.pi + math.atan(y/x)

    return [r, theta, phi]

def polar_to_geo(r, theta, phi):
    if phi <= math.pi:
        long = phi
    else:
        long = -2*math.pi + phi

    lat = (math.pi/2) - theta

    alt = r - Re

    return [lat, long, alt]


def get_mobile_frame(theta, phi):
    bx = numpy.array([[-1*math.cos(theta)*math.cos(phi)], [-1*math.cos(theta)*math.sin(phi)], [math.sin(theta)]])
    by = numpy.array([[-1*math.sin(phi)], [math.cos(phi)], [0]])
    bz = numpy.array([[-1*math.sin(theta)*math.cos(phi)], [-1*math.sin(theta)*math.sin(phi)], [-1*math.cos(theta)]])

    return [bx, by, bz]

def Rx(theta_x):
    return numpy.array([[1, 0, 0], [0, math.cos(theta_x), -1*math.sin(theta_x)], [0, math.sin(theta_x), math.cos(theta_x)]])

def Ry(theta_y):
    return numpy.array([[math.cos(theta_y), 0, math.sin(theta_y)], [0, 1, 0], [-1*math.sin(theta_y), 0, math.cos(theta_y)]])

def Rz(thata_z):
    return numpy.array([[math.cos(thata_z), -1*math.sin(thata_z), 0], [math.sin(thata_z), math.cos(thata_z), 0], [0, 0, 1]])

def rotate_vector(axis, theta, vector):
    A = Rx((math.atan(axis[1]/axis[2])))
    B = Ry((math.atan(axis[0]/math.pow((axis[1]**2 + axis[2]**2), 0.5))))
    C = Rz(theta)
    D = numpy.linalg.inv(B)
    E = numpy.linalg.inv(A)

    result = A
    for matrix in [B, C, D, E, vector]:
        result = numpy.matmul(result, matrix)
    return result

def rotate_frame(bx, by, bz, yaw, pitch, roll):
    # apply yaw
    b2x = rotate_vector(bz, yaw, bx)
    b2y = rotate_vector(bz, yaw, by)
    b2z = bz
    bx, by, bz = b2x, b2y, b2z

    #apply pitch
    b2x = rotate_vector(by, pitch, bx)
    b2y = by
    b2z = rotate_vector(by, pitch, bz)
    bx, by, bz = b2x, b2y, b2z

    #apply roll
    b2x = bx
    b2y = rotate_vector(bx, roll, by)
    b2z = rotate_vector(bx, roll, bz)
    bx, by, bz = b2x, b2y, b2z

    return [bx, by, bz]

def main():
    global Re
    k = 10_000
    # file_name = input("Enter file name: ")
    x_values = []
    y_values = []
    z_values = []

    with open(f"source/Data.csv", mode="r") as in_file, open("result/displacement.csv", mode="w") as out_file:
        csv_file = csv.reader(in_file)
        initial_line = csv_file.__next__()

        initial_time = datetime.strptime(initial_line[2].strip(),"%d/%m/%Y %H:%M:%S")
        lat = float(initial_line[17])
        lon = float(initial_line[19])
        theta = get_theta(lat)
        phi = get_phi(lon)
        bx, by, bz = get_mobile_frame(theta, phi)
        s = polar_to_cartesian(Re, theta, phi)
        u = numpy.zeros((3, 1))
        v = numpy.zeros((3, 1))
        a = numpy.zeros((3, 1))

        row = f"Time, {initial_line[2].strip()}, X, {float(s[0])}, Y, {float(s[1])}, Z, {float(s[2])}, Lat, {lat}, Lon, {lon}, Alt, {0}, Bx, {bx[0]}, {bx[1]}, {bx[2]}, By, {by[0]}, {by[1]}, {by[2]}, Bz, {bz[0]}, {bz[1]}, {bz[2]}\n"
        out_file.write(row)
        x_values.append(float(s[0]))
        y_values.append(float(s[1]))
        z_values.append(float(s[2]))

        for line in csv_file:
            delta_time = (datetime.strptime(line[2].strip(),"%d/%m/%Y %H:%M:%S") - initial_time).total_seconds()
            bx, by, bz = rotate_frame(bx, by, bz, float(line[11]), float(line[13]), float(line[15]))
            a = float(line[4])*bx + float(line[6])*by + float(line[8])*bz
            u = v
            for i in range(3):
                v[i] = u[i] + a[i]*delta_time
                s[i] += u[i]*delta_time + 0.5*a[i]*delta_time*delta_time
            r, theta, phi = cartesian_to_polar(s[0], s[1], s[2])
            lat, lon, alt = polar_to_geo(r, theta, phi)

            row = f"Time, {line[2].strip()}, X, {float(s[0])}, Y, {float(s[1])}, Z, {float(s[2])}, Lat, {lat}, Lon, {lon}, Alt, {alt}, Bx, {bx[0]}, {bx[1]}, {bx[2]}, By, {by[0]}, {by[1]}, {by[2]}, Bz, {bz[0]}, {bz[1]}, {bz[2]}\n"
            out_file.write(row)
            x_values.append(float(s[0]))
            y_values.append(float(s[1]))
            z_values.append(float(s[2]))

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.set_xlim([min(x_values), max(x_values)])
    ax.set_ylim([min(y_values), max(y_values)])
    ax.set_zlim([min(z_values), max(z_values)])
    animated_plot, = ax.plot3D([], [], [], 'r')
    ax.set_title("Dead Reackoning")
    ax.set_xlabel("X", labelpad=20)
    ax.set_ylabel("Y", labelpad=20)
    ax.set_zlabel("Z", labelpad=20)
    ax.grid()

    def update(frame):
        animated_plot.set_data(x_values[:frame], y_values[:frame])
        animated_plot.set_3d_properties(z_values[:frame])

    animation = FuncAnimation(fig=fig, func=update, frames=len(x_values), interval=25)
    animation.save("result/displacement.gif")
    plt.show()

if __name__ == "__main__":
    main()
