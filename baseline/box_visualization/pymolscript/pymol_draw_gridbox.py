from pymol.cgo import *
import numpy as np
from pathlib import Path


def gridbox(box_coords, index, r1=0, g1=0, b1=1, trasp=0.2):
    """
    DESCRIPTION
    Create a box from the center coordinate of the box and the size of box

    USAGE
    run gridbox.py
    1the simplest
    gridbox center_x,center_y,center_z,size_x,size_y,size_z
    2rename the box object
    gridbox center_x,center_y,center_z,size_x,size_y,size_z,name,
    3set the color of the box object
    gridbox center_x,center_y,center_z,size_x,size_y,size_z,name,r1,g1,b1
    4set the trasp of the box
    gridbox center_x,center_y,center_z,size_x,size_y,size_z,name,r1,g1,b1,trasp

    ps:the value of r1,g1,b1 trasp   range  is 0-1
       trasp=1,no trasprent


    """

    r1 = float(r1)
    g1 = float(g1)
    b1 = float(b1)
    trasp = float(trasp)

    p1, p2, p3, p4, p5, p6, p7, p8 = box_coords[1], box_coords[3], box_coords[7], box_coords[5], \
                                     box_coords[0], box_coords[2], box_coords[6], box_coords[4],
    obj = [
        ALPHA, trasp,
        COLOR, r1, g1, b1,
        BEGIN, TRIANGLE_STRIP,
        VERTEX, p1[0], p1[1], p1[2],
        VERTEX, p2[0], p2[1], p2[2],
        VERTEX, p4[0], p4[1], p4[2],
        VERTEX, p3[0], p3[1], p3[2],
        END,

        BEGIN, TRIANGLE_STRIP,
        # COLOR,1,0,0,
        VERTEX, p1[0], p1[1], p1[2],
        VERTEX, p5[0], p5[1], p5[2],
        VERTEX, p4[0], p4[1], p4[2],
        VERTEX, p8[0], p8[1], p8[2],
        END,

        BEGIN, TRIANGLE_STRIP,
        VERTEX, p4[0], p4[1], p4[2],
        VERTEX, p8[0], p8[1], p8[2],
        VERTEX, p3[0], p3[1], p3[2],
        VERTEX, p7[0], p7[1], p7[2],
        END,

        BEGIN, TRIANGLE_STRIP,
        VERTEX, p7[0], p7[1], p7[2],
        VERTEX, p3[0], p3[1], p3[2],
        VERTEX, p6[0], p6[1], p6[2],
        VERTEX, p2[0], p2[1], p2[2],
        END,

        BEGIN, TRIANGLE_STRIP,
        # COLOR,0,1,0,
        VERTEX, p2[0], p2[1], p2[2],
        VERTEX, p6[0], p6[1], p6[2],
        VERTEX, p1[0], p1[1], p1[2],
        VERTEX, p5[0], p5[1], p5[2],
        END,

        BEGIN, TRIANGLE_STRIP,
        VERTEX, p1[0], p1[1], p1[2],
        VERTEX, p5[0], p5[1], p5[2],
        VERTEX, p4[0], p4[1], p4[2],
        VERTEX, p8[0], p8[1], p8[2],
        END,

        BEGIN, TRIANGLE_STRIP,
        VERTEX, p5[0], p5[1], p5[2],
        VERTEX, p8[0], p8[1], p8[2],
        VERTEX, p6[0], p6[1], p6[2],
        VERTEX, p7[0], p7[1], p7[2],
        END,
    ]
    name = "gridbox" + str(index)
    cmd.load_cgo(obj, name)


def view_boxes():
    """Note that please use pymol to go to the current directory (box_visualization) for visualization."""
    path = Path.cwd()
    path_of_np_coords = path.parent.parent.joinpath("box_coords").joinpath("3c70.npy")
    box_coords = np.load(str(path_of_np_coords))
    for i, coord in enumerate(box_coords):
        gridbox(coord, i)


cmd.extend('view_boxes', view_boxes)
