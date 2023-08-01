import numpy as np
import urdfpy
import sys
import pdb

# input has to end with .bk
path = sys.argv[1]
assert(path[-3:]=='.bk')

urdf = urdfpy.URDF.load(path)
for link in urdf.links:
    if len(link.collisions)==0:
        if len(link.visuals)>0:
            collision = urdfpy.Collision(name=link.visuals[0].name, 
            origin = link.visuals[0].origin, 
            geometry=link.visuals[0].geometry)
            link.collisions.append( collision )
        else:
            origin=np.eye(4)
            collision = urdfpy.Collision(name='collision', 
            origin = origin, 
            geometry=urdfpy.Geometry(sphere=urdfpy.Sphere(radius=0.05)) )
            link.collisions.append( collision )

urdf.save(path[:-3])
