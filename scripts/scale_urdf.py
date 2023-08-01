import numpy as np
import urdfpy
import sys
import pdb

# input has to end with .bk
path = sys.argv[1]
assert(path[-3:]=='.bk')


#scale=2.5
scale=2
# 2 for human
# 2.5 for wolf
urdf = urdfpy.URDF.load(path)
for link in urdf.links:
    # collision
    origin = link.collisions[0].origin
    geometry = link.collisions[0].geometry

    origin[:3,3] *= scale
    if geometry.box is not None:
        geometry.box.size *= scale

    link.collisions[0] = urdfpy.Collision(name=link.collisions[0].name, 
    origin = origin, 
    geometry= geometry)
   
    # visual
    if len(link.visuals)==0: 
        #print(link.name)
        continue
    origin = link.visuals[0].origin
    geometry = link.visuals[0].geometry

    origin[:3,3] *= scale
    if geometry.box is not None:
        geometry.box.size *= scale

    link.visuals[0] = urdfpy.Visual(name=link.visuals[0].name, 
    origin = origin, 
    geometry= geometry)

for joint in urdf.joints:
    origin = joint.origin
    origin[:3,3] *= scale


urdf.save(path[:-3])
