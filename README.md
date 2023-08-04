# vidyn

## Install
`pip install urdfpy==0.0.22; pip install networkx==2.5`
`pip install git+https://github.com/MightyChaos/dqtorch`
`pip install vedo==2022.2.3`
`pip install warp==0.7.2`
`pip install open3d==0.14.1`

(0) use larger skeleton to avoid blowing up
(1) new urdf parser to deal with sperical joints => no reduncant links => 2x time steps
(2) smaller feet mass / normal gravity / use ke=20 to avoid bouncing artefact / limit_ke, shape_mu
