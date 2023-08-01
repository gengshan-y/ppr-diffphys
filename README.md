# vidyn

## Install
`pip install git+https://github.com/MightyChaos/dqtorch`

(0) use larger skeleton to avoid blowing up
(1) new urdf parser to deal with sperical joints => no reduncant links => 2x time steps
(2) smaller feet mass / normal gravity / use ke=20 to avoid bouncing artefact / limit_ke, shape_mu