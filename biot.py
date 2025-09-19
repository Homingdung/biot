# reproducing Rodrigo-GAHOZ-2024
from firedrake import *

baseN = 16
mesh = UnitCubeMesh(baseN, baseN, baseN)

scale = 64 
# map from [0, 1]^3 to [0, 64]^3
mesh.coordinates.dat.data[:] *= scale
x, y, z0 = SpatialCoordinate(mesh) 

# Function space for marking
M = FunctionSpace(mesh, "HDiv Trace", 0)
# “P” (degree 1) functions in 1D 
#or “HDiv Trace” (degree 0) functions in 2D or 3D to mark facet entities

# top, loading area (x in [16,48], y in [16,48])
m1 = Function(M).interpolate(
    conditional(And(eq(z0, 64), And(And(gt(x, 16), lt(x, 48)), And(gt(y, 16), lt(y, 48)))), 1, 0)
)

# top face, remaining domain
m2 = Function(M).interpolate(
    conditional(And(eq(z0, 64), Or(Or(le(x, 16), ge(x, 48)), Or(le(y, 16), ge(y, 48)))), 1, 0)
)

# left and right face x=0, x=64
m3 = Function(M).interpolate(
    conditional(Or(eq(x, 0), eq(x, 64)), 1, 0)
)

# front and back face y=0, y=64
m4 = Function(M).interpolate(
    conditional(Or(eq(y, 0), eq(y, 64)), 1, 0)
)

# bottom face z=0
m5 = Function(M).interpolate(
    conditional(eq(z0, 0), 1, 0)
)

# relable Mesh
# indicator function m1, m2, m3, m4, m5 
# label
mesh = RelabeledMesh(mesh, [m1, m2, m3, m4, m5], [101, 102, 103, 104, 105])
pvd = VTKFile("output/biot.pvd")

V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V1, V2])

z = Function(Z)
z_prev = Function(Z)
z_test = TestFunction(Z)

(u, p) = split(z)
(ut, pt) = split(z_test)
(up, pp) = split(z_prev)

eps = lambda x: sym(grad(x))

n = FacetNormal(mesh)
h = CellSize(mesh)
# time parameters
t = Constant(0)
dt = Constant(0.1)
T = 1.0
    
# physical parameters
E = Constant(3 * 10 ** 4)
nu = Constant(0.499)
mu = E/(1 + 2*nu)
alpha = Constant(1)
rhof = 1000
rhos = 500
phi0 = 0.1
rho = (1 - phi0) * rhos + phi0 * rhof

K = 1e-6 
muf = Constant(1)
eta = Constant(10)
lmbda = E*nu/((1-2*nu)*(1+nu))
M = 10**6
# source term
trac = as_vector([0, 0, -1e5 * t])
g = as_vector([0, 0, -9.8])
    
F = (
    2 * mu * inner(eps(u), eps(ut)) * dx
    + lmbda * inner(div(u), div(ut)) * dx
    - alpha * inner(div(ut), p) * dx
    #- rho * inner(g, ut) * dx
    - inner(trac, ut) * ds(101) # boundary for trac
    
    - alpha * inner(div (u - up)/dt, pt) * dx
    - K/muf * inner(grad(p), grad(pt)) * dx
    - eta * h **2 * inner(grad(p - pp)/dt, grad(pt)) * dx # stablization term
    #    - inner(f, pt) * dx
)

lu = {
	 "mat_type": "aij",
	 "snes_type": "newtonls",
	 "ksp_type":"preonly",
	 "pc_type": "lu",
	 "pc_factor_mat_solver_type":"mumps"
}
# Riesz preconditioning
d = 3
xi = sqrt(lmbda + 2*mu/d)
def riesz_u(u, v):
    return 2 * mu * inner(eps(u), eps(v)) * dx  + lmbda * inner(div(u),div(v)) * dx

def riesz_p(u, v):
    return dt * K/muf * inner(u, v) * dx + eta * h**2 * inner(grad(u), grad(v)) * dx + alpha ** 2 /xi**2 * inner(u, v) * dx

# FOV preconditioner
def fov_u(u, v):
    return 2 * mu * inner(eps(u), eps(v)) * dx  + lmbda * inner(div(u),div(v)) * dx

def fov_p(u, p, q):
    return dt * K/muf * inner(p, q) * dx + eta * h**2 * inner(grad(p), grad(q)) * dx + alpha ** 2 /xi**2 * inner(p, q) * dx - alpha * inner(div(u), q) * dx 


(u0, p0) = TrialFunctions(Z)
(u1, p1) = TestFunctions(Z)

Jp_riesz =  riesz_u(u0, u1) + riesz_p(p0, p1) 
Jp_fov = fov_u(u0, u1) + fov_p(u0, p0, p1)
    
riesz = {
        "mat_type": "matfree",
        "pmat_type": "nest",
        "snes_type": "newtonls",
        "snes_monitor":None,
        "snes_rtol": 1E-6,
        "snes_converged_reason":None,
        "ksp_monitor":None,
        "ksp_converged_reason":None,
        "ksp_type":"minres",
        "ksp_rtol": 1E-8,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type":"additive",
        "fieldsplit_ksp_type":"preonly",
        "fieldsplit_pc_type": "lu",
        "fieldsplit_pc_factor_mat_solver_type":"mumps"
}
fov = {
        "mat_type": "matfree",
        "pmat_type": "nest",
        "snes_type": "newtonls",
        "snes_monitor":None,
        "snes_rtol": 1E-6,
        "snes_converged_reason":None,
        "ksp_monitor":None,
        "ksp_converged_reason":None,
        "ksp_type":"gmres",
        "ksp_rtol": 1E-8,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type":"additive",
        "fieldsplit_ksp_type":"preonly",
        "fieldsplit_pc_type": "lu",
        "fieldsplit_pc_factor_mat_solver_type":"mumps"
}

sp = fov
Jp = Jp_riesz if sp == riesz else Jp_fov
bc_u = DirichletBC(Z.sub(0), 0, 105)
bc_p = DirichletBC(Z.sub(1), 0, [102, 103, 104, 105])
bcs = [bc_u, bc_p]

pb = NonlinearVariationalProblem(F, z, bcs, Jp = Jp)
solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    solver.solve()
    pvd.write(*z.subfunctions, time=float(t))
    z_prev.assign(z)


