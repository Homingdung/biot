# darcy flow
# reproduce Axelsson-BBKA-2015

from firedrake import *
from tabulate import tabulate

output = False
def patch_multigrid():
    return {
        "ksp_type":"preonly",
        "pc_type": "mg",
        "mg_coarse": {
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_pc_type": "lu",
            "assembled_pc_factor_mat_solver_type": "mumps",
        },
        "mg_levels": {
            "ksp_type": "chebyshev",
            "ksp_max_it": 4,
            "ksp_chebyshev_esteig": "0.625,0.125,0.125,1.125",
            "esteig_ksp_type": "minres",
            "esteig_ksp_norm_type": "preconditioned",
            "esteig_ksp_view_singularvalues": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.PatchPC",
            "patch_pc_patch_save_operators": True,
            "patch_pc_patch_precompute_element_tensors": True,
            "patch_pc_patch_construct_type": "star",
            "patch_pc_patch_construct_dim": 0,
            "patch_pc_patch_sub_mat_type": "seqdense",
            "patch_sub_ksp_type": "preonly",
            "patch_sub_pc_type": "lu",
        },
    }


def run_simulation(baseN, gamma):
    baseN = 100
    mesh = UnitSquareMesh(baseN, baseN)

    Vd = FunctionSpace(mesh, "RT", 1)
    Vn = FunctionSpace(mesh, "DG", 0)

    Z = MixedFunctionSpace([Vd, Vn])
    z = Function(Z)
    z_test = TestFunction(Z)

    (u, p) = split(z)
    (ut, pt) = split(z_test)

    sigma = 4.0#0, 2, 4
    rng = np.random.default_rng(2025) 

    k = Function(Vn, name="k")
    xi = rng.standard_normal(k.dat.data.shape)    # N(0,1) for each element
    k_vals = np.exp(sigma * xi)                   #k = exp(sigma * xi), log k ~ N(0, sigma^2)
    k.dat.data[:] = k_vals                        #DG0 coefficient

    kmin = float(k.dat.data_ro.min())
    kmax = float(k.dat.data_ro.max())
    print("contrast kmax/kmin ≈", kmax/kmin)

# boundary
    q1_left  = as_vector([0, 0])
    q1_right = as_vector([0, 0])
    q2_bottom = Constant(1.0)
    q2_top    = Constant(0.0)
    f = Constant(-9.8)

    F = (
          1/k * inner(u, ut) * dx
        - inner(p, div(ut)) * dx
        - inner(div(u), pt) * dx
        - inner(f, pt) * dx
    )

    lu = {
         "mat_type": "aij",
         "snes_type": "ksponly",
         "snes_monitor": None,
         "ksp_type":"preonly",
         "ksp_monitor": None,
         "pc_type": "lu",
         "pc_factor_mat_solver_type":"mumps"
    }
    riesz = {
            "mat_type": "nest",
            "snes_type": "ksponly",
            "snes_monitor": None,
            "ksp_monitor": None,
            "ksp_max_it": 1000,
            "ksp_norm_type": "preconditioned",
            "ksp_type": "minres",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "additive",
            "fieldsplit_pc_type": "lu",
            "fieldsplit_pc_factor_mat_solver_type": "mumps",
            "ksp_atol": 1.0e-5,
            "ksp_rtol": 1.0e-5,
            "ksp_minres_nutol": 1E-8,
            "ksp_convergence_test": "skip",
    }
    #riesz["fieldsplit"] = patch_multigrid()
    sp = riesz

    # preconditioning
    gamma = Constant(gamma)
    Up = 0.5 * (inner(1/k * u, u) + inner(div(u) * gamma, div(u)) + inner(p * (1/gamma), p))*dx
    Jp = derivative(derivative(Up, z), z)
    Jp = Jp if sp == riesz else None

    bcs = [DirichletBC(Z.sub(1), q2_bottom, (3, )),
           DirichletBC(Z.sub(1), q2_top, (4, ))]

    pb = NonlinearVariationalProblem(F, z, bcs, Jp = Jp)
    solver = NonlinearVariationalSolver(pb, solver_parameters = sp)
    solver.solve()

    if output:
        pvd = VTKFile("output/darcy.pvd")
        pvd.write(*z.subfunctions)
    its = solver.snes.getLinearSolveIterations()

    return its



    

baseNs = [4, 8, 16, 32, 64, 128]
gammas = [1e3, 1e5, 1e6]

# run sweep
table = []
for N in baseNs:
    row = []
    for g in gammas:
        try:
            its = run_simulation(N, g)
        except Exception as e:
            # record a sentinel on failure
            print(f"WARNING: (N={N}, gamma={g}) failed with: {e}")
            its = None
        row.append(its)
    table.append(row)

headers = ["baseN \\ gamma"] + [f"{g:.0e}" for g in gammas]
rows = [[baseNs[i]] + table[i] for i in range(len(baseNs))]

print("\n" + tabulate(rows, headers=headers, tablefmt="github", stralign="center", numalign="center"))
