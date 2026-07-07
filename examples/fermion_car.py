from clifford_qc import MV, I, c_op, cdag_op, number_op, number_op_pauli

n = 3
ok = True
for i in range(n):
    for j in range(n):
        ci, cj = c_op(n, i), c_op(n, j)
        cdi, cdj = cdag_op(n, i), cdag_op(n, j)
        ok &= (ci * cdj + cdj * ci).is_close(I(n) if i == j else MV(n))
        ok &= (ci * cj + cj * ci).is_zero()
print("CAR:", ok)
print("n_1 labels:", number_op(n, 1).to_labels())
print("matches 1/2(1-Z_1):", number_op(n, 1).is_close(number_op_pauli(n, 1)))
