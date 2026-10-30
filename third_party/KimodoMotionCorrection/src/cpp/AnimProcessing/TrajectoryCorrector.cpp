/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TrajectoryCorrector.h"
#include <iostream>

static void removeRows(
    Eigen::SparseMatrix<double>& M,
    Eigen::MatrixXd *v,
    int minCoeffs)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> rowMajorMat = M;
    rowMajorMat.makeCompressed(); // Ensure compressed format

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(rowMajorMat.nonZeros());

    int newRow = 0;
    for (int i = 0; i < rowMajorMat.outerSize(); ++i) {
        // Get nonzero count via outerIndexPtr (compressed format only)
        int nnz = rowMajorMat.outerIndexPtr()[i + 1] - rowMajorMat.outerIndexPtr()[i];

        if (nnz >= minCoeffs) {
            // Iterate through nonzeros in this row
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(rowMajorMat, i); it; ++it) {
                triplets.emplace_back(newRow, it.col(), it.value());
            }
            if (v)
            {
                v->row(newRow) = v->row(i);
            }
            newRow++;
        }
    }

    M = Eigen::SparseMatrix<double>(newRow, M.cols());
    M.setFromTriplets(triplets.begin(), triplets.end());
    if (v)
    {
        v->conservativeResize(newRow, v->cols());
    }
}

static void multVelWeights(
    Eigen::SparseMatrix<double>& V,
    Eigen::MatrixXd* v_rhs,
    const Eigen::VectorXd& velocityWeights
)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> rowMajorMat = V;
    rowMajorMat.makeCompressed(); // Ensure compressed format

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(rowMajorMat.nonZeros());

    for (int i = 0; i < rowMajorMat.outerSize(); ++i) {
        // Iterate through nonzeros in this row
        Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(rowMajorMat, i);
        double vel_weight = velocityWeights[it.col()];
        for(; it; ++it)
        {
            triplets.emplace_back(i, it.col(), it.value() * vel_weight);
        }
        if (v_rhs)
        {
            (*v_rhs).row(i) = (*v_rhs).row(i) * vel_weight;
        }
    }
}


void TrajectoryCorrector::computeDiffMats(
    Eigen::SparseMatrix<double>& V,
    Eigen::SparseMatrix<double>& A,
    uint32_t N,
    const Eigen::VectorXd& velocityWeights,
    Eigen::MatrixXd* v_rhs,
    Eigen::MatrixXd* a_rhs)
{
    std::vector<Eigen::Triplet<double>> tripletList;

    // Identity matrix"
    tripletList.clear();
    Eigen::SparseMatrix<double> I(N, N);
    for (uint32_t i = 0; i < N; ++i)
    {
        tripletList.emplace_back(i, i, 1);
    }
    I.setFromTriplets(tripletList.begin(), tripletList.end());

    // urr, a time translation operator? Gives you the value on the next frame.
    // Leave the last row blank because that's the end of the timeline.
    tripletList.clear();
    Eigen::SparseMatrix<double> T(N, N);
    Eigen::MatrixXd t_rhs;
    for(uint32_t i = 0; i < N-1; ++i)
    {
        // next frame is
        tripletList.emplace_back(i, i+1, 1.0);
    }
    T.setFromTriplets(tripletList.begin(), tripletList.end());

    // v = Tx + t_rhs - x;
    // v = (T - I)x + t_rhs;
    V = T - I;
    if (v_rhs)
    {
        *v_rhs = t_rhs;
    }
    removeRows(V, v_rhs, 2);

    // a = -x + 2 (T x + t_rhs) - (T (T x + t_rhs) + t_rhs)
    // a = (-I + 2 T - T^2) x + t_rhs - T t_rhs
    A = 2 * T - I - T * T;
    if (a_rhs)
    {
        *a_rhs = t_rhs - T * t_rhs;
    }
    removeRows(A, a_rhs, 3);

    if (velocityWeights.size() > 0)
    {
        multVelWeights(V, v_rhs, velocityWeights);
    }
}

TrajectoryCorrector::TrajectoryCorrector(
    const Eigen::VectorXd& margins,
    float pos_weight,
    float vel_weight,
    float acc_weight,
    const Eigen::VectorXd& velocityWeights,
    uint32_t admm_iters ) :
    m_admm_iters(admm_iters)
{

    // This class is used to modify a trajectory to hit specific values at
    // specific frames, while respecting the following soft constraints:

    // * Preserve the original positions
    // * Preserve the original velocities
    // * Preserve the original accelerations

    // The weights of these soft constraints are specified in "pos_weight" etc.

    // This is posed as a minimization problem:

    // E(x) = pos_weight * |x - x_orig|^2 + vel_weight * |V x - V x_orig| + acc_weight * |A x - A x_orig|

    // where you minimize E(x) subject to specified values at indices where "mask"
    // is equal to 1. V is a matrix that computes the N-1 velocities between frame n-1 and frame n,
    // and A computes the N-2 accelerations associated with frames n-1, n and n+1.

    // In addition to this, there are constraints where the trajectory is allowed to
    // deviate from the target points by a maximum margin. The "margins" input to this
    // constructor specifies what type of constraint is active on a particular frame:

    // margins[0] < 0   ==> unconstrained
    // margins[i] == 0  ==> pinned on this frame
    // margins[i] > 0   ==> can deviate within the margin

    // I'm solving the optimization problem using ADMM, ie following equations
    // 8,9,10 on this paper:

    // https://mattoverby.net/files/admm-pd-overby17.pdf

    uint32_t N = uint32_t(margins.rows());
    for(uint32_t i = 0; i < N; ++i)
    {
        if( margins[i] > 0 )
        {
            m_margin_locs.push_back(i);
            m_margin_vals.push_back(margins[i]);
        }

        if(margins[i] == 0)
        {
            m_constrained_locs.push_back(i);
        }
        else
        {
            m_unconstrained_locs.push_back(i);
        }
    }

    Eigen::SparseMatrix<double> V, A;
    computeDiffMats(
        V,  A,
        N, velocityWeights
    );

    // build an identity matrix:
    std::vector<Eigen::Triplet<double>> tripletList;
    Eigen::SparseMatrix<double> I(N, N);
    for (uint32_t i = 0; i < N; ++i)
    {
        tripletList.emplace_back(i, i, 1.0f);
    }
    I.setFromTriplets(tripletList.begin(), tripletList.end());

    /*
    self.N = (
            self.pos_weight * torch.diag_embed(torch.full_like(interp_mask, 1)) +
            self.vel_weight * torch.matmul(self.V.T, self.V) +
            self.acc_weight * torch.matmul(self.A.T, self.A)
        )
    */

    m_N = pos_weight * I + vel_weight * (V.transpose() * V) + acc_weight * (A.transpose() * A);

    double diagMax = 0;
    for (uint32_t i = 0; i < N; ++i)
    {
        diagMax = std::max(m_N.coeff(i,i), diagMax);
    }
    m_admm_stepsize = 0.5f * sqrtf(float(diagMax));

    /*
    M = (
        self.N +
        self.admm_stepsize * torch.matmul(self.S.T, self.S)
    )
    */
    tripletList.clear();
    Eigen::SparseMatrix<double> M(N, N);
    for( auto i : m_margin_locs)
    {
        tripletList.emplace_back(i, i, m_admm_stepsize);
    }
    M.setFromTriplets(tripletList.begin(), tripletList.end());
    M += m_N;

    /*
    self.lhsmat = torch.matmul(self.U.T, torch.matmul(self.M, self.U))
    self.lhsmat_inv = torch.inverse(self.lhsmat)
    */
    tripletList.clear();
    Eigen::SparseMatrix<double> S(m_unconstrained_locs.size(), N);
    for (uint32_t i = 0; i < m_unconstrained_locs.size(); ++i)
    {
        uint32_t ifull = m_unconstrained_locs[i];
        tripletList.emplace_back(i, ifull, 1.0f);
    }
    S.setFromTriplets(tripletList.begin(), tripletList.end());
    M = S * M * S.transpose();

    if(m_unconstrained_locs.size())
    {
        m_system_lu.compute(M);
    }
}


void TrajectoryCorrector::Interpolate(
    Eigen::MatrixXd& x,
    const Eigen::MatrixXd& observations,
    const Eigen::MatrixXd& ref_positions
) const
{
    if(
        m_constrained_locs.empty() &&
        m_margin_locs.empty()
    )
    {
        x = ref_positions;
        return;
    }

    uint32_t numCols = uint32_t(x.cols());
    if(m_margin_locs.empty())
    {
        x_update(
            x,
            Eigen::MatrixXd(0, numCols),
            Eigen::MatrixXd(0, numCols),
            ref_positions,
            observations
        );
    }
    else
    {
        x = ref_positions;
        Eigen::MatrixXd z(m_margin_locs.size(), numCols);
        Eigen::MatrixXd z_t(m_margin_locs.size(), numCols);
        Eigen::MatrixXd u(m_margin_locs.size(), numCols);
        for( uint32_t i = 0; i < m_margin_locs.size(); ++i)
        {
            for(uint32_t j = 0; j < numCols; ++j)
            {
                z_t(i, j) = observations(m_margin_locs[i], j);
                z(i, j) = ref_positions(m_margin_locs[i], j);
                u(i, j) =0;
            }
        }

        for(uint32_t i = 0; i < m_admm_iters; ++i)
        {
            x_update(
                x,
                z,
                u,
                ref_positions,
                observations
            );
            z_update(z, x, z_t, u);
            u_update(u, x, z);
        }
    }

}

void TrajectoryCorrector::x_update(
    Eigen::MatrixXd &x,
    const Eigen::MatrixXd &z,
    const Eigen::MatrixXd &u,
    const Eigen::MatrixXd &x_t, // reference positions - defines the original shape of the curve that we want to preserve
    const Eigen::MatrixXd &x_o  // target positions for constraints
) const
{

    uint32_t numRows = uint32_t(x.rows());
    uint32_t numCols = uint32_t(x.cols());

    // Here's what we're minimizing with ADMM:
    // min f(x) + g(z)
    // s.t A x + B z = c

    // Make these choices so that z = S x:
    // A = S, B = -I, c = 0
    //
    // g(z) = infinity when it's too far away from z_target, zero otherwise
    //
    // f(x) penalizes deviations in position, velocity and acceleration
    // from a reference trajectory:
    //
    // f(x) = 1/2(
    //    kx |I x - x_t|^2 +
    //    kv |V x - v_t|^2 +
    //    kx |A x - a_t|^2
    // )
    //
    // It's also infinite when components of x devaiate from their target
    // values where they're pinned...

    // Substituting the matrices into the standard admm update rules gives us this:
    // x{n+1} = argmin(f(x) + ρ/2 |S x - z{n} + u{n}|^2)
    // z{n+1} = argmin(g(z) + ρ/2 |S x{n+1} - z + u{n}|^2)
    // u{n+1} = u{n} + (S x{n+1} - z{n+1})
    //

    // x update:
    //
    // x{n+1} = argmin  1/2 (
    //     kx |I x - x_t|^2 +
    //     kv |V x - v_t|^2 +
    //     ka |A x - a_t|^2 +
    //     ρ  |S x - d|^2
    // )
    // d = (z{n} - u{n})

    // Rewrite in a friendlier way:
    // |A x - b|^2 = x^T A^T A x - 2 x^T A^T b + C
    // 1/2 (
    //     kx (x^T x - 2 x^T x_t) +
    //     kv (x^T V^T V x - 2 x^T V^T v_t) +
    //     ka (x^T A^T A x - 2 x^T A^T a_t) +
    //     ρ  (x^T S^T S x - 2 x^T S^T d)
    // ) + C
    //
    // 1/2 x^T (kx I + kv V^T V + ka A^T A + ρ S^T S) x
    //   - x^T (kx x_t + kv V^T v_t + ka A^T a_t + ρ S^T d)
    // + C
    //
    // voila:
    // M = kx I + kv V^T V + ka A^T A + ρ S^T S
    // r = kx x_t + kv V^T v_t + ka A^T a_t + ρ S^T d
    // E = 1/2 x^T M x - x^T r + C

    /*
    r = (
        torch.matmul(self.N, x_t - x_o_filtered) +
        self.admm_stepsize * torch.matmul(self.S.T, - u + z)
    )
    */
    Eigen::MatrixXd x_diffs(x_t);
    for(auto i : m_constrained_locs)
    {
        for(uint32_t j = 0; j < numCols; ++j)
        {
            x_diffs(i, j) = x_diffs(i,j) - x_o(i,j);
        }
    }

    Eigen::MatrixXd r = m_N * x_diffs;

    for(uint32_t i = 0; i < m_margin_locs.size(); ++i)
    {
        uint32_t ifull = m_margin_locs[i];
        for(uint32_t j = 0; j < numCols; ++j)
        {
            r(ifull, j) = r(ifull, j) + m_admm_stepsize * (z(i,j) - u(i,j));
        }
    }

    // Solve with respect to pin constraints:
    // x = U x_r + x_o
    // E = 1/2 (U x_r + x_o)^T M (U x_r + x_o) - (U x_r + x_o)^T r + C
    // E = 1/2 (x_r^T U^T + x_o^T) M (U x_r + x_o) - (x_r^T U^T + x_o^T) r + C
    // E = 1/2 (x_r^T U^T M (U x_r + x_o) + x_o^T M (U x_r + x_o)) - x_r^T U^T r - x_o^T r + C
    // E = 1/2 (x_r^T U^T M U x_r) + x_r^T U^T (M x_o - r) + C

    // minimized when x_r solves this equation:
    // U^T M U x_r + U^T (M x_o - r) = 0
    // x_r = (U^T M U)^-1 U^T (r - M x_o)

    // collapse r down to unconstrained variable set:
    // rhs = torch.matmul(self.U.T, r)

    uint32_t numRows_reduced = m_unconstrained_locs.size();
    Eigen::MatrixXd r_reduced(numRows_reduced, numCols);
    for(uint32_t i = 0; i < numRows_reduced; ++i)
    {
        uint32_t ifull = m_unconstrained_locs[i];
        for(uint32_t j = 0; j < numCols; ++j)
        {
            r_reduced(i,j) = r(ifull, j);
        }
    }

    // solve system:
    // x_r = torch.matmul(self.lhsmat_inv, rhs)
    r_reduced.conservativeResize(r_reduced.rows(), r_reduced.cols());

    Eigen::MatrixXd result;
    if(m_unconstrained_locs.size())
    {
        result = m_system_lu.solve(r_reduced);
    }

    // map back to full variable set:
    // return torch.matmul(self.U, x_r) + x_o_filtered
    for(uint32_t i = 0; i < numRows_reduced; ++i)
    {
        uint32_t ifull = m_unconstrained_locs[i];
        for(uint32_t j = 0; j < numCols; ++j)
        {
            x(ifull, j) = result(i, j);
        }
    }
    for(auto i : m_constrained_locs)
    {
        for(uint32_t j = 0; j < numCols; ++j)
        {
            x(i, j) = x_o(i, j);
        }
    }
}

void TrajectoryCorrector::z_update(
    Eigen::MatrixXd &z,
    const Eigen::MatrixXd &x,
    const Eigen::MatrixXd &z_t,
    const Eigen::MatrixXd &u
) const
{
    uint32_t numCols = uint32_t(z.cols());

    for(uint32_t i = 0; i < m_margin_locs.size(); ++i)
    {

        // z_diffs = S x + u - z_t
        uint32_t ifull = m_margin_locs[i];
        for(uint32_t j = 0; j < numCols; ++j)
        {
            z(i, j) = x(ifull, j) + u(i, j) - z_t(i, j);
        }

        // find the norm of the current z diff vector:
        double z_diff_norm = 0.0;
        for(uint32_t j = 0; j < numCols; ++j)
        {
            double z_diff = z(i, j);
            z_diff_norm += z_diff * z_diff;
        }
        z_diff_norm = sqrt(z_diff_norm);

        // if the norm is greater than the margin size, we need to rescale
        // the diff:
        if( z_diff_norm > m_margin_vals[i] )
        {
            for(uint32_t j = 0; j < numCols; ++j)
            {
                z(i, j) = z(i, j) * m_margin_vals[i] / z_diff_norm;
            }
        }

        // add the diff back on to the target:
        for(uint32_t j = 0; j < numCols; ++j)
        {
            z(i, j) = z_t(i, j) + z(i, j);
        }
    }
}

void TrajectoryCorrector::u_update(
    Eigen::MatrixXd &u,
    const Eigen::MatrixXd &x,
    const Eigen::MatrixXd &z
) const
{
    uint32_t numCols = uint32_t(z.cols());

    // u += S x - z
    for(uint32_t i = 0; i < m_margin_locs.size(); ++i)
    {
        uint32_t ifull = m_margin_locs[i];
        for(uint32_t j = 0; j < numCols; ++j)
        {
            u(i,j) += x(ifull, j) - z(i,j);
        }
    }
}
