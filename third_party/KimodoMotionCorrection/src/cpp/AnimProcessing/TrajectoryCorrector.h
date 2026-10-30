/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <Eigen/Sparse>

class TrajectoryCorrector
{
public:

	static void computeDiffMats(
		Eigen::SparseMatrix<double>& V,
		Eigen::SparseMatrix<double>& A,
		uint32_t N,
		const Eigen::VectorXd& velocityWeights = Eigen::VectorXd(),
		Eigen::MatrixXd* v_rhs = nullptr,
		Eigen::MatrixXd* a_rhs = nullptr
	);

	TrajectoryCorrector(
		const Eigen::VectorXd& margins,
        float pos_weight,
        float vel_weight,
        float acc_weight,
		const Eigen::VectorXd& velocityWeights = Eigen::VectorXd(),
		uint32_t admm_iters=100 );

	void Interpolate(
		Eigen::MatrixXd& ret,
		const Eigen::MatrixXd& observations,
		const Eigen::MatrixXd& ref_positions
	) const;

	void x_update(
		Eigen::MatrixXd& x,
		const Eigen::MatrixXd& z,
		const Eigen::MatrixXd& u,
		const Eigen::MatrixXd& x_t,
		const Eigen::MatrixXd& x_o
	) const;

	void z_update(
		Eigen::MatrixXd& z,
		const Eigen::MatrixXd& x,
		const Eigen::MatrixXd& z_t,
		const Eigen::MatrixXd& u
	) const;

	void u_update(
		Eigen::MatrixXd& u,
		const Eigen::MatrixXd& x,
		const Eigen::MatrixXd& z
	) const;

	float admm_stepsize() const { return m_admm_stepsize; }

	const std::vector<uint32_t>& margin_locs() { return m_margin_locs; }

private:

	Eigen::SparseMatrix<double> m_N;
	Eigen::SparseLU<Eigen::SparseMatrix<double>> m_system_lu;

	uint32_t m_admm_iters;

	std::vector<uint32_t> m_margin_locs;
	std::vector<double> m_margin_vals;

	std::vector<uint32_t> m_unconstrained_locs;
	std::vector<uint32_t> m_constrained_locs;

	float m_admm_stepsize;

};
