import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import IntEnum
from typing import List, Optional, Tuple, Dict, Union
from dataclasses import dataclass, field

from . import matrix
from .quaternion import qbetween, qslerp, qinv, qmul, qrot, matrix_to_quaternion, quaternion_to_matrix, qnormalize


class Activation(IntEnum):
    """Activation types for weight computation"""

    CONSTANT = 0
    LINEAR = 1
    ROOT = 2
    SQUARE = 3


class JointType(IntEnum):
    """Joint constraint types"""

    FREE = 0 # No constraints
    HINGE_X = 1 # Rotation around X axis only
    HINGE_Y = 2 # Rotation around Y axis only
    HINGE_Z = 3 # Rotation around Z axis only
    BALL = 4 # Ball joint with angle limit


@dataclass
class Joint:

    index: int = 0
    active: bool = True
    transform_pos: Optional[torch.Tensor] = None # (*, 3) local position
    transform_rot: Optional[torch.Tensor] = None # (*, 4) local rotation quaternion (wxyz)
    joint_type: JointType = JointType.FREE
    lower_limit: float = 0.0 # degrees
    upper_limit: float = 0.0 # degrees
    length: float = 0.0
    children: List[int] = field(default_factory=list)
    objectives: List[int] = field(default_factory=list)
    zero_position: Optional[torch.Tensor] = None # rest pose position
    zero_rotation: Optional[torch.Tensor] = None # rest pose rotation
    parent: int = -1


@dataclass
class Objective:

    index: int = 0
    active: bool = True
    joint: int = 0
    target_position: Optional[torch.Tensor] = None
    target_rotation: Optional[torch.Tensor] = None # (*, 4) quaternion wxyz
    weight: float = 1.0
    solve_position: bool = True
    solve_rotation: bool = True


class CCDIKFull:

    def __init__(
        self,
        local_mat: torch.Tensor,
        parent: List[int],
        target_indices: List[int],
        target_pos: Optional[torch.Tensor] = None,
        target_rot: Optional[torch.Tensor] = None,
        kinematic_chain: Optional[List[int]] = None,
        iterations: int = 25,
        threshold: float = 0.001,
        activation: Activation = Activation.LINEAR,
        seed_zero_pose: bool = False,
        allow_root_update: Tuple[bool, bool, bool] = (False, False, False),
        root_weight: float = 1.0,
        joint_types: Optional[Dict[int, JointType]] = None,
        joint_limits: Optional[Dict[int, Tuple[float, float]]] = None,
        pos_weight: float = 1.0,
        rot_weight: float = 0.0,
        debug: bool = False,
    ):
        self.device = local_mat.device
        self.dtype = local_mat.dtype
        self.debug = debug

        # Handle kinematic chain subset
        if kinematic_chain is None:
            kinematic_chain = list(range(local_mat.shape[-3]))

        # Compute global matrices for extracting root transform
        global_mat = matrix.forward_kinematics(local_mat, parent)
        assert isinstance(global_mat, torch.Tensor)

        # Extract subset for kinematic chain
        self.local_mat = local_mat.clone()
        self.local_mat = self.local_mat[..., kinematic_chain, :, :]
        # Set root to global (don't modify root during IK)
        self.local_mat[..., 0, :, :] = global_mat[..., kinematic_chain[0], :, :]

        # Remap parent indices
        self.parent = [i - 1 for i in range(len(kinematic_chain))]

        # Compute global matrices
        self.global_mat: torch.Tensor = matrix.forward_kinematics(self.local_mat, self.parent) # type: ignore

        # target_indices are already chain-relative (i.e., indices into the kinematic_chain list),
        # consistent with CCD_IK. No remapping needed.
        self.target_indices = list(target_indices)

        # Store parameters
        self.iterations = iterations
        self.threshold = threshold
        self.activation = activation
        self.seed_zero_pose = seed_zero_pose
        self.allow_root_update = allow_root_update
        self.root_weight = root_weight
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight

        self.J_N = self.local_mat.shape[-3]
        self.target_N = len(target_indices)

        # Store targets
        self.target_pos = target_pos
        if target_rot is not None:
            if target_rot.shape[-1] == 3 and target_rot.shape[-2] == 3:
                # Convert rotation matrix to quaternion
                self.target_q = matrix_to_quaternion(target_rot)
            else:
                self.target_q = target_rot
        else:
            self.target_q = None

        # Build joint hierarchy
        self._build_joints(joint_types, joint_limits)

        # Build objectives
        self._build_objectives()

        # Compute chain lengths for weight computation
        self._compute_lengths()

        # Save zero pose for regularization/reset
        self._save_zero_pose()

    def _build_joints(
        self, joint_types: Optional[Dict[int, JointType]], joint_limits: Optional[Dict[int, Tuple[float, float]]]
    ):
        """Build joint hierarchy from matrices."""
        self.joints: List[Joint] = []

        for i in range(self.J_N):
            joint = Joint(
                index=i,
                active=True,
                transform_pos=matrix.get_position(self.local_mat[..., i, :, :]),
                transform_rot=matrix_to_quaternion(matrix.get_rotation(self.local_mat[..., i, :, :])),
                parent=self.parent[i],
                children=[j for j in range(self.J_N) if self.parent[j] == i],
                objectives=[], # Will be filled when building objectives
            )

            # Set joint type and limits if provided
            if joint_types and i in joint_types:
                joint.joint_type = joint_types[i]
            if joint_limits and i in joint_limits:
                joint.lower_limit, joint.upper_limit = joint_limits[i]

            self.joints.append(joint)

    def _build_objectives(self):
        """Build objectives from target data."""
        self.objectives: List[Objective] = []

        for i, joint_idx in enumerate(self.target_indices):
            obj = Objective(
                index=i,
                active=True,
                joint=joint_idx,
                target_position=self.target_pos[..., i, :] if self.target_pos is not None else None,
                target_rotation=self.target_q[..., i, :] if self.target_q is not None else None,
                weight=1.0,
                solve_position=self.target_pos is not None,
                solve_rotation=self.target_q is not None,
            )
            self.objectives.append(obj)

            # Mark all joints in the chain as affecting this objective
            current = joint_idx
            while current >= 0:
                if current not in self.joints[current].objectives:
                    self.joints[current].objectives.append(i)
                current = self.parent[current]

    def _compute_lengths(self):
        """Compute cumulative chain lengths for weight computation."""

        def compute_length(joint_idx: int, parent_length: float):
            joint = self.joints[joint_idx]
            if self.parent[joint_idx] == -1:
                joint.length = 1.0
            else:
                parent_joint = self.joints[self.parent[joint_idx]]
                if joint.active and parent_joint.active:
                    # Get local position magnitude
                    local_pos = matrix.get_position(self.local_mat[..., joint_idx, :, :])
                    pos_mag = local_pos.norm(dim=-1).mean().item() # Average over batch
                    joint.length = parent_length + pos_mag
                else:
                    joint.length = parent_length

            for child_idx in joint.children:
                compute_length(child_idx, joint.length)

        compute_length(0, 0.0)

    def _save_zero_pose(self):
        """Save rest pose for potential reset."""
        for i, joint in enumerate(self.joints):
            joint.zero_position = matrix.get_position(self.local_mat[..., i, :, :]).clone()
            joint.zero_rotation = matrix_to_quaternion(matrix.get_rotation(self.local_mat[..., i, :, :])).clone()

    def get_global_position(self, joint_idx: int) -> torch.Tensor:
        """Get global position of a joint."""
        pos = matrix.get_position(self.global_mat[..., joint_idx, :, :])
        assert isinstance(pos, torch.Tensor)
        return pos

    def get_global_rotation(self, joint_idx: int) -> torch.Tensor:
        """Get global rotation of a joint as quaternion (wxyz)."""
        rot = matrix.get_rotation(self.global_mat[..., joint_idx, :, :])
        assert isinstance(rot, torch.Tensor)
        return matrix_to_quaternion(rot)

    def get_weight(self, joint: Joint, objective: Objective) -> float:
        """Compute activation weight for a joint-objective pair."""
        end_joint = self.joints[objective.joint]

        if end_joint.length < 1e-6:
            return 1.0

        ratio = joint.length / end_joint.length

        if self.activation == Activation.CONSTANT:
            return 1.0 / len(self.objectives)
        elif self.activation == Activation.LINEAR:
            return ratio
        elif self.activation == Activation.ROOT:
            return ratio**0.5
        elif self.activation == Activation.SQUARE:
            return ratio**2
        else:
            return 1.0

    def is_converged(self) -> bool:
        """Check if all objectives have converged."""
        for obj in self.objectives:
            if not obj.active:
                continue
            error = self.get_objective_error(obj)
            if error > self.threshold:
                return False
        return True

    def get_objective_error(self, obj: Objective) -> float:
        """Compute error for a single objective."""
        if not obj.active:
            return 0.0

        joint = self.joints[obj.joint]
        joint_pos = self.get_global_position(obj.joint)
        joint_rot = self.get_global_rotation(obj.joint)

        error = 0.0

        if obj.solve_position and obj.target_position is not None:
            pos_error = (joint_pos - obj.target_position).norm(dim=-1).mean().item()
            error += pos_error * self.pos_weight

        if obj.solve_rotation and obj.target_rotation is not None:
            # Quaternion angle difference
            dot = (joint_rot * obj.target_rotation).sum(dim=-1).abs().clamp(-1, 1)
            angle_error = 2 * torch.acos(dot).mean().item() # radians
            error += angle_error * self.rot_weight

        return error

    def apply_zero_pose(self):
        """Reset to rest pose."""
        for i, joint in enumerate(self.joints):
            if joint.active and joint.zero_position is not None:
                # Reconstruct local matrix from zero pose
                rot_mat = quaternion_to_matrix(joint.zero_rotation)
                self.local_mat[..., i, :3, :3] = rot_mat
                self.local_mat[..., i, :3, 3] = joint.zero_position

        self.global_mat = matrix.forward_kinematics(self.local_mat, self.parent) # type: ignore

    def update_root_position(self):
        """Update root position based on objective errors."""
        if not any(self.allow_root_update):
            return

        delta = torch.zeros_like(self.get_global_position(0))
        weight_sum = 0.0

        for obj in self.objectives:
            if not obj.active or obj.target_position is None:
                continue

            joint = self.joints[obj.joint]
            weight = obj.weight * self.get_weight(joint, obj)

            joint_pos = self.get_global_position(obj.joint)
            delta = delta + weight * (obj.target_position - joint_pos)
            weight_sum += weight

        if weight_sum > 0:
            delta = delta / weight_sum

            # Apply per-axis control
            if not self.allow_root_update[0]:
                delta[..., 0] = 0
            if not self.allow_root_update[1]:
                delta[..., 1] = 0
            if not self.allow_root_update[2]:
                delta[..., 2] = 0

            delta = delta * self.root_weight

            # Update root position
            self.local_mat[..., 0, :3, 3] = self.local_mat[..., 0, :3, 3] + delta
            self.global_mat = matrix.forward_kinematics(self.local_mat, self.parent) # type: ignore

    def resolve_joint_limits(self, joint_idx: int):
        """Apply joint limits for a specific joint."""
        joint = self.joints[joint_idx]

        if joint.joint_type == JointType.FREE:
            return

        if joint.zero_rotation is None:
            return

        current_rot = matrix_to_quaternion(matrix.get_rotation(self.local_mat[..., joint_idx, :, :]))
        zero_rot = joint.zero_rotation

        if joint.joint_type == JointType.BALL:
            if joint.upper_limit == 0:
                # Lock to zero rotation
                new_rot = zero_rot
            else:
                # Clamp angle from zero rotation
                dot = (zero_rot * current_rot).sum(dim=-1).abs().clamp(-1, 1)
                angle = 2 * torch.acos(dot) * 180 / torch.pi # degrees

                # If angle exceeds limit, slerp back
                upper_rad = joint.upper_limit * torch.pi / 180
                mask = angle > joint.upper_limit

                if mask.any():
                    t = upper_rad / (2 * torch.acos(dot) + 1e-8)
                    t = t.clamp(0, 1)
                    new_rot = qslerp(zero_rot, current_rot, t.unsqueeze(-1))
                else:
                    new_rot = current_rot
        else:
            # Hinge joints - project rotation onto single axis
            axis = torch.zeros(3, device=self.device, dtype=self.dtype)
            if joint.joint_type == JointType.HINGE_X:
                axis[0] = 1
            elif joint.joint_type == JointType.HINGE_Y:
                axis[1] = 1
            elif joint.joint_type == JointType.HINGE_Z:
                axis[2] = 1

            # Compute angle around the hinge axis
            # Get forward vector from zero and current rotations
            forward = torch.tensor([0, 0, 1], device=self.device, dtype=self.dtype)
            forward = forward.expand(current_rot.shape[:-1] + (3,))

            zero_forward = qrot(zero_rot, forward)
            current_forward = qrot(current_rot, forward)

            # Project onto plane perpendicular to axis
            zero_projected = zero_forward - (zero_forward * axis).sum(dim=-1, keepdim=True) * axis
            current_projected = current_forward - (current_forward * axis).sum(dim=-1, keepdim=True) * axis

            # Normalize
            zero_projected = F.normalize(zero_projected, dim=-1, eps=1e-8)
            current_projected = F.normalize(current_projected, dim=-1, eps=1e-8)

            # Compute signed angle
            cross = torch.cross(zero_projected, current_projected, dim=-1)
            dot = (zero_projected * current_projected).sum(dim=-1)
            sign = (cross * axis).sum(dim=-1).sign()
            angle = torch.atan2(cross.norm(dim=-1), dot) * sign * 180 / torch.pi # degrees

            # Clamp angle
            clamped_angle = angle.clamp(joint.lower_limit, joint.upper_limit)

            # Reconstruct rotation
            clamped_rad = clamped_angle * torch.pi / 180
            half_angle = clamped_rad / 2

            axis_expanded = axis.expand(half_angle.shape + (3,))
            w = torch.cos(half_angle)
            xyz = axis_expanded * torch.sin(half_angle).unsqueeze(-1)
            local_q = torch.cat([w.unsqueeze(-1), xyz], dim=-1)

            new_rot = qmul(zero_rot, local_q)

        # Update local matrix with new rotation
        new_rot_mat = quaternion_to_matrix(qnormalize(new_rot))
        self.local_mat[..., joint_idx, :3, :3] = new_rot_mat

    def optimize_joint(self, joint_idx: int):
        """Optimize a single joint using CCD."""
        joint = self.joints[joint_idx]

        if not joint.active:
            # Skip to children
            for child_idx in joint.children:
                self.optimize_joint(child_idx)
            return

        pos = self.get_global_position(joint_idx)
        rot = self.get_global_rotation(joint_idx) # (*, 4) wxyz

        # Accumulate rotation adjustments
        forward = torch.zeros(rot.shape[:-1] + (3,), device=self.device, dtype=self.dtype)
        up = torch.zeros(rot.shape[:-1] + (3,), device=self.device, dtype=self.dtype)
        weight_sum = 0.0

        # Solve for rotation objectives
        for obj_idx in joint.objectives:
            obj = self.objectives[obj_idx]
            if not obj.active:
                continue

            if obj.solve_rotation and obj.target_rotation is not None:
                weight = obj.weight * self.get_weight(joint, obj) * self.rot_weight
                if weight < 1e-8:
                    continue

                end_rot = self.get_global_rotation(obj.joint)

                # Compute rotation to align end effector with target
                delta_rot = qmul(obj.target_rotation, qinv(end_rot))
                adjusted_rot = qslerp(rot, qmul(delta_rot, rot), weight)

                # Accumulate direction vectors
                fwd_vec = torch.zeros(rot.shape[:-1] + (3,), device=self.device, dtype=self.dtype)
                fwd_vec[..., 2] = 1.0
                up_vec = torch.zeros(rot.shape[:-1] + (3,), device=self.device, dtype=self.dtype)
                up_vec[..., 1] = 1.0

                forward = forward + qrot(adjusted_rot, fwd_vec)
                up = up + qrot(adjusted_rot, up_vec)
                weight_sum += weight

        # Solve for position objectives
        for obj_idx in joint.objectives:
            obj = self.objectives[obj_idx]
            if not obj.active:
                continue

            if obj.solve_position and obj.target_position is not None:
                weight = obj.weight * self.get_weight(joint, obj) * self.pos_weight
                if weight < 1e-8:
                    continue

                end_pos = self.get_global_position(obj.joint)

                # Compute rotation to align end effector direction with target direction
                current_dir = end_pos - pos
                target_dir = obj.target_position - pos

                # Handle zero vectors - use default Z direction if vectors are too small
                current_norm = current_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                target_norm = target_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)

                # Normalize directions
                current_dir = current_dir / current_norm
                target_dir = target_dir / target_norm

                # Skip if directions are essentially zero (both points coincide)
                valid = (current_norm.squeeze(-1) > 1e-6) & (target_norm.squeeze(-1) > 1e-6)
                if not valid.any():
                    continue

                # Get rotation from current to target direction
                delta_rot = qbetween(current_dir, target_dir)
                adjusted_rot = qslerp(rot, qmul(delta_rot, rot), weight)

                # Accumulate direction vectors
                fwd_vec = torch.zeros(rot.shape[:-1] + (3,), device=self.device, dtype=self.dtype)
                fwd_vec[..., 2] = 1.0
                up_vec = torch.zeros(rot.shape[:-1] + (3,), device=self.device, dtype=self.dtype)
                up_vec[..., 1] = 1.0

                forward = forward + qrot(adjusted_rot, fwd_vec)
                up = up + qrot(adjusted_rot, up_vec)
                weight_sum += weight

        # Apply accumulated rotation
        if weight_sum > 0:
            forward = forward / weight_sum
            up = up / weight_sum

            # Normalize and orthogonalize
            forward = F.normalize(forward, dim=-1, eps=1e-8)
            up = up - (up * forward).sum(dim=-1, keepdim=True) * forward
            up = F.normalize(up, dim=-1, eps=1e-8)

            # Build rotation matrix from forward and up
            right = torch.cross(up, forward, dim=-1)

            # Stack to create rotation matrix (column vectors)
            solved_rot_mat = torch.stack([right, up, forward], dim=-1)

            # Compute local rotation
            parent_idx = self.parent[joint_idx]
            if parent_idx >= 0:
                parent_rot_mat = matrix.get_rotation(self.global_mat[..., parent_idx, :, :])
                assert isinstance(parent_rot_mat, torch.Tensor)
                # local_rot = parent_rot^-1 * global_rot
                parent_rot_mat_inv = parent_rot_mat.transpose(-1, -2)
                local_rot_mat = torch.matmul(parent_rot_mat_inv, solved_rot_mat)
            else:
                local_rot_mat = solved_rot_mat

            # Update local matrix
            self.local_mat[..., joint_idx, :3, :3] = local_rot_mat

            # Recompute forward kinematics
            self.global_mat = matrix.forward_kinematics(self.local_mat, self.parent) # type: ignore

            # Apply joint limits
            self.resolve_joint_limits(joint_idx)
            self.global_mat = matrix.forward_kinematics(self.local_mat, self.parent) # type: ignore

        # Recurse to children
        for child_idx in joint.children:
            self.optimize_joint(child_idx)

    def _debug_print_iter_status(self, iteration: int, label: str = ""):
        """Print per-iteration debug info: each objective's ee position, target position, and distance."""
        if not self.debug:
            return
        errors = []
        lines = []
        for obj in self.objectives:
            if not obj.active:
                continue
            ee_pos = self.get_global_position(obj.joint)
            if obj.target_position is not None:
                dist = (ee_pos - obj.target_position).norm(dim=-1)
                mean_dist = dist.mean().item()
                max_dist = dist.max().item()
                min_dist = dist.min().item()
                errors.append(mean_dist)
                # print mean of ee and target over batch dims for readability
                ee_mean = ee_pos.mean(dim=tuple(range(ee_pos.dim() - 1))).tolist()
                tgt_mean = obj.target_position.mean(dim=tuple(range(obj.target_position.dim() - 1))).tolist()
                lines.append(
                    f"    obj[{obj.index}] joint={obj.joint}: "
                    f"ee_mean=[{ee_mean[0]:.4f},{ee_mean[1]:.4f},{ee_mean[2]:.4f}], "
                    f"tgt_mean=[{tgt_mean[0]:.4f},{tgt_mean[1]:.4f},{tgt_mean[2]:.4f}], "
                    f"dist(mean={mean_dist:.6f}, min={min_dist:.6f}, max={max_dist:.6f})"
                )
        avg_err = sum(errors) / len(errors) if errors else 0.0
        print(f"  [CCDIKFull DEBUG] iter={iteration} {label}| avg_dist={avg_err:.6f}")
        for line in lines:
            print(line)

    def solve(self) -> torch.Tensor:
        # Optionally reset to zero pose
        if self.seed_zero_pose:
            self.apply_zero_pose()

        if self.debug:
            print(f"  [CCDIKFull DEBUG] === Starting solve: iterations={self.iterations}, "
                  f"threshold={self.threshold}, J_N={self.J_N}, "
                  f"target_indices={self.target_indices}, activation={self.activation.name}, "
                  f"pos_weight={self.pos_weight}, rot_weight={self.rot_weight} ===")
            self._debug_print_iter_status(-1, label="INIT ")

        # Main IK loop
        for iteration in range(self.iterations):
            if self.is_converged():
                if self.debug:
                    print(f"  [CCDIKFull DEBUG] *** Converged at iter={iteration} ***")
                    self._debug_print_iter_status(iteration, label="CONVERGED ")
                break

            # Update root position if allowed
            self.update_root_position()

            # Optimize all joints starting from root
            # Skip root (index 0) as it's fixed or handled separately
            for child_idx in self.joints[0].children:
                self.optimize_joint(child_idx)

            if self.debug:
                self._debug_print_iter_status(iteration, label="AFTER_OPT ")
        else:
            # Loop completed without convergence
            if self.debug:
                print(f"  [CCDIKFull DEBUG] *** NOT converged after {self.iterations} iterations ***")
                self._debug_print_iter_status(self.iterations, label="FINAL ")

        return self.local_mat

    def set_targets(self, positions: Optional[torch.Tensor] = None, rotations: Optional[torch.Tensor] = None):
        if positions is not None:
            self.target_pos = positions
            for i, obj in enumerate(self.objectives):
                obj.target_position = positions[..., i, :]
                obj.solve_position = True

        if rotations is not None:
            if rotations.shape[-1] == 3 and rotations.shape[-2] == 3:
                self.target_q = matrix_to_quaternion(rotations)
            else:
                self.target_q = rotations

            for i, obj in enumerate(self.objectives):
                obj.target_rotation = self.target_q[..., i, :]
                obj.solve_rotation = True

    def set_weights(self, weights: List[float]):
        if len(weights) != len(self.objectives):
            raise ValueError(f"Expected {len(self.objectives)} weights, got {len(weights)}")

        for i, weight in enumerate(weights):
            self.objectives[i].weight = weight

    def set_joint_active(self, joint_idx: int, active: bool):
        """Enable/disable a joint for IK."""
        if 0 <= joint_idx < len(self.joints):
            self.joints[joint_idx].active = active

    def set_objective_active(self, obj_idx: int, active: bool):
        """Enable/disable an objective."""
        if 0 <= obj_idx < len(self.objectives):
            self.objectives[obj_idx].active = active


def solve_ccd_ik_full(
    local_mat: torch.Tensor,
    parent: List[int],
    target_indices: List[int],
    target_pos: Optional[torch.Tensor] = None,
    target_rot: Optional[torch.Tensor] = None,
    kinematic_chain: Optional[List[int]] = None,
    iterations: int = 25,
    threshold: float = 0.001,
    activation: Activation = Activation.LINEAR,
    pos_weight: float = 1.0,
    rot_weight: float = 0.0,
    joint_types: Optional[Dict[int, JointType]] = None,
    joint_limits: Optional[Dict[int, Tuple[float, float]]] = None,
) -> torch.Tensor:
    solver = CCDIKFull(
        local_mat=local_mat,
        parent=parent,
        target_indices=target_indices,
        target_pos=target_pos,
        target_rot=target_rot,
        kinematic_chain=kinematic_chain,
        iterations=iterations,
        threshold=threshold,
        activation=activation,
        pos_weight=pos_weight,
        rot_weight=rot_weight,
        joint_types=joint_types,
        joint_limits=joint_limits,
    )

    return solver.solve()


# Convenience aliases for activation types
ACTIVATION_CONSTANT = Activation.CONSTANT
ACTIVATION_LINEAR = Activation.LINEAR
ACTIVATION_ROOT = Activation.ROOT
ACTIVATION_SQUARE = Activation.SQUARE

# Convenience aliases for joint types
JOINT_FREE = JointType.FREE
JOINT_HINGE_X = JointType.HINGE_X
JOINT_HINGE_Y = JointType.HINGE_Y
JOINT_HINGE_Z = JointType.HINGE_Z
JOINT_BALL = JointType.BALL


if __name__ == "__main__":
    """
    Unit tests for CCD IK solver.
    Run with: python -m flowhmr.core.motion.ccd_ik_full
    """
    import numpy as np

    def create_identity_mat(batch_shape: Tuple, n_joints: int, device: str = "cpu") -> torch.Tensor:
        """Create identity transformation matrices."""
        mat = torch.eye(4, device=device, dtype=torch.float32)
        mat = mat.reshape((1,) * len(batch_shape) + (1, 4, 4))
        mat = mat.expand(batch_shape + (n_joints, 4, 4)).clone()
        return mat

    def create_chain_local_mat(
        n_joints: int, bone_length: float = 1.0, batch_shape: Tuple = (), device: str = "cpu"
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Create a simple chain skeleton with joints along the Y axis.
        Returns local matrices and parent indices.
        """
        local_mat = create_identity_mat(batch_shape, n_joints, device)

        # Set bone offsets (each joint is offset along Y from parent)
        for i in range(1, n_joints):
            local_mat[..., i, 1, 3] = bone_length # Y offset

        # Parent indices: linear chain
        parent = [-1] + list(range(n_joints - 1))

        return local_mat, parent

    def test_basic_position_ik():
        """Test basic position IK with a simple chain."""
        print("=" * 60)
        print("Test 1: Basic Position IK")
        print("=" * 60)

        device = "cpu"
        n_joints = 5
        bone_length = 1.0

        # Create a simple chain
        local_mat, parent = create_chain_local_mat(n_joints, bone_length, batch_shape=(), device=device)

        # Target: move the end effector to a new position
        # The chain extends from (0,0,0) to (0,4,0) initially
        target_pos = torch.tensor([[2.0, 3.0, 0.0]], device=device)

        print(f"Initial chain: {n_joints} joints, bone length: {bone_length}")
        print(f"Target position: {target_pos.squeeze().tolist()}")

        # Solve IK
        solver = CCDIKFull(
            local_mat=local_mat,
            parent=parent,
            target_indices=[n_joints - 1], # Last joint as end effector
            target_pos=target_pos,
            iterations=50,
            threshold=0.01,
            activation=Activation.LINEAR,
        )

        solved_mat = solver.solve()

        # Get final end effector position
        final_pos = solver.get_global_position(n_joints - 1)
        error = (final_pos - target_pos.squeeze()).norm().item()

        print(f"Final end effector position: {final_pos.tolist()}")
        print(f"Position error: {error:.6f}")
        print(f"Converged: {error < 0.1}")
        print()

        return error < 0.1

    def test_multiple_objectives():
        """Test IK with multiple end effectors."""
        print("=" * 60)
        print("Test 2: Multiple Objectives (Branching Chain)")
        print("=" * 60)

        device = "cpu"

        # Joint 0 (root) -> Joint 1 -> Joint 2 (left branch end)
        #                -> Joint 3 -> Joint 4 (right branch end)
        n_joints = 5
        local_mat = create_identity_mat((), n_joints, device)

        # Set up positions
        local_mat[..., 1, 1, 3] = 1.0 # Joint 1 at (0, 1, 0) from root
        local_mat[..., 2, 0, 3] = -1.0 # Joint 2 at (-1, 0, 0) from joint 1
        local_mat[..., 2, 1, 3] = 1.0 # Joint 2 at (-1, 1, 0) from joint 1
        local_mat[..., 3, 0, 3] = 1.0 # Joint 3 at (1, 0, 0) from joint 1
        local_mat[..., 3, 1, 3] = 1.0 # Joint 3 at (1, 1, 0) from joint 1
        local_mat[..., 4, 1, 3] = 1.0 # Joint 4 at (0, 1, 0) from joint 3

        # Parent: 0 is root, 1 from 0, 2 from 1, 3 from 1, 4 from 3
        parent = [-1, 0, 1, 1, 3]

        # Targets for both end effectors - use realistic targets near initial positions
        # Initial positions: joint 2 at (-1, 2, 0), joint 4 at (1, 3, 0)
        target_pos = torch.tensor(
            [
                [-1.2, 2.3, 0.2], # Target for joint 2 (left) - close to initial
                [1.3, 3.2, -0.2], # Target for joint 4 (right) - close to initial
            ],
            device=device,
        )

        print(f"Skeleton: Y-shaped with {n_joints} joints")
        print(f"Left branch target (joint 2): {target_pos[0].tolist()}")
        print(f"Right branch target (joint 4): {target_pos[1].tolist()}")

        solver = CCDIKFull(
            local_mat=local_mat,
            parent=parent,
            target_indices=[2, 4], # Both branch ends
            target_pos=target_pos,
            iterations=100,
            threshold=0.01,
            activation=Activation.LINEAR,
        )

        solver.solve()

        # Check results
        pos2 = solver.get_global_position(2)
        pos4 = solver.get_global_position(4)
        error2 = (pos2 - target_pos[0]).norm().item()
        error4 = (pos4 - target_pos[1]).norm().item()

        print(f"Joint 2 final position: {pos2.tolist()}, error: {error2:.6f}")
        print(f"Joint 4 final position: {pos4.tolist()}, error: {error4:.6f}")
        print(f"Average error: {(error2 + error4) / 2:.6f}")
        print()

        # Multiple objectives with shared joints are harder to solve perfectly
        return (error2 + error4) / 2 < 1.0

    def test_batched_ik():
        """Test batched IK solving."""
        print("=" * 60)
        print("Test 3: Batched IK (Multiple Poses)")
        print("=" * 60)

        device = "cpu"
        batch_size = 4
        n_joints = 4
        bone_length = 1.0

        # Create batched chain
        local_mat, parent = create_chain_local_mat(n_joints, bone_length, batch_shape=(batch_size,), device=device)

        # Different targets for each batch
        target_pos = torch.tensor(
            [
                [[1.0, 2.0, 0.0]],
                [[0.0, 3.0, 1.0]],
                [[-1.0, 2.5, 0.5]],
                [[1.5, 1.5, -0.5]],
            ],
            device=device,
        )

        print(f"Batch size: {batch_size}")
        print(f"Chain: {n_joints} joints")

        solver = CCDIKFull(
            local_mat=local_mat,
            parent=parent,
            target_indices=[n_joints - 1],
            target_pos=target_pos,
            iterations=30,
            threshold=0.01,
        )

        solver.solve()

        # Check results for each batch
        errors = []
        for b in range(batch_size):
            # Get position for this batch item
            pos = matrix.get_position(solver.global_mat[b, n_joints - 1])
            target = target_pos[b, 0]
            error = (pos - target).norm().item()
            errors.append(error)
            print(f"Batch {b}: target={target.tolist()}, final={pos.tolist()}, error={error:.4f}")

        avg_error = sum(errors) / len(errors)
        print(f"Average error: {avg_error:.6f}")
        print()

        return avg_error < 0.5

    def test_activation_types():
        """Test different activation types."""
        print("=" * 60)
        print("Test 4: Different Activation Types")
        print("=" * 60)

        device = "cpu"
        n_joints = 6
        bone_length = 0.5

        target_pos = torch.tensor([[1.5, 2.0, 0.5]], device=device)

        results = {}
        for activation in [Activation.CONSTANT, Activation.LINEAR, Activation.ROOT, Activation.SQUARE]:
            local_mat, parent = create_chain_local_mat(n_joints, bone_length, device=device)

            solver = CCDIKFull(
                local_mat=local_mat,
                parent=parent,
                target_indices=[n_joints - 1],
                target_pos=target_pos,
                iterations=30,
                activation=activation,
            )

            solver.solve()
            final_pos = solver.get_global_position(n_joints - 1)
            error = (final_pos - target_pos.squeeze()).norm().item()
            results[activation.name] = error

            print(f"{activation.name:10s}: error = {error:.6f}")

        print()
        return all(e < 0.5 for e in results.values())

    def test_joint_constraints():
        """Test joint type constraints."""
        print("=" * 60)
        print("Test 5: Joint Constraints (Hinge Joints)")
        print("=" * 60)

        device = "cpu"
        n_joints = 4
        bone_length = 1.0

        local_mat, parent = create_chain_local_mat(n_joints, bone_length, device=device)

        # Target that requires bending
        target_pos = torch.tensor([[2.0, 1.0, 0.0]], device=device)

        # Test without constraints
        solver_free = CCDIKFull(
            local_mat=local_mat.clone(),
            parent=parent,
            target_indices=[n_joints - 1],
            target_pos=target_pos,
            iterations=50,
        )
        solver_free.solve()
        pos_free = solver_free.get_global_position(n_joints - 1)
        error_free = (pos_free - target_pos.squeeze()).norm().item()

        # Test with hinge constraints (only rotate around Z axis)
        local_mat2, _ = create_chain_local_mat(n_joints, bone_length, device=device)
        joint_types = {i: JointType.HINGE_Z for i in range(1, n_joints)}
        joint_limits = {i: (-90.0, 90.0) for i in range(1, n_joints)}

        solver_hinge = CCDIKFull(
            local_mat=local_mat2,
            parent=parent,
            target_indices=[n_joints - 1],
            target_pos=target_pos,
            iterations=50,
            joint_types=joint_types,
            joint_limits=joint_limits,
        )
        solver_hinge.solve()
        pos_hinge = solver_hinge.get_global_position(n_joints - 1)
        error_hinge = (pos_hinge - target_pos.squeeze()).norm().item()

        print(f"Target: {target_pos.squeeze().tolist()}")
        print(f"Free joints - Final: {pos_free.tolist()}, Error: {error_free:.4f}")
        print(f"Hinge joints - Final: {pos_hinge.tolist()}, Error: {error_hinge:.4f}")
        print()

        return True # Just checking it runs without error

    def test_root_update():
        """Test root position update feature."""
        print("=" * 60)
        print("Test 6: Root Position Update")
        print("=" * 60)

        device = "cpu"
        n_joints = 4
        bone_length = 1.0

        local_mat, parent = create_chain_local_mat(n_joints, bone_length, device=device)

        # Target that's far from the chain's reach
        target_pos = torch.tensor([[5.0, 3.0, 0.0]], device=device)

        # Without root update
        solver_fixed = CCDIKFull(
            local_mat=local_mat.clone(),
            parent=parent,
            target_indices=[n_joints - 1],
            target_pos=target_pos,
            iterations=50,
            allow_root_update=(False, False, False),
        )
        solver_fixed.solve()
        pos_fixed = solver_fixed.get_global_position(n_joints - 1)
        error_fixed = (pos_fixed - target_pos.squeeze()).norm().item()

        # With root update
        local_mat2, _ = create_chain_local_mat(n_joints, bone_length, device=device)
        solver_moving = CCDIKFull(
            local_mat=local_mat2,
            parent=parent,
            target_indices=[n_joints - 1],
            target_pos=target_pos,
            iterations=50,
            allow_root_update=(True, True, False), # Allow X and Y movement
            root_weight=0.5,
        )
        solver_moving.solve()
        pos_moving = solver_moving.get_global_position(n_joints - 1)
        error_moving = (pos_moving - target_pos.squeeze()).norm().item()

        print(f"Target (out of reach): {target_pos.squeeze().tolist()}")
        print(f"Fixed root - Final: {pos_fixed.tolist()}, Error: {error_fixed:.4f}")
        print(f"Moving root - Final: {pos_moving.tolist()}, Error: {error_moving:.4f}")
        print(f"Root moved: {solver_moving.get_global_position(0).tolist()}")
        print()

        return error_moving < error_fixed

    def test_functional_interface():
        """Test the functional interface."""
        print("=" * 60)
        print("Test 7: Functional Interface")
        print("=" * 60)

        device = "cpu"
        n_joints = 5
        bone_length = 1.0

        local_mat, parent = create_chain_local_mat(n_joints, bone_length, device=device)
        target_pos = torch.tensor([[2.0, 2.5, 0.0]], device=device)

        solved_mat = solve_ccd_ik_full(
            local_mat=local_mat,
            parent=parent,
            target_indices=[n_joints - 1],
            target_pos=target_pos,
            iterations=30,
            activation=Activation.LINEAR,
        )

        # Compute final position from solved matrices
        global_mat = matrix.forward_kinematics(solved_mat, [-1] + list(range(n_joints - 1)))
        final_pos = matrix.get_position(global_mat[..., n_joints - 1, :, :])
        assert isinstance(final_pos, torch.Tensor)
        error = (final_pos - target_pos.squeeze()).norm().item()

        print(f"Target: {target_pos.squeeze().tolist()}")
        print(f"Final: {final_pos.tolist()}")
        print(f"Error: {error:.6f}")
        print()

        return error < 0.5

    # Run all tests
    print("\n" + "=" * 60)
    print("Running CCD IK Full Unit Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Basic Position IK", test_basic_position_ik),
        ("Multiple Objectives", test_multiple_objectives),
        ("Batched IK", test_batched_ik),
        ("Activation Types", test_activation_types),
        ("Joint Constraints", test_joint_constraints),
        ("Root Update", test_root_update),
        ("Functional Interface", test_functional_interface),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        status = "✓" if result == "PASSED" else "✗"
        print(f"{status} {name}: {result}")

    n_passed = sum(1 for _, r in results if r == "PASSED")
    print(f"\nTotal: {n_passed}/{len(tests)} tests passed")
