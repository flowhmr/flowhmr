/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TrajectoryCorrector.h"
#include "InverseKinematics.h"

#include "Utility.h"

#include <map>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
using Pose = std::vector<Math::Transform>;

static const float pos_weight = 0.001f;
static const float vel_weight = 1.0f;
static const float acc_weight = 10.0f;


namespace {

    // Enable with: MOTIONCORRECTION_DEBUG_INTERVALS=1
    // Default: off (no Interval printing).
    bool DebugPrintIntervalsEnabled()
    {
        const char* v = std::getenv("MOTIONCORRECTION_DEBUG_INTERVALS");
        if (v == nullptr || v[0] == '\0')
        {
            return false;
        }
        // Treat "0" as false; any other non-empty value enables.
        return v[0] != '0';
    }


    void FilterContactIntervals(
        std::vector<std::pair<int, int>>& contactIntervals,
        const std::vector<float>& mask,
        bool one_bone_contact = false)
    {
        std::vector<uint32_t> keepIntervals;
        for (size_t i = 0; i < contactIntervals.size(); ++i)
        {
            const auto& interval = contactIntervals[i];

            bool startConstrained = (interval.first != 0 && mask[interval.first - 1]);
            bool endConstrained;

            endConstrained = (interval.second != mask.size() && mask[interval.second]);

            if (one_bone_contact)
            {
                if (startConstrained || endConstrained)
                {
                    continue;
                }
            }
            else
            {
                // If both the start and end of the contact interval are masked,
                // there's no way we can correct the contact without popping, so
                // let's filter these out:
                if (startConstrained && endConstrained)
                {
                    continue;
                }
            }

            keepIntervals.push_back(i);
        }

        for (size_t i = 0; i < keepIntervals.size(); ++i)
        {
            contactIntervals[i] = contactIntervals[keepIntervals[i]];
        }
        contactIntervals.resize(keepIntervals.size());
    }

    std::vector<std::pair<int, int>> ComputeContactIntervals(
        const std::vector<float>& contacts,
        const std::vector<float>& mask,
        float contactThreshold)
    {
        // turn off the contacts for all frames that are constrained/masked:
        std::vector<float> contactsNoMask = contacts;
        for (size_t i = 0; i < mask.size(); ++i)
        {
            if (mask[i])
            {
                contactsNoMask[i] = 0;
            }
        }

        // Find intervals that are in contact:
        std::vector<std::pair<int, int>> contactIntervals;
        int start = -1;
        for (int frame = 0; frame < mask.size(); ++frame)
        {
            bool isContact = contactsNoMask[frame] > contactThreshold;
            if (isContact && start == -1)
            {
                start = frame;
            }
            else if (!isContact && start != -1)
            {
                contactIntervals.emplace_back(start, frame);
                start = -1;
            }
        }

        // Close the final interval if needed:
        if (start != -1)
        {
            contactIntervals.emplace_back(start, mask.size());
        }
        return contactIntervals;
    }

    void FindContactPoints(
        std::vector<Math::Vector> &points,
        std::vector<int> &inContact,
        const std::vector<int>& joint_parents_vec,
        int32_t jointIndex,
        const std::vector<Pose> &poses,
        const std::vector<std::pair<int, int>>& contactIntervals,
        const std::vector<float>& mask,
        size_t frameCount,
        float minHeight)
    {
        // Find a representative frame for each interval.
        // If the interval starts after a masked frame, use the start
        // of the interval, if it ends before a mask use the end,
        // otherwise use the middle frame.
        inContact.clear();
        inContact.resize(frameCount, 0);
        points.clear();
        points.resize(frameCount);
        for (size_t i = 0; i < contactIntervals.size(); ++i)
        {
            const auto& interval = contactIntervals[i];
            int frame = -1;
            bool startConstrained = (interval.first != 0 && mask[interval.first - 1]);
            bool endConstrained;

            endConstrained = (interval.second != mask.size() && mask[interval.second]);

            // Debug output (opt-in via env var)
            if (DebugPrintIntervalsEnabled())
            {
                std::cout << "Interval " << i << ": start=" << interval.first
                          << ", end=" << interval.second
                          << ", startConstrained=" << startConstrained
                          << ", endConstrained=" << endConstrained << std::endl;
            }

            if(startConstrained)
            {
                // If the interval starts on a constraint, use the constrained frame
                // as a target (doing this modulo mask.size() in case we're looping)
                frame = interval.first - 1;
            }
            else if (endConstrained)
            {
                // If the interval ends on a constraint, use the constrained frame
                // as a target:
                frame = interval.second;
            }
            else
            {
                // Otherwise use the midpoint of the interval:
                frame = (interval.first + interval.second) / 2;
            }

            // get the target point:
            Math::Vector target = Animation::JointLocalToGlobal(joint_parents_vec, jointIndex, poses[frame]).GetTranslation();
            for(int i = interval.first; i < interval.second; ++i)
            {
                Math::Vector framePt = Animation::JointLocalToGlobal(joint_parents_vec, jointIndex, poses[i]).GetTranslation();
                inContact[i] = 1;
                points[i] = target;
                if (!startConstrained && !endConstrained)
                {
                    points[i].SetY(std::max(framePt.GetY(), minHeight));
                    // std::cout << "  Frame " << i << ": SetY with framePt.GetY()=" << framePt.GetY()
                    //           << ", minHeight=" << minHeight << std::endl;
                }
            }
        }
    }

    float TargetReachFalloff(
        const std::vector<int>& joint_parents_vec,
        const Pose& defaultPose,
        int32_t jointIndex,
        Animation::IKType ikType,
        const Math::Vector& target,
        const Pose& pose,
        const Math::Transform& rootTx = Math::Transform::Identity)
    {
        float maxReach = defaultPose[jointIndex].GetTranslation().GetLength3();
        if (ikType == Animation::IKType::kTwoBone)
        {
            jointIndex = joint_parents_vec[jointIndex];
            ASSERT(jointIndex > -1);
            maxReach += defaultPose[jointIndex].GetTranslation().GetLength3();
        }
        // Get base joint world Tx
        jointIndex = joint_parents_vec[jointIndex];
        ASSERT(jointIndex > -1);
        const auto worldTx = Animation::JointLocalToGlobal(joint_parents_vec, jointIndex, pose, rootTx);

        // Gaussian falloff
        float targetDist = target.GetDistance3(worldTx.GetTranslation());
        float tmp = Math::Max(targetDist / maxReach - 0.99f, 0.f) / 0.01f;
        tmp = tmp * tmp;
        return std::exp(-2.f * tmp * tmp);
    }

    void CorrectHipsY(
        std::vector<Pose>& poses,
        const std::vector<Pose>& targetPoses,
        const std::vector<float>& fullBodyMask,
        const std::vector<Animation::ContactInfo>& contacts,
        float contactThreshold
    )
    {
        // Correct the y coordinates of the root.
        auto N = poses.size();
        Eigen::MatrixXd x(N, 1);
        Eigen::MatrixXd observations(N, 1);
        Eigen::MatrixXd xfixed(N, 1);

        // Fill in the initial trajectory (x) and the values we want to hit when we
        // warp it (observations):
        Eigen::VectorXd yCorrectMargins(N);
        for(size_t frame = 0; frame < N; ++frame)
        {
            yCorrectMargins[frame] = fullBodyMask[frame] ? 0.0f : -1.0f;
            x(frame, 0) = ((float*)&poses[frame][0].GetTranslation())[1];
            observations(frame, 0) = ((float*)&targetPoses[frame][0].GetTranslation())[1];
        }

        TrajectoryCorrector ycorrector(
            yCorrectMargins,
            pos_weight * 10,
            vel_weight,
            acc_weight * 0.1f
        );
        ycorrector.Interpolate(
            xfixed,
            observations,
            x
        );

        // fill channel again:
        for (uint32_t frame = 0; frame < N; ++frame)
        {
            ((float*)&poses[frame][0].GetTranslation())[1] = float(xfixed(frame, 0));
        }
    }

    void SmoothChannels(
        Eigen::MatrixXd &x,
        const std::vector<float>& mask
    )
    {
        for( uint32_t i=0; i < mask.size(); ++i)
        {
            uint32_t i_prev = i == 0 ? 0 : i-1;
            uint32_t i_next = std::min(uint32_t(i+1), uint32_t(mask.size()-1));
            if(i > 0 && mask[i] > 0 && mask[i_prev] == 0)
            {
                // if the previous frame is unconstrained and the current frame is constrained,
                // replace the current frame with the average of its neighbors:
                for(long j=0; j < x.cols(); ++j)
                {
                    x(i, j) = 0.5f * (x(i_prev, j) + x(i_next, j));
                }
            }
            if(mask[i] > 0 && mask[i_next] == 0)
            {
                // if the next frame is unconstrained and the current frame is constrained,
                // replace the current frame with the average of its neighbors:
                for(long j=0; j < x.cols(); ++j)
                {
                    x(i, j) = 0.5f * (x(i_prev, j) + x(i_next, j));
                }
            }
        }
    }


    void CorrectHipsXZ(
        std::vector<Pose>& poses,
        const std::vector<Pose>& targetPoses,
        const std::vector<float>& fullBodyMask,
        const std::vector<float>& rootMask,
        const std::vector<Animation::ContactInfo>& endEffectorPins,
        const Eigen::VectorXd& velocity_weights,
        float root_margin
    )
    {
        auto N = poses.size();
        Eigen::VectorXd margins(N);
        for( size_t i = 0; i < N; ++i )
        {
            margins[i] = fullBodyMask[i] ? 0.0f : -1.0f;
        }

        std::vector<float> rootCombinedMask(N, 0.0f);
        for(size_t i = 0; i < N; ++i)
        {
            rootCombinedMask[i] = (fullBodyMask[i] > 0) || (rootMask[i] > 0);
            if(rootMask[i] > 0 && margins[i] != 0)
            {
                margins[i] = root_margin;
            }
            for (auto& c : endEffectorPins)
            {
                if (c.contactMask[i] && margins[i] != 0)
                {
                    margins[i] = root_margin;
                }
            }
        }
        TrajectoryCorrector xzcorrector(
            margins,
            pos_weight,
            vel_weight,
            acc_weight,
            velocity_weights
        );

        // Enforce pose constraints on root xz trajectory:
        Eigen::MatrixXd x(N, 2);
        Eigen::MatrixXd observations(N, 2);
        Eigen::MatrixXd x_fixed(N, 2);

        observations.setZero();
        for (uint32_t frame = 0; frame < N; ++frame)
        {
            x(frame, 0) = ((float*)&poses[frame][0].GetTranslation())[0];
            x(frame, 1) = ((float*)&poses[frame][0].GetTranslation())[2];

            observations(frame, 0) = ((float*)&targetPoses[frame][0].GetTranslation())[0];
            observations(frame, 1) = ((float*)&targetPoses[frame][0].GetTranslation())[2];
        }

        SmoothChannels(x, rootCombinedMask);

        xzcorrector.Interpolate(
            x_fixed,
            observations,
            x
        );

        // fill channels again:
        for (uint32_t frame = 0; frame < N; ++frame)
        {
            ((float*)&poses[frame][0].GetTranslation())[0] = float(x_fixed(frame, 0));
            ((float*)&poses[frame][0].GetTranslation())[2] = float(x_fixed(frame, 1));
        }
    }

    void CorrectRotationsForBone(
        std::vector<Pose>& poses,
        const std::vector<Pose>& targetPoses,
        const std::vector<float>& mask,
        const TrajectoryCorrector& corrector,
        int boneIdx,
        bool performChannelSmoothing)
    {
        auto N = poses.size();
        Eigen::MatrixXd x(N, 1);
        Eigen::MatrixXd observations(N, 1);
        observations.setZero();
        Eigen::MatrixXd x_fixed(N, 1);

        // Quaternion components can flip when they pass through 180 degree
        // rotations, so let's convert all the quaternions in this channel to
        // the forward/up vector representation, modify them, then convert back
        // to quaternions:

        // convert time series to 6d forward/up:
        std::vector<float> forwardUp(6 * N);
        std::vector<float> targetForwardUp(6 * N);
        for (uint32_t frame = 0; frame < N; ++frame)
        {
            auto q = poses[frame][boneIdx].GetRotation();
            auto forward = q.ZAxis();
            auto up = q.YAxis();
            forwardUp[N * 0 + frame] = forward.GetX();
            forwardUp[N * 1 + frame] = forward.GetY();
            forwardUp[N * 2 + frame] = forward.GetZ();
            forwardUp[N * 3 + frame] = up.GetX();
            forwardUp[N * 4 + frame] = up.GetY();
            forwardUp[N * 5 + frame] = up.GetZ();

            q = targetPoses[frame][boneIdx].GetRotation();
            forward = q.ZAxis();
            up = q.YAxis();
            targetForwardUp[N * 0 + frame] = forward.GetX();
            targetForwardUp[N * 1 + frame] = forward.GetY();
            targetForwardUp[N * 2 + frame] = forward.GetZ();
            targetForwardUp[N * 3 + frame] = up.GetX();
            targetForwardUp[N * 4 + frame] = up.GetY();
            targetForwardUp[N * 5 + frame] = up.GetZ();
        }

        // correct trajectories:
        for (uint32_t dim = 0; dim < 6; ++dim)
        {
            for (uint32_t frame = 0; frame < N; ++frame)
            {
                x(frame, 0) = forwardUp[N * dim + frame];
                observations(frame, 0) = mask[frame] * targetForwardUp[N * dim + frame];
            }

            if (performChannelSmoothing)
            {
                SmoothChannels(x, mask);
            }

            corrector.Interpolate(
                x_fixed,
                observations,
                x
            );

            // fill channel again:
            for (uint32_t frame = 0; frame < N; ++frame)
            {
                forwardUp[N * dim + frame] = float(x_fixed(frame, 0));
            }
        }

        for (uint32_t frame = 0; frame < N; ++frame)
        {
            Math::Vector forward = { forwardUp[N * 0 + frame] ,forwardUp[N * 1 + frame] ,forwardUp[N * 2 + frame] };
            Math::Vector up = { forwardUp[N * 3 + frame] ,forwardUp[N * 4 + frame] ,forwardUp[N * 5 + frame] };

            forward.Normalize3();
            up.Normalize3();

            poses[frame][boneIdx].SetRotation(Math::Quaternion::LookRotation(forward, up));
        }
    }

    void CorrectJointRotations(
        std::vector<Pose>& poses,
        const std::vector<Pose>& targetPoses,
        const std::vector<float>& fullBodyMask,
        const Eigen::VectorXd& velocity_weights
    )
    {
        auto N = poses.size();

        // Create a trajectory corrector for fixing the full body fullBodyMask positions:
        Eigen::VectorXd margins(N);
        for( size_t i = 0; i < N; ++i )
        {
            margins[i] = fullBodyMask[i] ? 0.0f : -1.0f;
        }
        TrajectoryCorrector corrector(
            margins,
            pos_weight * 10,
            vel_weight,
            acc_weight,
            velocity_weights
        );

        for (uint32_t boneIdx = 0; boneIdx < poses[0].size(); ++boneIdx)
        {
            CorrectRotationsForBone(
                poses,
                targetPoses,
                fullBodyMask,
                corrector,
                boneIdx,
                true
            );
        }
    }

    void DoEffectorIK(
        std::vector<Pose>& poses,
        const std::vector<Pose>& targetPoses,
        const std::vector<float>& fullBodyMask,
        const std::vector<Animation::ContactInfo>& endEffectorPins,
        const std::vector<int>& joint_parents_vec,
        const std::vector<Math::Transform>& defaultPose
    )
    {
        // Apply IK for effector pins
        auto N = poses.size();
        std::map<uint32_t, std::vector<float>> jointCorrectionMasks;
        std::vector<Pose> ikFixedPoses = poses;
        for (auto& c : endEffectorPins)
        {
            auto jointIdx = c.jointIndex;

            if(jointCorrectionMasks[jointIdx].empty())
            {
                // initialize to the full body constraint mask because we
                // want to constrain that anyway:
                jointCorrectionMasks[jointIdx] = fullBodyMask;
            }

            // Add a trajectory correction mask for the parent joint:
            auto parentIdx = joint_parents_vec[jointIdx];
            if(jointCorrectionMasks[parentIdx].empty())
            {
                // initialize to the full body constraint mask because we
                // want to constrain that anyway:
                jointCorrectionMasks[parentIdx] = fullBodyMask;
            }

            // Add a trajectory correction mask for its parent if this is
            // 2 bone IK:
            auto parentParentIdx = joint_parents_vec[parentIdx];
            if(c.contactType == Animation::kTwoBone)
            {
                if(jointCorrectionMasks[parentParentIdx].empty())
                {
                    // initialize to the full body constraint mask because we
                    // want to constrain that anyway:
                    jointCorrectionMasks[parentParentIdx] = fullBodyMask;
                }
            }

            for (uint32_t fixFrame = 0; fixFrame < fullBodyMask.size(); ++fixFrame)
            {
                if (c.contactMask[fixFrame])
                {
                    const auto targetGlobalTransform = Animation::JointLocalToGlobal(joint_parents_vec, jointIdx, targetPoses[fixFrame]);

                    // flag the parent joint as fixed in its correction mask:
                    jointCorrectionMasks[parentIdx][fixFrame] = 1;
                    switch(c.contactType)
                    {
                        case Animation::kOneBone:
                        {
                            IK::OneBoneIk(
                                ikFixedPoses[fixFrame],
                                Math::Transform::Identity,
                                jointIdx,
                                1.0,
                                targetGlobalTransform.GetTranslation(),
                                joint_parents_vec
                            );
                            break;
                        }
                        case Animation::kTwoBone:
                        {
                            // flag the parent parent joint as fixed in its correction mask:
                            jointCorrectionMasks[parentParentIdx][fixFrame] = 1;
                            IK::TwoBoneIk(
                                ikFixedPoses[fixFrame],
                                Math::Transform::Identity,
                                jointIdx,
                                1.0,
                                targetGlobalTransform.GetTranslation(),
                                joint_parents_vec,
                                c.hintOffset
                            );
                            break;
                        }
                    }

                    // now we need to fix things so the global rotation of the joint
                    // matches the input:
                    jointCorrectionMasks[jointIdx][fixFrame] = 1;
                    auto parentGlobalTransform = Animation::JointLocalToGlobal(joint_parents_vec, parentIdx, ikFixedPoses[fixFrame]);
                    ikFixedPoses[fixFrame][jointIdx].SetRotation(
                        targetGlobalTransform.GetRotation() * parentGlobalTransform.GetRotation().GetConjugate()
                    );

                }
            }
        }

        // Applying the effector pin IK introduces popping into the animation,
        // so let's apply the interpolator to all the joints we modified so as to
        // line the trajectory up properly again:
        Eigen::VectorXd margins(N);
        for( auto &kv : jointCorrectionMasks)
        {
            for( size_t i = 0; i < N; ++i )
            {
                margins[i] = kv.second[i] ? 0.0f : -1.0f;
            }
            TrajectoryCorrector corrector(margins, pos_weight * 10, vel_weight, acc_weight);

            CorrectRotationsForBone(
                poses,
                ikFixedPoses,
                kv.second,
                corrector,
                kv.first,
                false
            );
        }
    }

    void DoContactIK(
        std::vector<Pose>& poses,
        const std::vector<float>& fullBodyMask,
        const std::vector<Animation::ContactInfo>& contacts,
        const std::vector<Animation::ContactInfo>& endEffectorPins,
        const std::vector<int>& joint_parents_vec,
        const std::vector<Math::Transform>& defaultPose,
        float contactThreshold,
        bool has_double_ankle_joints
    )
    {
        auto N = poses.size();
        Eigen::VectorXd margins = Eigen::VectorXd::Zero(N);

        // Apply IK to stabilize limbs on contacts
        std::map<uint32_t, std::vector<float>> jointCorrectionMasks;
        std::vector<Pose> ikFixedPoses = poses;

        // Save original poses before any modifications (for double ankle correction later)
        const std::vector<Pose> originalPoses = poses;

        // Track which frames were corrected for each 2-bone contact (for double ankle correction later)
        std::map<uint32_t, std::vector<bool>> twoBoneContactFrames;

        auto addEndEffectorMask = [&](uint32_t jointIdx, uint32_t parentIdx, std::vector<float>& jointMask)
        {
            auto it = std::find_if(
                endEffectorPins.begin(), endEffectorPins.end(),
                [&](const auto &c)
                {
                    if(jointIdx == c.jointIndex)
                    {
                        return true;
                    }
                    return false;
                }
            );
            if(it == endEffectorPins.end())
            {
                // We could be correcting the toe joint, in which case we need to use
                // the parent joint instead:
                it = std::find_if(
                    endEffectorPins.begin(), endEffectorPins.end(),
                    [&](const auto &c)
                    {
                        if(parentIdx == c.jointIndex)
                        {
                            return true;
                        }
                        return false;
                    }
                );
            }
            if(it != endEffectorPins.end())
            {
                const auto &msk = it->contactMask;
                for(size_t i=0; i < msk.size(); ++i)
                {
                    if(msk[i])
                    {
                        jointMask[i] = 1.0f;
                    }
                }
            }
        };

        // Process two bone contacts first:
        for (auto& c : contacts)
        {
            if(c.contactType != Animation::kTwoBone)
            {
                continue;
            }
            const auto jointIdx = c.jointIndex;
            auto parentIdx = joint_parents_vec[jointIdx];
            auto parentParentIdx = joint_parents_vec[parentIdx];

            auto jointMask = fullBodyMask;
            addEndEffectorMask(jointIdx, parentIdx, jointMask);

            // We'll actually be modifying 3 joints here:
            // * The two joints immediately up in the hierarchy because of the 2 bone IK
            // * The joint itself because we restore its original global rotation
            if(jointCorrectionMasks[parentIdx].empty())
            {
                jointCorrectionMasks[parentIdx] = jointMask;
            }
            if(jointCorrectionMasks[parentParentIdx].empty())
            {
                jointCorrectionMasks[parentParentIdx] = jointMask;
            }
            if(jointCorrectionMasks[jointIdx].empty())
            {
                jointCorrectionMasks[jointIdx] = jointMask;
            }

            // Compute the intervals in which the joint is in contact with the floor:
            auto contactIntervals = ComputeContactIntervals(c.contactMask, jointMask, contactThreshold);
            FilterContactIntervals(contactIntervals, jointMask);

            std::vector<Math::Vector> contactPoints;
            std::vector<int> inContact;
            FindContactPoints(
                contactPoints,
                inContact,
                joint_parents_vec,
                jointIdx,
                poses,
                contactIntervals,
                jointMask,
                c.contactMask.size(),
                c.minHeight
            );

            // RGB MoCap: anchors were resolved globally before windowing.
            if (c.targetPositions.size() == N)
                for (size_t f = 0; f < N; ++f)
                    if (inContact[f]) contactPoints[f] = c.targetPositions[f];

            for (uint32_t fixFrame = 0; fixFrame < fullBodyMask.size(); ++fixFrame)
            {
                if (inContact[fixFrame])
                {
                    auto target = contactPoints[fixFrame];
                    jointCorrectionMasks[parentIdx][fixFrame] = 1.0f;
                    jointCorrectionMasks[parentParentIdx][fixFrame] = 1.0f;
                    jointCorrectionMasks[jointIdx][fixFrame] = 1.0f;

                    // Track this frame for double ankle correction later
                    if (has_double_ankle_joints)
                    {
                        if (twoBoneContactFrames[jointIdx].empty())
                            twoBoneContactFrames[jointIdx].resize(fullBodyMask.size(), false);
                        twoBoneContactFrames[jointIdx][fixFrame] = true;
                    }

                    // save the original global rotation of the joint:
                    auto jointGlobalRotation = Animation::JointLocalToGlobal(
                        joint_parents_vec,
                        jointIdx,
                        ikFixedPoses[fixFrame]
                    ).GetRotation();

                    const float w = TargetReachFalloff(
                        joint_parents_vec,
                        defaultPose,
                        jointIdx,
                        c.contactType,
                        target,
                        ikFixedPoses[fixFrame]
                    );
                    // std::cout << "Frame " << fixFrame << ": w=" << w << std::endl;

                    // apply the 2 bone IK:
                    auto origParentRotation = ikFixedPoses[fixFrame][parentIdx].GetRotation();
                    auto origParentParentRotation = ikFixedPoses[fixFrame][parentParentIdx].GetRotation();
                    IK::TwoBoneIk(
                        ikFixedPoses[fixFrame],
                        Math::Transform::Identity,
                        jointIdx,
                        1.0f,
                        target,
                        joint_parents_vec,
                        c.hintOffset
                    );
                    ikFixedPoses[fixFrame][parentIdx].SetRotation(Math::Quaternion::SLerp(origParentRotation, ikFixedPoses[fixFrame][parentIdx].GetRotation(), w));
                    ikFixedPoses[fixFrame][parentParentIdx].SetRotation(Math::Quaternion::SLerp(origParentParentRotation, ikFixedPoses[fixFrame][parentParentIdx].GetRotation(), w));

                    // restore previous global rotation of this joint:
                    auto parentGloblalRotation = Animation::JointLocalToGlobal(
                        joint_parents_vec,
                        parentIdx,
                        ikFixedPoses[fixFrame]
                    ).GetRotation();

                    jointCorrectionMasks[jointIdx][fixFrame] = 1.0f;
                    ikFixedPoses[fixFrame][jointIdx].SetRotation(
                        jointGlobalRotation * parentGloblalRotation.GetConjugate()
                    );

                    auto result = Animation::JointLocalToGlobal(
                        joint_parents_vec,
                        jointIdx,
                        ikFixedPoses[fixFrame]
                    ).GetTranslation();
                }
            }

        }

        for( auto &kv : jointCorrectionMasks)
        {
            for( size_t i = 0; i < N; ++i )
            {
                margins[i] = kv.second[i] ? 0.0f : -1.0f;
            }
            TrajectoryCorrector corrector(margins, pos_weight * 10, vel_weight, acc_weight);
            CorrectRotationsForBone(
                poses,
                ikFixedPoses,
                kv.second,
                corrector,
                kv.first,
                false
            );
        }
        jointCorrectionMasks.clear();

        // Then process one bone contacts:
        for(auto &c : contacts)
        {
            if(c.contactType != Animation::kOneBone)
            {
                continue;
            }
            const auto jointIdx = c.jointIndex;
            auto parentIdx = joint_parents_vec[jointIdx];

            // We can't touch frames that have been constrained with full body constraints
            // or the end effector constraints for this joint, so let's combine fullBodyMask
            // with the end effector mask for this joint if it exists so we can use that
            // information later:
            auto jointMask = fullBodyMask;
            addEndEffectorMask(jointIdx, parentIdx, jointMask);

            // Add a trajectory correction mask for the parent joint:
            if(jointCorrectionMasks[parentIdx].empty())
            {
                jointCorrectionMasks[parentIdx] = jointMask;
            }

            // Compute the intervals in which the joint is in contact with the floor:
            auto contactIntervals = ComputeContactIntervals(c.contactMask, jointMask, contactThreshold);
            FilterContactIntervals(contactIntervals, jointMask, true);
            for(const auto &interval : contactIntervals)
            {
                for (int fixFrame = interval.first; fixFrame < interval.second; ++fixFrame)
                {
                    // All we're going to do here is stick the joint to the floor -
                    // we're going to allow it to slide from side to side.

                    // Find a target position that lies on the floor by iteratively
                    // projecting the joint to the floor (pure laziness really, this could
                    // be done analytically):
                    Math::Vector parentPos = Animation::JointLocalToGlobal(joint_parents_vec, parentIdx, ikFixedPoses[fixFrame]).GetTranslation();
                    Math::Vector target = Animation::JointLocalToGlobal(joint_parents_vec, jointIdx, ikFixedPoses[fixFrame]).GetTranslation();
                    float jointLength = (target - parentPos).GetLength3();
                    for(int32_t i = 0; i < 10; ++i)
                    {
                        target.SetY(c.minHeight);
                        auto dir = (target - parentPos).GetNormalized3();
                        target = parentPos + dir * jointLength;
                    }

                    IK::OneBoneIk(
                        ikFixedPoses[fixFrame],
                        Math::Transform::Identity,
                        jointIdx,
                        1.0f,
                        target,
                        joint_parents_vec
                    );
                    jointCorrectionMasks[parentIdx][fixFrame] = 1.0f;
                }
            }

        }

        // Fixing the contacts with IK will introduce popping into the animation,
        // so let's apply the interpolator to all the joints we modified so as to
        // line the trajectory up properly again:
        for( auto &kv : jointCorrectionMasks)
        {
            for( size_t i = 0; i < N; ++i )
            {
                margins[i] = kv.second[i] ? 0.0f : -1.0f;
            }
            TrajectoryCorrector corrector(margins, pos_weight * 10, vel_weight, acc_weight);
            CorrectRotationsForBone(
                poses,
                ikFixedPoses,
                kv.second,
                corrector,
                kv.first,
                false
            );
        }

        if (has_double_ankle_joints)
        {
            // Maps to save target positions BEFORE 2-bone IK modifies them
            std::map<uint32_t, std::map<uint32_t, Math::Vector>> savedFirstAnkleTargets;  // [firstAnkleIdx][frame] -> position
            std::map<uint32_t, std::map<uint32_t, Math::Vector>> savedToeTargets;         // [firstAnkleIdx][frame] -> position
            std::map<uint32_t, uint32_t> contactToToeIdx;  // firstAnkleIdx -> toeIdx

            // Find toe joints for each leg
            for (const auto& tc : contacts)
            {
                if (tc.contactType == Animation::kOneBone)
                {
                    // The parent of the toe is the 1st ankle
                    int parentIdx = joint_parents_vec[tc.jointIndex];
                    if (parentIdx >= 0)
                    {
                        contactToToeIdx[parentIdx] = tc.jointIndex;
                    }
                }
            }

            // For each 2-bone contact, correct the parent (2nd ankle) joint
            for (auto& c : contacts)
            {
                if (c.contactType != Animation::kTwoBone)
                    continue;

                const auto firstAnkleIdx = c.jointIndex;
                const auto secondAnkleIdx = joint_parents_vec[firstAnkleIdx];
                const auto kneeIdx = joint_parents_vec[secondAnkleIdx];
                const auto hipIdx = joint_parents_vec[kneeIdx];

                if (hipIdx < 0) continue;  // safety check

                // Get saved contact frames for this ankle
                auto it = twoBoneContactFrames.find(firstAnkleIdx);
                if (it == twoBoneContactFrames.end())
                    continue;
                const auto& contactFrames = it->second;

                // Add correction mask for knee and hip
                auto jointMask = fullBodyMask;
                addEndEffectorMask(firstAnkleIdx, secondAnkleIdx, jointMask);

                if (jointCorrectionMasks[kneeIdx].empty())
                    jointCorrectionMasks[kneeIdx] = jointMask;
                if (jointCorrectionMasks[hipIdx].empty())
                    jointCorrectionMasks[hipIdx] = jointMask;

                for (uint32_t fixFrame = 0; fixFrame < fullBodyMask.size(); ++fixFrame)
                {
                    // Only correct frames where the 1st ankle was corrected
                    if (!contactFrames[fixFrame])
                        continue;

                    // *** SAVE TARGET POSITIONS BEFORE 2-BONE IK ***
                    savedFirstAnkleTargets[firstAnkleIdx][fixFrame] = Animation::JointLocalToGlobal(
                        joint_parents_vec, firstAnkleIdx, ikFixedPoses[fixFrame]).GetTranslation();

                    if (contactToToeIdx.count(firstAnkleIdx))
                    {
                        savedToeTargets[firstAnkleIdx][fixFrame] = Animation::JointLocalToGlobal(
                            joint_parents_vec, contactToToeIdx[firstAnkleIdx], ikFixedPoses[fixFrame]).GetTranslation();
                    }

                    // Get original global transforms (before any IK corrections)
                    auto originalFirstAnkleGlobal = Animation::JointLocalToGlobal(
                        joint_parents_vec, firstAnkleIdx, originalPoses[fixFrame]);
                    auto originalSecondAnkleGlobal = Animation::JointLocalToGlobal(
                        joint_parents_vec, secondAnkleIdx, originalPoses[fixFrame]);

                    // Compute delta from 1st ankle to 2nd ankle in original animation
                    auto deltaFirstToSecond = originalFirstAnkleGlobal.GetDeltaToOther(originalSecondAnkleGlobal);

                    // Get corrected 1st ankle global transform
                    auto correctedFirstAnkleGlobal = Animation::JointLocalToGlobal(
                        joint_parents_vec, firstAnkleIdx, ikFixedPoses[fixFrame]);

                    // Apply the original delta to the corrected 1st ankle to get target for 2nd ankle
                    auto target = (deltaFirstToSecond * correctedFirstAnkleGlobal).GetTranslation();

                    // print current and target second ankle positions
                    auto currPos = Animation::JointLocalToGlobal(
                        joint_parents_vec, secondAnkleIdx, ikFixedPoses[fixFrame]).GetTranslation();

                    // Apply 2-bone IK: Hip -> Knee -> 2nd Ankle
                    IK::TwoBoneIk(
                        ikFixedPoses[fixFrame],
                        Math::Transform::Identity,
                        secondAnkleIdx,
                        1.0f,
                        target,
                        joint_parents_vec,
                        c.hintOffset
                    );

                    // auto correctedPos = Animation::JointLocalToGlobal(
                    //     joint_parents_vec, secondAnkleIdx, ikFixedPoses[fixFrame]).GetTranslation();
                    // std::cout << "Frame " << fixFrame << ": target second ankle=(" << target.GetX() << ", " << target.GetY() << ", " << target.GetZ() << "), corrected second ankle position=(" << correctedPos.GetX() << ", " << correctedPos.GetY() << ", " << correctedPos.GetZ() << ")" << std::endl;

                    jointCorrectionMasks[kneeIdx][fixFrame] = 1.0f;
                    jointCorrectionMasks[hipIdx][fixFrame] = 1.0f;
                }
            }

            // Smooth the corrected joints
            for (auto& kv : jointCorrectionMasks)
            {
                for (size_t i = 0; i < N; ++i)
                    margins[i] = kv.second[i] ? 0.0f : -1.0f;

                TrajectoryCorrector corrector(margins, pos_weight * 10, vel_weight, acc_weight);
                CorrectRotationsForBone(poses, ikFixedPoses, kv.second, corrector, kv.first, false);
            }

            // *** PHASE 2: 1-bone IKs to restore 1st ankle and toe ***
            jointCorrectionMasks.clear();

            for (auto& c : contacts)
            {
                if (c.contactType != Animation::kTwoBone)
                    continue;

                const auto firstAnkleIdx = c.jointIndex;
                const auto secondAnkleIdx = joint_parents_vec[firstAnkleIdx];

                auto it = twoBoneContactFrames.find(firstAnkleIdx);
                if (it == twoBoneContactFrames.end())
                    continue;

                // Setup correction masks
                auto jointMask = fullBodyMask;
                addEndEffectorMask(firstAnkleIdx, secondAnkleIdx, jointMask);

                if (jointCorrectionMasks[secondAnkleIdx].empty())
                    jointCorrectionMasks[secondAnkleIdx] = jointMask;
                if (jointCorrectionMasks[firstAnkleIdx].empty())
                    jointCorrectionMasks[firstAnkleIdx] = jointMask;

                for (uint32_t fixFrame = 0; fixFrame < fullBodyMask.size(); ++fixFrame)
                {
                    if (!it->second[fixFrame])
                        continue;

                    // 1-bone IK: Rotate 2nd ankle so 1st ankle reaches saved target
                    IK::OneBoneIk(
                        ikFixedPoses[fixFrame],
                        Math::Transform::Identity,
                        firstAnkleIdx,
                        1.0f,
                        savedFirstAnkleTargets[firstAnkleIdx][fixFrame],
                        joint_parents_vec
                    );
                    jointCorrectionMasks[secondAnkleIdx][fixFrame] = 1.0f;

                    // auto target = savedFirstAnkleTargets[firstAnkleIdx][fixFrame];
                    // auto corrected = Animation::JointLocalToGlobal(
                    //     joint_parents_vec, firstAnkleIdx, ikFixedPoses[fixFrame]).GetTranslation();
                    // std::cout << "Frame " << fixFrame << ": target first ankle=(" << target.GetX() << ", " << target.GetY() << ", " << target.GetZ() << "), corrected first ankle=(" << corrected.GetX() << ", " << corrected.GetY() << ", " << corrected.GetZ() << ")" << std::endl;

                    // 1-bone IK: Rotate 1st ankle so toe reaches saved target
                    if (contactToToeIdx.count(firstAnkleIdx) && savedToeTargets[firstAnkleIdx].count(fixFrame))
                    {
                        IK::OneBoneIk(
                            ikFixedPoses[fixFrame],
                            Math::Transform::Identity,
                            contactToToeIdx[firstAnkleIdx],
                            1.0f,
                            savedToeTargets[firstAnkleIdx][fixFrame],
                            joint_parents_vec
                        );
                        jointCorrectionMasks[firstAnkleIdx][fixFrame] = 1.0f;
                    }

                    // target = savedToeTargets[firstAnkleIdx][fixFrame];
                    // corrected = Animation::JointLocalToGlobal(
                    //     joint_parents_vec, contactToToeIdx[firstAnkleIdx], ikFixedPoses[fixFrame]).GetTranslation();
                    // std::cout << "Frame " << fixFrame << ": target toe=(" << target.GetX() << ", " << target.GetY() << ", " << target.GetZ() << "), corrected toe=(" << corrected.GetX() << ", " << corrected.GetY() << ", " << corrected.GetZ() << ")" << std::endl;
                }
            }

            // Smooth 2nd ankle and 1st ankle
            for (auto& kv : jointCorrectionMasks)
            {
                for (size_t i = 0; i < N; ++i)
                    margins[i] = kv.second[i] ? 0.0f : -1.0f;

                TrajectoryCorrector corrector(margins, pos_weight * 10, vel_weight, acc_weight);
                CorrectRotationsForBone(poses, ikFixedPoses, kv.second, corrector, kv.first, false);
            }
        }
    }

}


Math::Transform Animation::JointLocalToGlobal(
    const std::vector<int>& joint_parents_vec,
    int32_t index,
    const Pose& localPose,
    const Math::Transform& rootTx)
{
    Math::Transform worldTx = Math::Transform::Identity;
    while (index > -1)
    {
        worldTx = worldTx * localPose[index];
        index = joint_parents_vec[index];
    }

    return worldTx * rootTx;
}

void Animation::CorrectMotion(
    std::vector<Pose>& poses,
    const std::vector<Pose>& targetPoses,
    const std::vector<float>& fullBodyMask,
    const std::vector<float>& rootMask,
    const std::vector<ContactInfo>& contacts,
    const std::vector<ContactInfo>& endEffectorPins,
    const std::vector<int>& joint_parents_vec,
    const std::vector<Math::Transform>& defaultPose,
    float contactThreshold,
    float root_margin,
    bool has_double_ankle_joints
)
{

    // Calculate some weights so we can preserve velocities more strongly on frames where
    // the root velocity is low
    const uint32_t N = poses.size();
    Eigen::VectorXd velocity_weights(N);
    for (uint32_t frame = 1; frame < N; ++frame)
    {
        // work out xz velocity for this frame:
        float xdiff = poses[frame][0].GetTranslation()[0] - poses[frame - 1][0].GetTranslation()[0];
        float zdiff = poses[frame][0].GetTranslation()[2] - poses[frame - 1][0].GetTranslation()[2];

        // find velocity magnitude, divided by a typical walking speed:
        float v_mag = sqrtf(xdiff*xdiff + zdiff*zdiff) / 0.05f;

        // weight lower velocities higher so that the corrector doesn't make the character drift around
        // when it's supposed to stand still:
        v_mag = std::max(v_mag, 1.0f/1000.0f);
        velocity_weights(frame) = 1.0f / v_mag;
    }
    velocity_weights[0] = velocity_weights[1];

    // Correct root y coordinates.
    // This will warp the root y coordinates in "poses" so they match the root y coordinates
    // in "targetPoses", on frames where the root y coordinates are constrained, ie the frames
    // where fullBodyMask = 1.
    // In addition to this, it preserves the root y coordinates in "pose" on frames where foot
    // contacts are active, to avoid mushiness when characters are jumping.
    CorrectHipsY(
        poses,
        targetPoses,
        fullBodyMask,
        contacts,
        contactThreshold
    );

    // Correct root xz coordinates:
    // This will warp the root xz coordinates in "poses" so they match the xz coordinates
    // in "targetPoses" on frames where fullBodyMask = 1, and warp them so they're within
    // "root_margin" units of targetPoses on frames where rootMask = 1.
    CorrectHipsXZ(
        poses,
        targetPoses,
        fullBodyMask,
        rootMask,
        endEffectorPins,
        velocity_weights,
        root_margin
    );

    // Correct joint rotations by warping the rotations so they match targetPoses on frames
    // where fullBodyMask = 1:
    CorrectJointRotations(
        poses,
        targetPoses,
        fullBodyMask,
        velocity_weights
    );

    // Apply IK for end effector pins
    DoEffectorIK(
        poses,
        targetPoses,
        fullBodyMask,
        endEffectorPins,
        joint_parents_vec,
        defaultPose
    );

    // Apply IK to stabilize limbs on contacts
    DoContactIK(
        poses,
        fullBodyMask,
        contacts,
        endEffectorPins,
        joint_parents_vec,
        defaultPose,
        contactThreshold,
        has_double_ankle_joints
    );
    // std::cout << "Running post processing." << std::endl;
}
