/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "AnimProcessing/Utility.h"
#include "AnimProcessing/RGBMoCapAnatomical.h"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4623 4191 4686 4868 5219 4191 4355)
#endif
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#ifdef _WIN32
#pragma warning(pop)
#endif

namespace py = pybind11;

float strip_nan_inf(float x) noexcept
{
    if (std::isnan(x)) return 0;
    if (std::isinf(x)) return 0;
    return x;
}

py::dict correct_motion(
    py::array_t<float> &rootTranslations,
    py::array_t<float> &jointRotations,
    const py::array_t<float>& rootTranslationsTarget,
    const py::array_t<float>& jointRotationsTarget,
    const py::array_t<float>& fullPoseMask,
    const py::array_t<float>& leftHandMask,
    const py::array_t<float>& rightHandMask,
    const py::array_t<float>& leftFootMask,
    const py::array_t<float>& rightFootMask,
    const py::array_t<float>& rootMask,
    const py::array_t<float>& contacts,
    const py::list& joint_parents,
    const py::list& joint_ref_translations,
    const py::list& joint_ref_rotations,
    int left_hand_idx,
    int right_hand_idx,
    int left_foot_idx,
    int right_foot_idx,
    float contact_threshold,
    float root_margin,
    bool has_double_ankle_joints,
    const py::array_t<float>& contact_targets,
    const py::array_t<double>& anatomical_axes
)
{
    if(joint_parents.size() != joint_ref_translations.size())
    {
        throw std::runtime_error("correct_motion python bindings: joint_parents and joint_ref_translations must have the same size");
    }
    if(joint_parents.size() != joint_ref_rotations.size())
    {
        throw std::runtime_error("correct_motion python bindings: joint_parents and joint_ref_rotations must have the same size");
    }
    if(left_hand_idx < 0 || right_hand_idx < 0 || left_foot_idx < 0 || right_foot_idx < 0)
    {
        throw std::runtime_error("correct_motion python bindings: left_hand_idx, right_hand_idx, left_foot_idx, and right_foot_idx must be non-negative");
    }
    if(left_hand_idx >= joint_parents.size() || right_hand_idx >= joint_parents.size() || left_foot_idx >= joint_parents.size() || right_foot_idx >= joint_parents.size())
    {
        throw std::runtime_error("correct_motion python bindings: left_hand_idx, right_hand_idx, left_foot_idx, and right_foot_idx must be less than the number of joints");
    }

    std::vector<Math::Transform> defaultPose(joint_parents.size());
    for (size_t i = 0; i < joint_ref_translations.size(); ++i)
    {
        if (!py::isinstance<py::list>(joint_ref_translations[i]))
        {
            throw std::runtime_error("correct_motion python bindings: Expected joint_ref_translations to be a list of lists");
        }
        py::list inner_list = joint_ref_translations[i].cast<py::list>();
        if (inner_list.size() != 3) {
            throw std::runtime_error("correct_motion python bindings: Expected joint_ref_translations to be a list of lists, length 3");
        }

        if (
            !py::isinstance<py::float_>(inner_list[0]) ||
            !py::isinstance<py::float_>(inner_list[1]) ||
            !py::isinstance<py::float_>(inner_list[2])
        )
        {
            throw std::runtime_error("correct_motion python bindings: Expected joint_ref_translations to be a list of lists, length 3, float values");
        }


        if (!py::isinstance<py::list>(joint_ref_rotations[i]))
        {
            throw std::runtime_error("correct_motion python bindings: Expected joint_ref_rotations to be a list of lists");
        }
        py::list inner_list_rot = joint_ref_rotations[i].cast<py::list>();
        if (inner_list_rot.size() != 4) {
            throw std::runtime_error("correct_motion python bindings: Expected joint_ref_rotations to be a list of lists, length 4");
        }

        if (
            !py::isinstance<py::float_>(inner_list_rot[0]) ||
            !py::isinstance<py::float_>(inner_list_rot[1]) ||
            !py::isinstance<py::float_>(inner_list_rot[2]) ||
            !py::isinstance<py::float_>(inner_list_rot[3])
        )
        {
            throw std::runtime_error("correct_motion python bindings: Expected joint_ref_rotations to be a list of lists, length 4, float values");
        }

        defaultPose[i].SetTranslation(Math::Vector(
            inner_list[0].cast<float>(),
            inner_list[1].cast<float>(),
            inner_list[2].cast<float>()));
        defaultPose[i].SetRotation(Math::Quaternion(
            inner_list_rot[0].cast<float>(),
            inner_list_rot[1].cast<float>(),
            inner_list_rot[2].cast<float>(),
            inner_list_rot[3].cast<float>()
        ));
    }

    std::vector<int> joint_parents_vec(joint_parents.size());
    for (size_t i = 0; i < joint_parents.size(); ++i)
    {
        if (!py::isinstance<py::int_>(joint_parents[i]))
        {
            throw std::runtime_error("correct_motion python bindings: Expected joint_parents to be a list of ints");
        }
        joint_parents_vec[i] = joint_parents[i].cast<int>();
        if (joint_parents_vec[i] >= (int)joint_parents.size())
        {
            throw std::runtime_error("correct_motion python bindings: joint_parents must be a list of ints, and all values must be less than the number of joints");
        }
    }

    size_t num_joints = defaultPose.size();
    size_t gen_length = fullPoseMask.size();
    if (gen_length < 2 || contacts.size() != 4 * gen_length)
        throw std::runtime_error("Expected at least two frames and four contacts per frame");
    if (contact_targets.size() && contact_targets.size() != 6 * gen_length)
        throw std::runtime_error("Expected [T,2,3] explicit contact targets");
    const bool anatomical=anatomical_axes.size()!=0;
    std::array<RGBMoCap::V,2> axes;
    std::vector<float> originalRotations;
    std::vector<RGBMoCap::V> referenceOffsets;
    if(anatomical) {
        if(anatomical_axes.size()!=6 || contact_targets.size()!=6*gen_length || has_double_ankle_joints)
            throw std::runtime_error("Anatomical contacts require two axes, explicit targets and a single ankle chain");
        for(size_t j=0;j<num_joints;++j) {
            if((j==0 && joint_parents_vec[j]!=-1) || (j>0 && (joint_parents_vec[j]<0 || joint_parents_vec[j]>=int(j))))
                throw std::runtime_error("Anatomical parents must be topologically ordered with one root");
            auto t=defaultPose[j].GetTranslation();
            referenceOffsets.emplace_back(t.GetX(),t.GetY(),t.GetZ());
            if(!referenceOffsets.back().allFinite())throw std::runtime_error("Non-finite anatomical rest offset");
        }
        for(int foot=0;foot<2;++foot) {
            axes[foot]=RGBMoCap::V(anatomical_axes.at(3*foot),anatomical_axes.at(3*foot+1),anatomical_axes.at(3*foot+2));
            const int ankle=foot==0 ? left_foot_idx : right_foot_idx;
            const int knee=joint_parents_vec[ankle],hip=knee>0 ? joint_parents_vec[knee] : -1;
            if(!axes[foot].allFinite() || std::abs(axes[foot].norm()-1.)>1e-5 || hip<=0 || referenceOffsets[knee].norm()<1e-6 || referenceOffsets[ankle].norm()<1e-6)
                throw std::runtime_error("Invalid anatomical knee axis or leg chain");
        }
        for(size_t f=0;f<gen_length;++f) {
            if(fullPoseMask.at(f) || leftHandMask.at(f) || rightHandMask.at(f) || leftFootMask.at(f) || rightFootMask.at(f) || rootMask.at(f) || contacts.at(4*f+1) || contacts.at(4*f+3))
                throw std::runtime_error("Anatomical contacts cannot silently override generic keyframe or toe constraints");
        }
        originalRotations.assign(jointRotations.data(),jointRotations.data()+jointRotations.size());
        for(size_t k=0;k+3<originalRotations.size();k+=4) {
            double norm=0.;
            for(int component=0;component<4;++component) {
                const double value=originalRotations[k+component];
                if(!std::isfinite(value))throw std::runtime_error("Non-finite anatomical source rotation");
                norm+=value*value;
            }
            if(norm<1e-16)throw std::runtime_error("Zero anatomical source quaternion");
        }
        for(py::ssize_t k=0;k<rootTranslations.size();++k)if(!std::isfinite(rootTranslations.at(k)))throw std::runtime_error("Non-finite anatomical root");
        for(py::ssize_t k=0;k<contact_targets.size();++k)if(!std::isfinite(contact_targets.at(k)))throw std::runtime_error("Non-finite anatomical contact target");
        for(py::ssize_t k=0;k<contacts.size();++k) {
            const float value=contacts.at(k);
            if(!std::isfinite(value) || value<0.f || value>1.f)throw std::runtime_error("Invalid anatomical contact probability");
        }
    }

    if(
        leftHandMask.size() != (int)gen_length ||
        rightHandMask.size() != (int)gen_length ||
        leftFootMask.size() != (int)gen_length ||
        rightFootMask.size() != (int)gen_length ||
        rootMask.size() != (int)gen_length
    )
    {
        throw std::runtime_error("correct_motion python bindings: all masks must have the same size");
    }

    if(rootTranslations.size() != 3 * (int)gen_length)
    {
        throw std::runtime_error("correct_motion python bindings: rootTranslations has the wrong size");
    }
    if(jointRotations.size() != 4 * (int)num_joints * (int)gen_length)
    {
        throw std::runtime_error("correct_motion python bindings: jointRotations has the wrong size");
    }

    if(rootTranslationsTarget.size() != 3 * (int)gen_length)
    {
        throw std::runtime_error("correct_motion python bindings: rootTranslationsTarget has the wrong size");
    }
    if(jointRotationsTarget.size() != 4 * (int)num_joints * (int)gen_length)
    {
        throw std::runtime_error("correct_motion python bindings: jointRotationsTarget has the wrong size");
    }

    std::vector<Animation::ContactInfo> endEffectorPins(4);
    endEffectorPins[0].jointIndex = left_hand_idx;
    endEffectorPins[0].hintOffset = Math::Vector(0.0f, 0.0f, -0.1f);

    endEffectorPins[1].jointIndex = right_hand_idx;
    endEffectorPins[1].hintOffset = Math::Vector(0.0f, 0.0f, -0.1f);

    endEffectorPins[2].jointIndex = left_foot_idx;
    endEffectorPins[2].hintOffset = Math::Vector(0.0f, 0.0f, 0.1f);

    endEffectorPins[3].jointIndex = right_foot_idx;
    endEffectorPins[3].hintOffset = Math::Vector(0.0f, 0.0f, 0.1f);

    endEffectorPins[0].contactMask.reserve(gen_length);
    endEffectorPins[1].contactMask.reserve(gen_length);
    endEffectorPins[2].contactMask.reserve(gen_length);
    endEffectorPins[3].contactMask.reserve(gen_length);
    for(size_t i = 0; i < gen_length; ++i)
    {
        endEffectorPins[0].contactMask.push_back((1.0f - fullPoseMask.at(i)) * leftHandMask.at(i));
        endEffectorPins[1].contactMask.push_back((1.0f - fullPoseMask.at(i)) * rightHandMask.at(i));
        endEffectorPins[2].contactMask.push_back((1.0f - fullPoseMask.at(i)) * leftFootMask.at(i));
        endEffectorPins[3].contactMask.push_back((1.0f - fullPoseMask.at(i)) * rightFootMask.at(i));
    }

    std::vector<Animation::ContactInfo> contactInfo(2);

    auto footTranslation = Animation::JointLocalToGlobal(
        joint_parents_vec,
        right_foot_idx,
        defaultPose
    ).GetTranslation();

    contactInfo[0].jointIndex = right_foot_idx;
    contactInfo[0].hintOffset = Math::Vector(0.0f, 0.0f, 0.1f);
    contactInfo[0].minHeight = footTranslation.GetY();

    footTranslation = Animation::JointLocalToGlobal(
        joint_parents_vec,
        left_foot_idx,
        defaultPose
    ).GetTranslation();

    contactInfo[1].jointIndex = left_foot_idx;
    contactInfo[1].hintOffset = Math::Vector(0.0f, 0.0f, 0.1f);
    contactInfo[1].minHeight = footTranslation.GetY();
    // RGB MoCap: position goals prepared from sole patches and one global
    // contact schedule; preserve the original bend plane instead of a rig hint.
    if (contact_targets.size()) {
        contactInfo[0].hintOffset = Math::Vector::Zero;
        contactInfo[1].hintOffset = Math::Vector::Zero;
        for (size_t f = 0; f < gen_length; ++f) {
            contactInfo[1].targetPositions.emplace_back(contact_targets.at(6*f), contact_targets.at(6*f+1), contact_targets.at(6*f+2));
            contactInfo[0].targetPositions.emplace_back(contact_targets.at(6*f+3), contact_targets.at(6*f+4), contact_targets.at(6*f+5));
        }
    }

    auto& rContacts = contactInfo[0].contactMask;
    auto& lContacts = contactInfo[1].contactMask;

    rContacts.resize(fullPoseMask.size());
    lContacts.resize(fullPoseMask.size());
    for (int i = 0; i < fullPoseMask.size(); ++i)
    {
        // don't flag it as a contact if it's been masked:
        rContacts[i] = rightFootMask.at(i) ? 0 : contacts.at(4 * i + 2);
        lContacts[i] = leftFootMask.at(i) ? 0 : contacts.at(4 * i + 0);

        // Flag the heel as a contact if the toe is a contact:
        rContacts[i] = std::min((rightFootMask.at(i) ? 0 : contacts.at(4 * i + 3)) + rContacts[i], 1.0f);
        lContacts[i] = std::min((leftFootMask.at(i) ? 0 : contacts.at(4 * i + 1)) + lContacts[i], 1.0f);
    }

    int left_toe_idx = -1;
    int right_toe_idx = -1;
    for(int i = 0; i < num_joints; ++i)
    {
        if(joint_parents_vec[i] == left_foot_idx)
        {
            left_toe_idx = i;
        }
        if(joint_parents_vec[i] == right_foot_idx)
        {
            right_toe_idx = i;
        }
    }

    if(left_toe_idx != -1 && right_toe_idx != -1)
    {
        auto toeTranslation = Animation::JointLocalToGlobal(
            joint_parents_vec,
            right_toe_idx,
            defaultPose
        ).GetTranslation();

        contactInfo.resize(4);
        contactInfo[2].jointIndex = right_toe_idx;
        contactInfo[2].contactType = Animation::kOneBone;
        contactInfo[2].minHeight = toeTranslation.GetY();

        contactInfo[3].jointIndex = left_toe_idx;
        contactInfo[3].contactType = Animation::kOneBone;
        contactInfo[3].minHeight = Animation::JointLocalToGlobal(
            joint_parents_vec, left_toe_idx, defaultPose).GetTranslation().GetY();

        auto& rToeContacts = contactInfo[2].contactMask;
        auto& lToeContacts = contactInfo[3].contactMask;

        // fill up the ankle contacts:
        rToeContacts.resize(fullPoseMask.size());
        lToeContacts.resize(fullPoseMask.size());

        for (int i = 0; i < fullPoseMask.size(); ++i)
        {
            // don't flag it as a contact if it's been masked:
            rToeContacts[i] = rightFootMask.at(i) ? 0 : contacts.at(4 * i + 3);
            lToeContacts[i] = leftFootMask.at(i) ? 0 : contacts.at(4 * i + 1);
        }
    }


    auto setTransforms = [gen_length, num_joints](
        std::vector< std::vector<Math::Transform> > &poses,
        const py::array_t<float> &rootTranslations,
        const py::array_t<float> &jointRotations
    )
    {
        for (size_t f = 0; f < gen_length; ++f)
        {
            poses[f][0].SetTranslation({
                strip_nan_inf(rootTranslations.at(3*f+0)),
                strip_nan_inf(rootTranslations.at(3*f+1)),
                strip_nan_inf(rootTranslations.at(3*f+2))
            });
        }

        for (size_t f = 0; f < gen_length; ++f)
        {
            for (size_t j = 0; j < num_joints; ++j)
            {
                // x y z w order:
                Math::Quaternion q(
                    strip_nan_inf(jointRotations.at(4 * (num_joints * f + j) + 1)),
                    strip_nan_inf(jointRotations.at(4 * (num_joints * f + j) + 2)),
                    strip_nan_inf(jointRotations.at(4 * (num_joints * f + j) + 3)),
                    strip_nan_inf(jointRotations.at(4 * (num_joints * f + j) + 0))
                );
                q.Normalize();
                poses[f][j].SetRotation(q);
            }
        }
    };

    std::vector< std::vector<Math::Transform> > posesFixed(gen_length, defaultPose);
    setTransforms(posesFixed, rootTranslations, jointRotations);

    std::vector< std::vector<Math::Transform> > posesTarget(gen_length, defaultPose);
    setTransforms(posesTarget, rootTranslationsTarget, jointRotationsTarget);

    std::vector<float> fullPoseMask_vec;
    std::vector<float> rootMask_vec;
    for (size_t f = 0; f < gen_length; ++f)
    {
        fullPoseMask_vec.push_back(fullPoseMask.at(f));
        rootMask_vec.push_back(rootMask.at(f));
    }

    Animation::CorrectMotion(
        posesFixed,
        posesTarget,
        fullPoseMask_vec,
        rootMask_vec,
        anatomical ? std::vector<Animation::ContactInfo>{} : contactInfo,
        endEffectorPins,
        joint_parents_vec,
        defaultPose,
        contact_threshold,
        root_margin,
        has_double_ankle_joints
    );

    for (size_t f = 0; f < gen_length; ++f)
    {
        auto t = posesFixed[f][0].GetTranslation();
        rootTranslations.mutable_at(3*f+0) = t.GetX();
        rootTranslations.mutable_at(3*f+1) = t.GetY();
        rootTranslations.mutable_at(3*f+2) = t.GetZ();
    }

    for (size_t f = 0; f < gen_length; ++f)
    {
        for (size_t j = 0; j < num_joints; ++j)
        {
            auto q = posesFixed[f][j].GetRotation();
            // w x y z order
            jointRotations.mutable_at(4 * (num_joints * f + j) + 0) = ((float*)&q)[3];
            jointRotations.mutable_at(4 * (num_joints * f + j) + 1) = ((float*)&q)[0];
            jointRotations.mutable_at(4 * (num_joints * f + j) + 2) = ((float*)&q)[1];
            jointRotations.mutable_at(4 * (num_joints * f + j) + 3) = ((float*)&q)[2];
        }
    }

    py::dict metrics;
    if(anatomical) {
        // RGB MoCap: the native backend is the final anatomical solver. No
        // observed-plane IK followed by a second Python pose rewrite.
        std::copy(originalRotations.begin(),originalRotations.end(),jointRotations.mutable_data());
        const auto result=RGBMoCap::SolveLegs(jointRotations,rootTranslations,originalRotations,contact_targets,contacts,
            joint_parents_vec,referenceOffsets,axes,left_foot_idx,right_foot_idx,gen_length,contact_threshold);
        metrics["max_added_knee_off_axis_deg"]=0.;
        metrics["max_added_knee_rotation_deg"]=result.max_delta*180./3.14159265358979323846;
        metrics["max_corrected_knee_flex_deg"]=result.max_flex*180./3.14159265358979323846;
    }
    return metrics;
}

PYBIND11_MODULE(_motion_correction, m) {
    m.doc() = "Motion Correction Python bindings";
    m.def("correct_motion", &correct_motion,
        py::arg("rootTranslations"), py::arg("jointRotations"),
        py::arg("rootTranslationsTarget"), py::arg("jointRotationsTarget"),
        py::arg("fullPoseMask"), py::arg("leftHandMask"), py::arg("rightHandMask"),
        py::arg("leftFootMask"), py::arg("rightFootMask"), py::arg("rootMask"),
        py::arg("contacts"), py::arg("joint_parents"), py::arg("joint_ref_translations"),
        py::arg("joint_ref_rotations"), py::arg("left_hand_idx"), py::arg("right_hand_idx"),
        py::arg("left_foot_idx"), py::arg("right_foot_idx"), py::arg("contact_threshold"),
        py::arg("root_margin"), py::arg("has_double_ankle_joints"),
        py::arg("contact_targets") = py::array_t<float>(),
        py::arg("anatomical_axes") = py::array_t<double>());
    m.attr("rgb_mocap_anatomical_version")=1;
}
