/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 RGB MoCap contributors
 * SPDX-License-Identifier: Apache-2.0
 * RGB MoCap extension: source-relative anatomical leg contacts. Generic
 * upstream MotionCorrection remains available without anatomical_axes.
 */
#pragma once
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>

namespace RGBMoCap {
using V = Eigen::Vector3d;
using M = Eigen::Matrix3d;
inline V Unit(V value) {
    return value.norm()>1e-10 ? V(value.normalized()) : V(0,1,0);
}
inline M Align(V a,V b) {
    a=Unit(a);b=Unit(b);
    const V v=a.cross(b);const double d=std::clamp(a.dot(b),-1.,1.);
    if(d < -1.+1e-8) {
        Eigen::Index i; a.cwiseAbs().minCoeff(&i);
        const V axis=Unit(a.cross(V::Unit(i)));
        return 2.*axis*axis.transpose()-M::Identity();
    }
    M skew;skew << 0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;
    return M::Identity()+skew+skew*skew/std::max(1.+d,1e-10);
}
struct Metrics { double max_delta=0.,max_flex=0.; };

// q is scalar-first. All angular math is double; public motion stays FP32.
// Output changes only hip/knee/ankle. Ankle world direction is retained.
template<class Array>
Metrics SolveLegs(Array& q,const Array& roots,const std::vector<float>& source,
                 const Array& targets,const Array& contacts,
                 const std::vector<int>& parents,const std::vector<V>& offsets,
                 const std::array<V,2>& axes,int left_ankle,int right_ankle,
                 size_t frames,double threshold) {
    const size_t joints=parents.size();Metrics stats;
    std::vector<M> local(joints),world(joints);std::vector<V> pos(joints);
    for(size_t f=0;f<frames;++f) {
        for(size_t j=0;j<joints;++j) {
            const size_t k=4*(f*joints+j);
            Eigen::Quaterniond quat(source[k],source[k+1],source[k+2],source[k+3]);
            if(!quat.coeffs().allFinite() || quat.norm()<1e-8)
                throw std::runtime_error("Invalid anatomical source quaternion");
            local[j]=quat.normalized().toRotationMatrix();
            world[j]=parents[j]<0 ? local[j] : M(world[parents[j]]*local[j]);
            pos[j]=parents[j]<0 ? V(roots.at(3*f),roots.at(3*f+1),roots.at(3*f+2)) : V(pos[parents[j]]+world[parents[j]]*offsets[j]);
        }
        for(int foot=0;foot<2;++foot) {
            if(contacts.at(4*f+2*foot)<=threshold)continue;
            const int ankle=foot==0 ? left_ankle : right_ankle;
            const int knee=parents[ankle],hip=parents[knee],parent=parents[hip];
            const V upper=offsets[knee],lower=offsets[ankle],axis=axes[foot];
            const V v=local[knee]*lower;
            const double dot=v.dot(axis),a=(v-axis*dot).dot(upper),b=axis.cross(v).dot(upper);
            const V target(targets.at(6*f+3*foot),targets.at(6*f+3*foot+1),targets.at(6*f+3*foot+2));
            if(!target.allFinite())throw std::runtime_error("Non-finite anatomical contact target");
            const V delta=target-pos[hip];
            const double desired=.5*(delta.squaredNorm()-upper.squaredNorm()-lower.squaredNorm())-upper.dot(axis)*dot;
            const double flex=std::min(std::acos(std::clamp(desired/std::max(std::hypot(a,b),1e-12),-1.,1.)),160.*3.14159265358979323846/180.);
            const double increment=std::atan2(b,a)+flex;
            const M knee_local=Eigen::AngleAxisd(increment,axis).toRotationMatrix()*local[knee];
            const V end=upper+knee_local*lower;
            const M hip_world=Align(world[hip]*end,delta)*world[hip];
            const std::array<int,3> ids={hip,knee,ankle};
            const std::array<M,3> result={M(world[parent].transpose()*hip_world),knee_local,M((hip_world*knee_local).transpose()*world[ankle])};
            for(int index=0;index<3;++index) {
                Eigen::Quaterniond value(result[index]);value.normalize();
                if(!value.coeffs().allFinite())throw std::runtime_error("Non-finite anatomical IK output");
                const size_t k=4*(f*joints+ids[index]);
                q.mutable_at(k)=float(value.w());q.mutable_at(k+1)=float(value.x());
                q.mutable_at(k+2)=float(value.y());q.mutable_at(k+3)=float(value.z());
            }
            stats.max_delta=std::max(stats.max_delta,std::abs(increment));
            stats.max_flex=std::max(stats.max_flex,flex);
        }
    }
    return stats;
}
}
