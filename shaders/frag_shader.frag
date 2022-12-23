/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "host_device.h"


layout(push_constant) uniform _PushConstantRaster
{
  PushConstantRaster pcRaster;
};

// clang-format off
// Incoming 
layout(location = 1) in vec3 i_worldPos;
layout(location = 2) in vec3 i_worldNrm;
layout(location = 3) in vec3 i_viewDir;
layout(location = 4) in vec2 i_texCoord;
// Outgoing
layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 o_gbuffer;

layout(binding = 0) uniform _GlobalUniforms
{
  GlobalUniforms uni;
};

layout(buffer_reference, scalar) buffer  GltfMaterial { GltfShadeMaterial m[]; };
layout(set = 0, binding = eSceneDesc ) readonly buffer SceneDesc_ { SceneDesc sceneDesc; } ;
layout(set = 0, binding = eTextures) uniform sampler2D[] textureSamplers;

struct CoordinateSystem
{
  vec3 normal;
  vec3 tangent;
  vec3 bitangent;
};

#define C_Stack_Max 3.402823466e+38f
uint CompressUnitVec(vec3 nv)
{
  // map to octahedron and then flatten to 2D (see 'Octahedron Environment Maps' by Engelhardt & Dachsbacher)
  if((nv.x < C_Stack_Max) && !isinf(nv.x))
  {
    const float d = 32767.0f / (abs(nv.x) + abs(nv.y) + abs(nv.z));
    int         x = int(roundEven(nv.x * d));
    int         y = int(roundEven(nv.y * d));
    if(nv.z < 0.0f)
    {
      const int maskx = x >> 31;
      const int masky = y >> 31;
      const int tmp   = 32767 + maskx + masky;
      const int tmpx  = x;
      x               = (tmp - (y ^ masky)) ^ maskx;
      y               = (tmp - (tmpx ^ maskx)) ^ masky;
    }
    uint packed = (uint(y + 32767) << 16) | uint(x + 32767);
    if(packed == ~0u)
      return ~0x1u;
    return packed;
  }
  else
  {
    return ~0u;
  }
}

float ShortToFloatM11(const int v)  // linearly maps a short 32767-32768 to a float -1-+1 //!! opt.?
{
  return (v >= 0) ? (uintBitsToFloat(0x3F800000u | (uint(v) << 8)) - 1.0f) :
                    (uintBitsToFloat((0x80000000u | 0x3F800000u) | (uint(-v) << 8)) + 1.0f);
}
vec3 DecompressUnitVec(uint packed)
{
  if(packed != ~0u)  // sanity check, not needed as isvalid_unit_vec is called earlier
  {
    int       x     = int(packed & 0xFFFFu) - 32767;
    int       y     = int(packed >> 16) - 32767;
    const int maskx = x >> 31;
    const int masky = y >> 31;
    const int tmp0  = 32767 + maskx + masky;
    const int ymask = y ^ masky;
    const int tmp1  = tmp0 - (x ^ maskx);
    const int z     = tmp1 - ymask;
    float     zf;
    if(z < 0)
    {
      x  = (tmp0 - ymask) ^ maskx;
      y  = tmp1 ^ masky;
      zf = uintBitsToFloat((0x80000000u | 0x3F800000u) | (uint(-z) << 8)) + 1.0f;
    }
    else
    {
      zf = uintBitsToFloat(0x3F800000u | (uint(z) << 8)) - 1.0f;
    }
    return normalize(vec3(ShortToFloatM11(x), ShortToFloatM11(y), zf));
  }
  else
  {
    return vec3(C_Stack_Max);
  }
}

// Return the tangent and binormal from the incoming normal
CoordinateSystem createCoordinateSystem(in vec3 normal) {
    CoordinateSystem coordinate_system = { normal, vec3(0,0,0), vec3(0,0,0) };

    if (abs(normal.x) > abs(normal.y)) {
        coordinate_system.tangent = vec3(normal.z, 0, -normal.x) / sqrt(normal.x * normal.x + normal.z * normal.z);
    }
    else {
        coordinate_system.tangent = vec3(0, -normal.z, normal.y) / sqrt(normal.y * normal.y + normal.z * normal.z);
    }
    
    coordinate_system.bitangent = cross(coordinate_system.normal, coordinate_system.tangent);

    return coordinate_system;
}


void main()
{
  // Material of the object
  GltfMaterial      gltfMat = GltfMaterial(sceneDesc.materialAddress);
  GltfShadeMaterial mat     = gltfMat.m[pcRaster.materialId];

  vec3 N = normalize(i_worldNrm);

  if (mat.texture_IDs.x > -1.0f) {
      uint txtId = int(mat.texture_IDs.x);
      CoordinateSystem NTB = createCoordinateSystem(N);


      vec3 mappedNor = textureLod(textureSamplers[nonuniformEXT(txtId)], i_texCoord, 0).xyz;
      mappedNor = normalize((2.0f * mappedNor) - 1.0f);
      vec3 normalizedTan = normalize(NTB.tangent);
      vec3 normalizedBit = normalize(NTB.bitangent);
      mat3 surfaceToWorldN = mat3(normalizedTan, normalizedBit, N);
      mappedNor = surfaceToWorldN * mappedNor;
      N = normalize(mappedNor);

  }
  

  // Result
  if (mat.emissiveFactor.w == 1.0f) {
    //vec3 Nc = (N + 1.0f) * 0.5f;
    //vec3 random_emission = vec3(random1D(float(float(gl_PrimitiveID) * Nc.x)), random1D(float(float(gl_PrimitiveID) * Nc.y)), random1D(float(float(gl_PrimitiveID) * Nc.z)));
    vec3 emission = mat.emissiveFactor.xyz;
    if(int(mat.texture_IDs.z) > -1) {
        uint txtId = uint(mat.texture_IDs.z);
        if (txtId < uni.num_textures) {
            emission *= textureLod(textureSamplers[nonuniformEXT(txtId)], i_texCoord, 0).xyz;
        }
    }
	o_color        = vec4(emission, 1);
  }
  else {
    vec3 albedo = mat.pbrBaseColor.xyz;
	if(int(mat.pbrBaseColor.w) > -1) {
        uint txtId = uint(mat.pbrBaseColor.w);
        if (txtId < uni.num_textures) {
            albedo *= textureLod(textureSamplers[nonuniformEXT(txtId)], i_texCoord, 0).xyz;
        }
    }
    o_color = vec4(albedo, 0);
  }

  o_gbuffer.rgba = vec4(i_worldPos, uintBitsToFloat(CompressUnitVec(N)));
}
