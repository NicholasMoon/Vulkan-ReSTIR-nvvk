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


#ifndef COMMON_HOST_DEVICE
#define COMMON_HOST_DEVICE

#ifdef __cplusplus
#include "nvmath/nvmath.h"
// GLSL Type
using vec2 = nvmath::vec2f;
using vec3 = nvmath::vec3f;
using vec4 = nvmath::vec4f;
using mat4 = nvmath::mat4f;
using uint = unsigned int;
#endif

// clang-format off
#ifdef __cplusplus // Descriptor binding helper for C++ and GLSL
 #define START_BINDING(a) enum a {
 #define END_BINDING() }
#else
 #define START_BINDING(a)  const uint
 #define END_BINDING() 
#endif

START_BINDING(SceneBindings)
  eGlobals  = 0,  // Global uniform containing camera matrices
  eSceneDesc = 1,  // Access to the scene buffers
  eTextures = 2,   // Access to textures
  eReSTIRGlobals = 3   // Global ReSTIR uniform containing ReSTIR information
END_BINDING();

START_BINDING(RtxBindings)
  eTlas     = 0,  // Top-level acceleration structure
  eOutImage = 1,   // Ray tracer output image
  ePrimLookup = 2
END_BINDING();
// clang-format on

// Scene buffer addresses
struct SceneDesc
{
  uint64_t vertexAddress;    // Address of the Vertex buffer
  uint64_t normalAddress;    // Address of the Normal buffer
  uint64_t uvAddress;        // Address of the texture coordinates buffer
  uint64_t indexAddress;     // Address of the triangle indices buffer
  uint64_t materialAddress;  // Address of the Materials buffer (GltfShadeMaterial)
  uint64_t primInfoAddress;  // Address of the mesh primitives buffer (PrimMeshInfo)
  uint64_t lightIDAddress;  // Address of the light to mesh ID buffer
};

// Structure used for retrieving the primitive information in the closest hit
struct PrimMeshInfo
{
  mat4 object_to_world;

  mat4 world_to_object;

  uint indexOffset;
  uint vertexOffset;
  uint triCount;
  int  materialIndex;

  vec4 dummy_data_0; // need dummy data to pad out struct

  vec4 dummy_data_1;

  vec4 dummy_data_2;

  vec4 dummy_data_3;
};

struct LightPointer
{
  int mesh_ID;	// pointer to primitive
  int tri_ID;	// pointer to tri in primitive (only support tri area lights)
};



// Uniform buffer set at each frame
struct GlobalUniforms
{
mat4 viewProj;  // Camera view * projection
mat4 viewInverse;  // Camera inverse view matrix
mat4 projInverse;  // Camera inverse projection matrix

int num_meshes; // num meshes in the scene
int num_lights;	// num lights in the scene
int num_textures; // num textures in the scenne
int wre; // dummy data

vec4 cam_world_pos;

};

// Push constant structure for the raster
struct PushConstantRaster
{
  mat4  modelMatrix;  // matrix of the instance
  uint  objIndex;
  int   materialId;
};

struct Vertex  // See ObjLoader, copy of VertexObj, could be compressed for device
{
  vec3 pos;
  vec3 nrm;
  vec3 color;
  vec2 texCoord;
};

struct GltfShadeMaterial
{
  vec4 pbrBaseColor; // xyz = color, w = color texture

  vec4 emissiveFactor; // xyz = emissive color, w = is_light

  vec4 pbrAttributes; // 0 = metallic, 1 = roughnesss, 2 = ior, 3 = material BSDF

  vec4 texture_IDs; // 0 = normal, 1 = metallic, 2 = emission, 3 = ?
  
};

// ReSTIR buffer addresses
struct ReSTIRDesc
{
  uint64_t previousReservoirAddress;    // Address of the previous reservior buffer (written to at end of spatial reuse)
  uint64_t temporalToSpatialReservoirAddress;  // Address of the reservoir buffer that is updated in temporal component and sent to spatial reuse
};

// Uniform buffer set at each frame
struct ReSTIRUniforms
{
  mat4 previous_viewProj;     // Camera view * projection
};

struct Reservoir
{
  float sampled_light_ID;  // light ID to sample
  float wsum;              // sum of weights
  float M;                 // number of light samples accounted for in this reservoir
  float W;                 // weight for the current sample
  vec3  point_on_light;	   // object space point on the light source
};

#endif
