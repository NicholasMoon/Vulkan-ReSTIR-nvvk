/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


//-
// This utility compresses a normal(x,y,z) to a uint and decompresses it

#include "host_device.h"

#define INV_PI 0.31831f


#define M_MAX 32

layout(buffer_reference, scalar) readonly buffer Vertices
{
  vec3 v[];
};
layout(buffer_reference, scalar) readonly buffer Indices
{
  ivec3 i[];
};
layout(buffer_reference, scalar) readonly buffer Normals
{
  vec3 n[];
};
layout(buffer_reference, scalar) readonly buffer TexCoords
{
  vec2 t[];
};
layout(buffer_reference, scalar) readonly buffer Materials
{
  GltfShadeMaterial m[];
};
layout(buffer_reference, scalar) readonly buffer LightIDs
{
  LightPointer l[];
};

layout(buffer_reference, scalar) readonly buffer PreviousReservoirs
{
  Reservoir r_prev[];
};
layout(buffer_reference, scalar) readonly buffer TemporalToSpatialReservoirs
{
  Reservoir r_temp_to_spat[];
};

layout(set = 1, binding = eGlobals) uniform _GlobalUniforms
{
  GlobalUniforms uni;
};
layout(set = 1, binding = eSceneDesc) readonly buffer SceneDesc_
{
  SceneDesc sceneDesc;
};
layout(set = 1, binding = eTextures) uniform sampler2D texturesMap[];  // all textures
layout(set = 1, binding = eReSTIRGlobals) uniform _ReSTIRUniforms
{
  ReSTIRUniforms uniforms_restir;
};

struct SceneBufferPointers
{
  Materials materials;
  Vertices  vertices;
  Indices   indices;
  Normals   normals;
  TexCoords texCoords;
  LightIDs  light_IDs;
};


struct CoordinateSystem
{
  vec3 normal;
  vec3 tangent;
  vec3 bitangent;
};

struct Ray
{
  vec3 origin;
  vec3 direction;
};

struct Intersection
{
  vec3  p;
  vec3  n;
  vec3  n_planar;
  vec2  uv;
  int   mat_ID;
  int   mesh_ID;
  int   tri_ID;
  float t;
};

struct Tri
{
  vec3 p[3];
  vec3 n[3];
  vec2 uv[3];
  vec3 planar_normal;
};

float random1D(float x)
{
  return fract(sin(x * 14.11f) * 34.13f);
}

Tri getTriangle(PrimMeshInfo pInfo, SceneBufferPointers scene_data)
{

  // get offsets for this mesh in the index and vertex arrays
  uint indexOffset  = (pInfo.indexOffset / 3);
  uint vertexOffset = pInfo.vertexOffset;

  // get the first tri of the mesh (assume square plane area light support for now only)
  ivec3 triangleIndex = scene_data.indices.i[indexOffset];
  triangleIndex += ivec3(vertexOffset);

  // get three vertices of first tri
  const vec3 p0 = scene_data.vertices.v[triangleIndex.x];
  const vec3 p1 = scene_data.vertices.v[triangleIndex.y];
  const vec3 p2 = scene_data.vertices.v[triangleIndex.z];

  const vec3 n0 = scene_data.normals.n[triangleIndex.x];
  const vec3 n1 = scene_data.normals.n[triangleIndex.y];
  const vec3 n2 = scene_data.normals.n[triangleIndex.z];

  const vec2 uv0 = scene_data.texCoords.t[triangleIndex.x];
  const vec2 uv1 = scene_data.texCoords.t[triangleIndex.y];
  const vec2 uv2 = scene_data.texCoords.t[triangleIndex.z];

  vec3 planar_normal = normalize(cross(p1 - p0, p2 - p0));

  Tri tri = {{p0, p1, p2}, {n0, n1, n2}, {uv0, uv1, uv2}, planar_normal};

  return tri;
}

Tri getTriangleReal(PrimMeshInfo pInfo, int tri_ID, SceneBufferPointers scene_data)
{

  // get offsets for this mesh in the index and vertex arrays
  uint indexOffset  = (pInfo.indexOffset / 3) + tri_ID;
  uint vertexOffset = pInfo.vertexOffset;

  // get the first tri of the mesh (assume square plane area light support for now only)
  ivec3 triangleIndex = scene_data.indices.i[indexOffset];
  triangleIndex += ivec3(vertexOffset);

  // get three vertices of first tri
  const vec3 p0 = scene_data.vertices.v[triangleIndex.x];
  const vec3 p1 = scene_data.vertices.v[triangleIndex.y];
  const vec3 p2 = scene_data.vertices.v[triangleIndex.z];

  const vec3 n0 = scene_data.normals.n[triangleIndex.x];
  const vec3 n1 = scene_data.normals.n[triangleIndex.y];
  const vec3 n2 = scene_data.normals.n[triangleIndex.z];

  const vec2 uv0 = scene_data.texCoords.t[triangleIndex.x];
  const vec2 uv1 = scene_data.texCoords.t[triangleIndex.y];
  const vec2 uv2 = scene_data.texCoords.t[triangleIndex.z];

  vec3 planar_normal = normalize(cross(p1 - p0, p2 - p0));

  Tri tri = {{p0, p1, p2}, {n0, n1, n2}, {uv0, uv1, uv2}, planar_normal};

  return tri;
}

void getTriangleIntersection(PrimMeshInfo pInfo, SceneBufferPointers scene_data, inout Intersection isect, vec3 isect_barys, mat4x3 object_to_world, mat4x3 world_to_object)
{
  // get offsets for this mesh in the index and vertex arrays
  uint indexOffset  = (pInfo.indexOffset / 3) + isect.tri_ID;
  uint vertexOffset = pInfo.vertexOffset;

  // get the first tri of the mesh (assume square plane area light support for now only)
  ivec3 triangleIndex = scene_data.indices.i[indexOffset];
  triangleIndex += ivec3(vertexOffset);

  // get three vertices of first tri
  const vec3 p0 = scene_data.vertices.v[triangleIndex.x];
  const vec3 p1 = scene_data.vertices.v[triangleIndex.y];
  const vec3 p2 = scene_data.vertices.v[triangleIndex.z];
  isect.p       = p0 * isect_barys.x + p1 * isect_barys.y + p2 * isect_barys.z;
  isect.p       = vec3(object_to_world * vec4(isect.p, 1.0));

  const vec3 n0 = scene_data.normals.n[triangleIndex.x];
  const vec3 n1 = scene_data.normals.n[triangleIndex.y];
  const vec3 n2 = scene_data.normals.n[triangleIndex.z];
  isect.n       = normalize(n0 * isect_barys.x + n1 * isect_barys.y + n2 * isect_barys.z);
  isect.n       = normalize(vec3(isect.n * world_to_object));
  isect.n_planar = normalize(cross(p1 - p0, p2 - p0));

  const vec2 uv0 = scene_data.texCoords.t[triangleIndex.x];
  const vec2 uv1 = scene_data.texCoords.t[triangleIndex.y];
  const vec2 uv2 = scene_data.texCoords.t[triangleIndex.z];
  isect.uv       = uv0 * isect_barys.x + uv1 * isect_barys.y + uv2 * isect_barys.z;
}

vec3 calcTriBaryFromPoint(vec3 p, vec3 p0, vec3 p1, vec3 p2)
{
  vec3 v0 = p1 - p0, v1 = p2 - p0, v2 = p - p0;
  float  d00   = dot(v0, v0);
  float  d01   = dot(v0, v1);
  float  d11   = dot(v1, v1);
  float  d20   = dot(v2, v0);
  float  d21   = dot(v2, v1);
  float  denom = d00 * d11 - d01 * d01;
  vec3   barys = vec3(0,0,0);
  barys.y            = (d11 * d20 - d01 * d21) / denom;
  barys.z            = (d00 * d21 - d01 * d20) / denom;
  barys.x            = 1.0f - barys.y - barys.z;
  return barys;
}



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


//-------------------------------------------------------------------------------------------------
// Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
//
vec3 OffsetRay(in vec3 p, in vec3 n)
{
  const float intScale   = 256.0f;
  const float floatScale = 1.0f / 65536.0f;
  const float origin     = 1.0f / 32.0f;

  ivec3 of_i = ivec3(intScale * n.x, intScale * n.y, intScale * n.z);

  vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x,  //
              abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y,  //
              abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}


//////////////////////////// AO //////////////////////////////////////
#define EPS 0.05
const float M_PI = 3.141592653589;

void ComputeDefaultBasis(const vec3 normal, out vec3 x, out vec3 y)
{
  // ZAP's default coordinate system for compatibility
  vec3        z  = normal;
  const float yz = -z.y * z.z;
  y = normalize(((abs(z.z) > 0.99999f) ? vec3(-z.x * z.y, 1.0f - z.y * z.y, yz) : vec3(-z.x * z.z, yz, 1.0f - z.z * z.z)));

  x = cross(y, z);
}

//-------------------------------------------------------------------------------------------------
// Random
//-------------------------------------------------------------------------------------------------


// Generate a random unsigned int from two unsigned int values, using 16 pairs
// of rounds of the Tiny Encryption Algorithm. See Zafar, Olano, and Curtis,
// "GPU Random Numbers via the Tiny Encryption Algorithm"
uint tea(uint val0, uint val1)
{
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;

  for(uint n = 0; n < 16; n++)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

uvec2 pcg2d(uvec2 v)
{
  v = v * 1664525u + 1013904223u;

  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;

  v = v ^ (v >> 16u);

  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;

  v = v ^ (v >> 16u);

  return v;
}

// Generate a random unsigned int in [0, 2^24) given the previous RNG state
// using the Numerical Recipes linear congruential generator
uint lcg(inout uint prev)
{
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate a random float in [0, 1) given the previous RNG state
float rnd(inout uint seed)
{
  return (float(lcg(seed)) / float(0x01000000));
}


//-------------------------------------------------------------------------------------------------
// Sampling
//-------------------------------------------------------------------------------------------------

// Randomly sampling around +Z
vec3 samplingHemisphere(inout uint seed, in vec3 x, in vec3 y, in vec3 z)
{
#define M_PI 3.141592

  float r1 = rnd(seed);
  float r2 = rnd(seed);
  float sq = sqrt(1.0 - r2);

  vec3 direction = vec3(cos(2 * M_PI * r1) * sq, sin(2 * M_PI * r1) * sq, sqrt(r2));
  direction      = direction.x * x + direction.y * y + direction.z * z;

  return direction;
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

int uniformSample1DIndex(inout uint rng_seed, int max_index)
{
  return min(int(floor(rnd(rng_seed) * float(max_index))), max_index - 1);
}


bool updateReservoir(inout Reservoir res, float mth_sampled_light_ID, float w, float u01, vec3 point_on_light)
{
  res.wsum += w;
  res.M += 1;
  if(u01 < w / res.wsum)
  {
    res.sampled_light_ID = mth_sampled_light_ID;
    res.point_on_light   = point_on_light;
    return true;
  }
  return false;
}


vec3 pointSampleSquarePlane(vec3 p0, vec3 p1, vec3 p2, out float scale_x, out float scale_z)
{
  vec2 center_point = vec2(0.0f);
  if(abs(p1.x - p0.x) > 0.1)
  {
    scale_x        = abs(p1.x - p0.x);
    center_point.x = (p1.x + p0.x) / 2.0f;
  }
  else if(abs(p2.x - p0.x) > 0.01)
  {
    scale_x        = abs(p2.x - p0.x);
    center_point.x = (p2.x + p0.x) / 2.0f;
  }
  else
  {
    scale_x        = abs(p2.x - p1.x);
    center_point.x = (p2.x + p1.x) / 2.0f;
  }
  if(abs(p1.z - p0.z) > 0.1)
  {
    scale_z        = abs(p1.z - p0.z);
    center_point.y = (p1.z + p0.z) / 2.0f;
  }
  else if(abs(p2.z - p0.z) > 0.1)
  {
    scale_z        = abs(p2.z - p0.z);
    center_point.y = (p1.z + p0.z) / 2.0f;
  }
  else
  {
    scale_z        = abs(p2.z - p1.z);
    center_point.y = (p1.z + p0.z) / 2.0f;
  }

  return vec3(center_point.x, p0.y, center_point.y);
}

vec3 uniformSampleSquarePlane(vec3 p0, vec3 p1, vec3 p2, vec2 u01, out float scale_x, out float scale_z)
{
  vec2 center_point = vec2(0.0f);
  if(abs(p1.x - p0.x) > 0.1)
  {
    scale_x        = abs(p1.x - p0.x);
    center_point.x = (p1.x + p0.x) / 2.0f;
  }
  else if(abs(p2.x - p0.x) > 0.01)
  {
    scale_x        = abs(p2.x - p0.x);
    center_point.x = (p2.x + p0.x) / 2.0f;
  }
  else
  {
    scale_x        = abs(p2.x - p1.x);
    center_point.x = (p2.x + p1.x) / 2.0f;
  }
  if(abs(p1.z - p0.z) > 0.1)
  {
    scale_z        = abs(p1.z - p0.z);
    center_point.y = (p1.z + p0.z) / 2.0f;
  }
  else if(abs(p2.z - p0.z) > 0.1)
  {
    scale_z        = abs(p2.z - p0.z);
    center_point.y = (p1.z + p0.z) / 2.0f;
  }
  else
  {
    scale_z        = abs(p2.z - p1.z);
    center_point.y = (p1.z + p0.z) / 2.0f;
  }

  return vec3(((u01.x * scale_x) - (scale_x / 2.0f)) + center_point.x, p0.y,
              ((u01.y * scale_z) - (scale_z / 2.0f)) + center_point.y);
}

vec2 uniformSampleTriangleBarycentric(vec2 u01)
{
  float su0 = sqrt(u01.x);
  return vec2(1.0f - su0, u01.y * su0);
}

vec3 uniformSampleTriangle(vec3 p0, vec3 p1, vec3 p2, vec2 u01)
{
  vec2 sampled_barycentrics = uniformSampleTriangleBarycentric(u01);
  return sampled_barycentrics.x * p0 + sampled_barycentrics.y * p1
         + (1.0f - sampled_barycentrics.x - sampled_barycentrics.y) * p2;
}

vec3 f_Diffuse_BRDF(vec3 R)
{
  return R * INV_PI;
}

float pdf_Diffuse_BRDF(float cos_theta)
{
  return cos_theta * INV_PI;
}

vec3 sample_f_Diffuse_BRDF(in CoordinateSystem coordinate_system, in vec3 R, in vec3 wo, out vec3 wi, out float pdf, out float cos_theta, inout uint rng_seed)
{
  wi        = normalize(samplingHemisphere(rng_seed, coordinate_system.tangent, coordinate_system.bitangent, coordinate_system.normal));
  cos_theta = abs(dot(coordinate_system.normal, wi));
  pdf       = pdf_Diffuse_BRDF(cos_theta);
  return f_Diffuse_BRDF(R);
}


/*
*
* Square Plane Area Light
*
*/

vec3 Le_SquarePlaneDiffuseAreaLight(PrimMeshInfo pInfo, SceneBufferPointers scene_data, in vec3 wi, in int is_two_sided, in int randomize_color, in vec3 light_color, in float light_intensity)
{
  vec3 random_emission = vec3(random1D(float(pInfo.indexOffset)), random1D(float(pInfo.vertexOffset)),
                              random1D(float(pInfo.indexOffset + pInfo.vertexOffset)));
  
  vec3 mat_emission = scene_data.materials.m[max(0, pInfo.materialIndex)].emissiveFactor.xyz;

  if(length(mat_emission) > 0.01f)
  {
    vec3 Le;
    if(randomize_color == 1)
    {
      Le = random_emission * light_color * light_intensity;
    }
    else
    {
      Le = mat_emission * light_color * light_intensity;
    } 
    if(is_two_sided == 1)
    {
      return Le;
    }
    else
    {
      if(wi.y > 0.0f)
      {
        return Le;
      }
      else
      {
        return vec3(0.0f);
      }
    }
  }
  else
  {
    return vec3(0.0f);
  }

}


float pdf_SquarePlaneDiffuseAreaLight(PrimMeshInfo pInfo, SceneBufferPointers scene_data, in vec3 o, in vec3 point_on_light)
{
  Tri   isect_tri = getTriangle(pInfo, scene_data);
  float scale_x   = 1.0f;
  float scale_z   = 1.0f;
  uniformSampleSquarePlane(isect_tri.p[0], isect_tri.p[1], isect_tri.p[2], vec2(0, 0), scale_x, scale_z);
  //Tri isect_tri = getTriangle(pInfo, scene_data);

  vec3 wi = normalize(point_on_light - o);

  float distance_to_light = length(point_on_light - o);

  float cos_theta = abs(dot(wi, vec3(0, -1, 0)));

  return (distance_to_light * distance_to_light) / (cos_theta * scale_x * scale_z);
}

vec3 sample_Le_SquarePlaneDiffuseAreaLight(PrimMeshInfo        pInfo,
                                           SceneBufferPointers scene_data,
                                           in int              is_two_sided,
                                           in int              randomize_color,
                                           in vec3             light_color,
                                           in float            light_intensity,
                                           in vec3             o,
                                           out vec3            wi,
                                           out float           pdf,
                                           out vec3            point_on_light,
                                           inout uint          rng_seed)
{
  Tri isect_tri = getTriangle(pInfo, scene_data);

  // generate a world space point on the light uniformly
  float scale_x  = 1.0f;
  float scale_z  = 1.0f;
  point_on_light = uniformSampleSquarePlane(isect_tri.p[0], isect_tri.p[1], isect_tri.p[2],
                                            vec2(rnd(rng_seed), rnd(rng_seed)), scale_x, scale_z);


  // set direction to be from ray origin to the sampled surface point on the sampled light
  wi = normalize(point_on_light - o);

  float distance_to_light = length(point_on_light - o);

  float cos_theta = abs(dot(wi, vec3(0, -1, 0)));

  pdf = (distance_to_light * distance_to_light) / (cos_theta * scale_x * scale_z);

  return Le_SquarePlaneDiffuseAreaLight(pInfo, scene_data, wi, is_two_sided, randomize_color, light_color, light_intensity);
}


/*
*
* Mesh Area Light
*
*/

vec3 Le_TriDiffuseAreaLight(PrimMeshInfo        pInfo,
                            in int              tri_ID,
                            SceneBufferPointers scene_data,
                            in vec3             point_on_light,
                            in vec3             wi,
                            in int              is_two_sided,
                            in int              randomize_color,
                            in vec3             light_color,
                            in float            light_intensity)
{

  Tri isect_tri = getTriangleReal(pInfo, tri_ID, scene_data);

  vec3 world_space_normal = normalize(vec3(vec4(isect_tri.planar_normal, 0) * pInfo.world_to_object));

  vec3 world_space_p0 = vec3(pInfo.object_to_world * vec4(isect_tri.p[0], 1));
  vec3 world_space_p1 = vec3(pInfo.object_to_world * vec4(isect_tri.p[1], 1));
  vec3 world_space_p2 = vec3(pInfo.object_to_world * vec4(isect_tri.p[2], 1));

  vec3 Nc = (world_space_normal + 1.0f) * 0.5f;

  vec3 random_emission =
      vec3(random1D(float(float(tri_ID) * Nc.x)), random1D(float(float(tri_ID) * Nc.y)),
           random1D(float(float(tri_ID) * Nc.z)));

  vec3 mat_emission = scene_data.materials.m[max(0, pInfo.materialIndex)].emissiveFactor.xyz;

  if(int(scene_data.materials.m[max(0, pInfo.materialIndex)].texture_IDs.z) > -1)
  {
    uint txtId = uint(scene_data.materials.m[max(0, pInfo.materialIndex)].texture_IDs.z);
    if(txtId < uni.num_textures)
    {
      vec3 barys = calcTriBaryFromPoint(point_on_light, world_space_p0, world_space_p1, world_space_p2);
      vec2 uvs   = isect_tri.uv[0] * barys.x + isect_tri.uv[1] * barys.y + isect_tri.uv[2] * barys.z;
      mat_emission *= textureLod(texturesMap[nonuniformEXT(txtId)], uvs, 0).xyz;
    }
  }

  if(length(mat_emission) > 0.01f)
  {
    vec3 Le;
    if(randomize_color == 1)
    {
      Le = random_emission * light_color * light_intensity;
    }
    else
    {
      Le = mat_emission * light_color * light_intensity;
    }
    if(is_two_sided == 1)
    {
      return Le;
    }
    else
    {
      if(dot(wi, world_space_normal) < 0.0f)
      {
        return Le;
      }
      else
      {
        return vec3(0.0f);
      }
    }
  }
  else
  {
    return vec3(0.0f);
  }
}

float pdf_TriDiffuseAreaLight(PrimMeshInfo pInfo, in int tri_ID, SceneBufferPointers scene_data, in vec3 o, in vec3 point_on_light)
{
  Tri isect_tri = getTriangleReal(pInfo, tri_ID, scene_data);

  vec3 world_space_normal = normalize(vec3(vec4(isect_tri.planar_normal, 0) * pInfo.world_to_object));

  vec3 world_space_p0 = vec3(pInfo.object_to_world * vec4(isect_tri.p[0], 1));
  vec3 world_space_p1 = vec3(pInfo.object_to_world * vec4(isect_tri.p[1], 1));
  vec3 world_space_p2 = vec3(pInfo.object_to_world * vec4(isect_tri.p[2], 1));
  

  vec3 wi = normalize(point_on_light - o);

  float distance_to_light = length(point_on_light - o);

  float cos_theta = abs(dot(wi, world_space_normal));

  return (distance_to_light * distance_to_light)
         / (cos_theta * 0.5 * length(cross(world_space_p1 - world_space_p0, world_space_p2 - world_space_p0)));
}

vec3 sample_Le_TriDiffuseAreaLight(PrimMeshInfo        pInfo,
                                           in int              tri_ID,
                                           SceneBufferPointers scene_data,
                                           in int              is_two_sided,
                                           in int              randomize_color,
                                           in vec3             light_color,
                                           in float            light_intensity,
                                           in vec3             o,
                                           out vec3            wi,
                                           out float           pdf,
                                           out vec3            point_on_light,
                                           inout uint          rng_seed)
{
  Tri isect_tri = getTriangleReal(pInfo, tri_ID, scene_data);

  // generate a world space point on the light uniformly
  point_on_light = uniformSampleTriangle(isect_tri.p[0], isect_tri.p[1], isect_tri.p[2],
                                            vec2(rnd(rng_seed), rnd(rng_seed)));

  point_on_light = vec3(pInfo.object_to_world * vec4(point_on_light, 1));

  vec3 world_space_normal = normalize(vec3(vec4(isect_tri.planar_normal, 0) * pInfo.world_to_object));

  vec3 world_space_p0 = vec3(pInfo.object_to_world * vec4(isect_tri.p[0], 1));
  vec3 world_space_p1 = vec3(pInfo.object_to_world * vec4(isect_tri.p[1], 1));
  vec3 world_space_p2 = vec3(pInfo.object_to_world * vec4(isect_tri.p[2], 1));


  // set direction to be from ray origin to the sampled surface point on the sampled light
  wi = normalize(point_on_light - o);

  float distance_to_light = length(point_on_light - o);

  float cos_theta = abs(dot(wi, world_space_normal));

  pdf = (distance_to_light * distance_to_light)
        / (cos_theta * 0.5 * length(cross(world_space_p1 - world_space_p0, world_space_p2 - world_space_p0)));

  return Le_TriDiffuseAreaLight(pInfo, tri_ID, scene_data, point_on_light, wi, is_two_sided, randomize_color, light_color, light_intensity);
}



/*
*
* Point Light
*
*/
vec3 Le_PointLight(PrimMeshInfo pInfo, SceneBufferPointers scene_data, in vec3 wi, in vec3 o, in float t, in vec3 light_color, in float light_intensity)
{
  /* vec3 random_lighting = vec3(random1D(float(pInfo.indexOffset)), random1D(float(pInfo.vertexOffset)),
                              random1D(float(pInfo.indexOffset + pInfo.vertexOffset)));*/
  vec3 Le = light_color * light_intensity
            * scene_data.materials.m[max(0, pInfo.materialIndex)].emissiveFactor.xyz;

  Tri isect_tri = getTriangle(pInfo, scene_data);

  // generate a world space point on the light uniformly
  float scale_x        = 1.0f;
  float scale_z        = 1.0f;
  vec3  point_on_light = pointSampleSquarePlane(isect_tri.p[0], isect_tri.p[1], isect_tri.p[2], scale_x, scale_z);

  vec3 point_isect = o + t * wi;

  if(point_isect.x > point_on_light.x - 0.01f && point_isect.x < point_on_light.x + 0.01f
     && point_isect.y > point_on_light.y - 0.01f && point_isect.y < point_on_light.y + 0.01f
     && point_isect.z > point_on_light.z - 0.01f && point_isect.z < point_on_light.z + 0.01f)
  {
    return Le / (t * t);
  }
  else
  {
    return vec3(0.0f);
  }
}

vec3 sample_Le_PointLight(PrimMeshInfo        pInfo,
                          SceneBufferPointers scene_data,
                          in vec3             o,
                          out vec3            wi,
                          out float           pdf,
                          out vec3            point_on_light,
                          in vec3             light_color,
                          in float            light_intensity)
{
  Tri isect_tri = getTriangle(pInfo, scene_data);

  // generate a world space point on the light uniformly
  float scale_x  = 1.0f;
  float scale_z  = 1.0f;
  point_on_light = pointSampleSquarePlane(isect_tri.p[0], isect_tri.p[1], isect_tri.p[2], scale_x, scale_z);

  // set direction to be from ray origin to the sampled surface point on the sampled light
  wi = normalize(point_on_light - o);

  float distance_to_light = length(point_on_light - o);

  pdf = 1.0f;

  return Le_PointLight(pInfo, scene_data, wi, o, distance_to_light, light_color, light_intensity);
}


