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

#pragma once

#include "nvvk/appbase_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "shaders/host_device.h"

// #VKRay
#include "nvh/gltfscene.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"

struct ReSTIRControl
{
  int   spp{1};         // Nb samples at each iteration
  int   ray_depth{1};
  float light_intensity{20.0f};        // Darkness is stronger for more hits
  float light_color_X{1.0f};
  float light_color_Y{1.0f};
  float light_color_Z{1.0f};
  int   randomize_color{0};
  int   light_type{2};  // 0 is area, 1 is point light, 2 is mesh light
  int   two_sided_lights{0};   // 0 is single-sided, 1 is double_sided
  int   frame{0};                // Current frame
  int   mode{5};    // Direct Lighting Mode
  int   GI_mode{3};  // Direct Lighting Mode
  int   M{16};
  int   num_neighbors{5};
  int   neighbor_radius{30};
};


//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class HelloVulkan : public nvvk::AppBaseVk
{
public:
  void setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily) override;
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void loadSceneGLTF(const std::string& filename);
  void updateDescriptorSet();
  void createUniformBuffer();
  void createObjDescriptionBuffer();
  void createTextureImages(const VkCommandBuffer& cmdBuf, const std::vector<std::string>& textures);
  void createTextureImages(const VkCommandBuffer& cmdBuf, tinygltf::Model& gltfModel);
  void updateUniformBuffer(const VkCommandBuffer& cmdBuf);
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();
  void rasterize(const VkCommandBuffer& cmdBuff);

  // Information pushed at each draw call
  PushConstantRaster m_pcRaster{{1},                // Identity matrix
                                0,                  // instance Id
                                0};

  nvh::GltfScene m_gltfScene;
  nvvk::Buffer   m_vertexBuffer;
  nvvk::Buffer   m_normalBuffer;
  nvvk::Buffer   m_uvBuffer;
  nvvk::Buffer   m_indexBuffer;
  nvvk::Buffer   m_materialBuffer;
  nvvk::Buffer   m_primInfo;
  nvvk::Buffer   m_lightToMeshIDBuffer;
  nvvk::Buffer   m_sceneDesc;

  int num_meshes = 0;
  int num_lights = 0;
  int num_textures = 0;


  // Graphic pipeline
  VkPipelineLayout            m_pipelineLayout;
  VkPipeline                  m_graphicsPipeline;
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  VkDescriptorPool            m_descPool;
  VkDescriptorSetLayout       m_descSetLayout;
  VkDescriptorSet             m_descSet;

  nvvk::Buffer m_bGlobals;  // Device-Host of the camera matrices

  VkSamplerCreateInfo gltfSamplerToVulkan(tinygltf::Sampler& tsampler);

  std::vector<nvvk::Texture> m_textures;  // vector of all textures of the scene
  std::vector<std::pair<nvvk::Image, VkImageCreateInfo>> m_images;
  std::vector<size_t>                                    m_defaultTextures;  // for cleanup


  nvvk::ResourceAllocatorDma m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil            m_debug;  // Utility to name objects


  // #Post - Draw the rendered image on a quad using a tonemapper
  void createOffscreenRender();
  void createPostPipeline();
  void createPostDescriptor();
  void updatePostDescriptorSet();
  void drawPost(VkCommandBuffer cmdBuf);

  nvvk::DescriptorSetBindings m_postDescSetLayoutBind;
  VkDescriptorPool            m_postDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout       m_postDescSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet             m_postDescSet{VK_NULL_HANDLE};
  VkPipeline                  m_postPipeline{VK_NULL_HANDLE};
  VkPipelineLayout            m_postPipelineLayout{VK_NULL_HANDLE};
  VkRenderPass                m_offscreenRenderPass{VK_NULL_HANDLE};
  VkFramebuffer               m_offscreenFramebuffer{VK_NULL_HANDLE};
  nvvk::Texture               m_offscreenColor;
  nvvk::Texture               m_offscreenDepth;
  VkFormat                    m_offscreenColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat                    m_offscreenDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};
  nvvk::Texture               m_gBuffer;
  nvvk::Texture               m_aoBuffer;

  // #Tuto_rayquery
  void initRayTracing();
  auto primitiveToVkGeometry(const nvh::GltfPrimMesh& prim);
  void createBottomLevelAS();
  void createTopLevelAS();


  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::RaytracingBuilderKHR                      m_rtBuilder;


  // #Tuto_animation
  void createCompDescriptors();
  void updateCompDescriptors();
  void createCompPipelines();
  void runCompute(VkCommandBuffer cmdBuf, ReSTIRControl& restirControl);

  nvvk::DescriptorSetBindings m_compDescSetLayoutBind;
  VkDescriptorPool            m_compDescPool;
  VkDescriptorSetLayout       m_compDescSetLayout;
  VkDescriptorSet             m_compDescSet;
  VkPipeline                  m_compPipeline;
  VkPipelineLayout            m_compPipelineLayout;

   // #Tuto_animation
  void createSpatialReuseDescriptors();
  void updateSpatialReuseDescriptors();
  void createSpatialReusePipelines();
  void runSpatialReuse(VkCommandBuffer cmdBuf, ReSTIRControl& restirControl);

  nvvk::DescriptorSetBindings m_spatialReuseDescSetLayoutBind;
  VkDescriptorPool            m_spatialReuseDescPool;
  VkDescriptorSetLayout       m_spatialReuseDescSetLayout;
  VkDescriptorSet             m_spatialReuseDescSet;
  VkPipeline                  m_spatialReusePipeline;
  VkPipelineLayout            m_spatialReusePipelineLayout;

  // #Tuto_jitter_cam
  void updateFrame();
  void resetFrame();
  int  m_frame{0};
  int  m_time{0};
  int  cur_rendering_mode{1};

  void createReservoirBuffers();

  nvvk::Buffer m_restirDesc;
  nvvk::Buffer m_previousReservoirBuffer;
  nvvk::Buffer m_temporalToSpatialReservoirBuffer;
  nvvk::Buffer m_bReSTIRGlobals;
  nvmath::mat4f cur_viewProj = nvmath::mat4f(1);

  std::vector<VkAccelerationStructureInstanceKHR> m_tlas;

  std::vector<mat4> m_modelmatrices;

  void animateLights();
  
  void updatePrimitiveBuffer();

  std::vector<PrimMeshInfo> m_primLookup;
  std::vector<LightPointer> m_light_IDs;
};
