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


#include <sstream>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "obj_loader.h"

#include "hello_vulkan.h"
#include "nvh/alignment.hpp"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvh/gltfscene.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/buffers_vk.hpp"

extern std::vector<std::string> defaultSearchPaths;


//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily)
{
  AppBaseVk::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(instance, device, physicalDevice);
  m_debug.setup(m_device);
  m_offscreenDepthFormat = nvvk::findDepthFormat(physicalDevice);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer(const VkCommandBuffer& cmdBuf)
{
  ReSTIRUniforms hostUBO_ReSTIR = {};
  hostUBO_ReSTIR.previous_viewProj     = cur_viewProj;

  // Prepare new UBO contents on host.
  const float    aspectRatio = m_size.width / static_cast<float>(m_size.height);
  GlobalUniforms hostUBO     = {};
  const auto&    view        = CameraManip.getMatrix();
  const auto&    proj        = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  // proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).

  hostUBO.viewProj = proj * view;
  cur_viewProj     = hostUBO.viewProj;

  //std::cout << previous_viewProj[0][0] << " " << hostUBO.viewProj[0][0] << " " << cur_viewProj[0][0] << std::endl;

  hostUBO.viewInverse = nvmath::invert(view);
  hostUBO.projInverse = nvmath::invert(proj);
  vec3 camera_eye, camera_ref, camera_up;
  CameraManip.getLookat(camera_eye, camera_ref, camera_up);
  hostUBO.cam_world_pos = vec4(camera_eye, 1);

  hostUBO.num_meshes = num_meshes;
  hostUBO.num_lights = num_lights;
  hostUBO.num_textures = num_textures;

  // UBO on the device, and what stages access it.
  VkBuffer deviceUBO      = m_bGlobals.buffer;
  auto     uboUsageStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier beforeBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  beforeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  beforeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  beforeBarrier.buffer        = deviceUBO;
  beforeBarrier.offset        = 0;
  beforeBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, uboUsageStages, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &beforeBarrier, 0, nullptr);


  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_bGlobals.buffer, 0, sizeof(GlobalUniforms), &hostUBO);

  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier afterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  afterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  afterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  afterBarrier.buffer        = deviceUBO;
  afterBarrier.offset        = 0;
  afterBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, uboUsageStages, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &afterBarrier, 0, nullptr);



  // Handling ReSTIR uniforms

  VkBuffer deviceUBO_ReSTIR = m_bReSTIRGlobals.buffer;
  auto     uboUsageStages_ReSTIR = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                               | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier beforeBarrier_ReSTIR{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  beforeBarrier_ReSTIR.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  beforeBarrier_ReSTIR.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  beforeBarrier_ReSTIR.buffer        = deviceUBO_ReSTIR;
  beforeBarrier_ReSTIR.offset        = 0;
  beforeBarrier_ReSTIR.size          = sizeof(hostUBO_ReSTIR);
  vkCmdPipelineBarrier(cmdBuf, uboUsageStages_ReSTIR, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &beforeBarrier_ReSTIR, 0, nullptr);

  // Schedule the host-to-device upload. (hostUBO_ReSTIR is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_bReSTIRGlobals.buffer, 0, sizeof(ReSTIRUniforms), &hostUBO_ReSTIR);

  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier afterBarrier_ReSTIR{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  afterBarrier_ReSTIR.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  afterBarrier_ReSTIR.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  afterBarrier_ReSTIR.buffer        = deviceUBO_ReSTIR;
  afterBarrier_ReSTIR.offset        = 0;
  afterBarrier_ReSTIR.size          = sizeof(hostUBO_ReSTIR);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, uboUsageStages_ReSTIR, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &afterBarrier_ReSTIR, 0, nullptr);

}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  auto nbTxt = static_cast<uint32_t>(m_textures.size());

  //std::cout << m_textures.size() << std::endl;

  // Camera matrices
  m_descSetLayoutBind.addBinding(SceneBindings::eGlobals, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
  // Textures
  m_descSetLayoutBind.addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nbTxt, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

    // Scene buffers
  m_descSetLayoutBind.addBinding(SceneBindings::eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

      // Scene buffers
  m_descSetLayoutBind.addBinding(SceneBindings::eReSTIRGlobals, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<VkWriteDescriptorSet> writes;

  // Camera matrices and scene description
  VkDescriptorBufferInfo dbiUnif{m_bGlobals.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eGlobals, &dbiUnif));

  /* VkDescriptorBufferInfo dbiSceneDesc{m_bObjDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eObjDescs, &dbiSceneDesc));*/

  // All texture samplers
  std::vector<VkDescriptorImageInfo> diit;
  for(auto& texture : m_textures)
  {
    diit.emplace_back(texture.descriptor);
  }
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, SceneBindings::eTextures, diit.data()));

  VkDescriptorBufferInfo sceneDesc{m_sceneDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, eSceneDesc, &sceneDesc));

  // Camera matrices and scene description
  VkDescriptorBufferInfo dbiUnif_ReSTIR{m_bReSTIRGlobals.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eReSTIRGlobals, &dbiUnif_ReSTIR));

  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline()
{
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster)};

  // Creating the Pipeline Layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_descSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_pipelineLayout);


  // Creating the Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_offscreenRenderPass);
  gpb.depthStencilState.depthTestEnable = true;
  gpb.addShader(nvh::loadFile("spv/vert_shader.vert.spv", true, paths, true), VK_SHADER_STAGE_VERTEX_BIT);
  gpb.addShader(nvh::loadFile("spv/frag_shader.frag.spv", true, paths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  gpb.addBindingDescriptions({{0, sizeof(nvmath::vec3f)}, {1, sizeof(nvmath::vec3f)}, {2, sizeof(nvmath::vec2f)}});
  gpb.addAttributeDescriptions({
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},  // Position
      {1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0},  // Normal
      {2, 2, VK_FORMAT_R32G32_SFLOAT, 0},     // Texcoord0
  });

  VkPipelineColorBlendAttachmentState res{};
  res.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  res.colorBlendOp = VK_BLEND_OP_ADD;
  gpb.addBlendAttachmentState(res);

  m_graphicsPipeline = gpb.createPipeline();
  m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}

void HelloVulkan::loadSceneGLTF(const std::string& filename)
{
  using vkBU = VkBufferUsageFlagBits;
  tinygltf::Model    tmodel;
  tinygltf::TinyGLTF tcontext;
  std::string        warn, error;

  LOGI("Loading file: %s", filename.c_str());
  if(!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename))
  {
    assert(!"Error while loading scene");
  }
  LOGW(warn.c_str());
  LOGE(error.c_str());


  m_gltfScene.importMaterials(tmodel);
  m_gltfScene.importDrawableNodes(tmodel, nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0);

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = cmdBufGet.createCommandBuffer();

  m_vertexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_positions,
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_indexBuffer  = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_indices,
                                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                           | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_normalBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_normals,
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  m_uvBuffer     = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_texcoords0,
                                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  
  // Copying all materials, only the elements we need
  std::vector<GltfShadeMaterial> shadeMaterials;
  for(const auto& m : m_gltfScene.m_materials)
  {
    float is_light = 0.0f;
    if(m.emissiveFactor.x > 0.0f || m.emissiveFactor.y > 0.0f || m.emissiveFactor.z > 0.0f)
    {
      is_light = 1.0f;
    }
    float is_specular = 0.0f;
    if(m.metallicFactor == 1.0f && m.roughnessFactor == 0.0f)
    {
      is_specular = 1.0f;
    }
     
    shadeMaterials.emplace_back(GltfShadeMaterial {
      vec4(m.baseColorFactor.x, m.baseColorFactor.y, m.baseColorFactor.z, m.baseColorTexture),
          vec4(m.emissiveFactor.x, m.emissiveFactor.y, m.emissiveFactor.z, is_light), 
          vec4(m.metallicFactor, m.roughnessFactor, -1, is_specular),
          vec4(m.normalTexture, m.metallicRoughnessTexture, m.emissiveTexture, -1)});

    std::cout << m.baseColorTexture << " "  << m.normalTexture << " " << m.metallicRoughnessTexture << " "
              << m.emissiveTexture << std::endl;
  }
  m_materialBuffer = m_alloc.createBuffer(cmdBuf, shadeMaterials,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  num_lights = 0;
  num_meshes = 0;
  std::cout << " " << std::endl;
  // The following is used to find the primitive mesh information in the CHIT

  //std::cout << m_gltfScene.m_primMeshes.size() << " )" << std::endl;


  for(auto& node : m_gltfScene.m_nodes)
  {

    auto& primMesh       = m_gltfScene.m_primMeshes[node.primMesh];

    //std::cout << primMesh.name << std::endl;
    m_primLookup.push_back({node.worldMatrix, nvmath::invert(node.worldMatrix), primMesh.firstIndex, primMesh.vertexOffset,
                          primMesh.indexCount / 3, primMesh.materialIndex, vec4(-1, -1, -1, -1), vec4(-1, -1, -1, -1), vec4(-1, -1, -1, -1)});

     if(shadeMaterials[primMesh.materialIndex].emissiveFactor[3] == 1.0f)
    {
       for(int i = 0; i < primMesh.indexCount / 3; ++i)
       {
         //std::cout << num_meshes << " " << i << std::endl;
         m_light_IDs.push_back({num_meshes, i});
         ++num_lights;
       }
        
      //++num_lights;
    }
    
    m_modelmatrices.push_back(node.worldMatrix);
    ++num_meshes;
    //std::cout << num_meshes << " " << num_lights << std::endl;
  }
  m_primInfo = m_alloc.createBuffer(cmdBuf, m_primLookup, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  m_lightToMeshIDBuffer = m_alloc.createBuffer(cmdBuf, m_light_IDs, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);


  SceneDesc sceneDesc;
  sceneDesc.vertexAddress   = nvvk::getBufferDeviceAddress(m_device, m_vertexBuffer.buffer);
  sceneDesc.indexAddress    = nvvk::getBufferDeviceAddress(m_device, m_indexBuffer.buffer);
  sceneDesc.normalAddress   = nvvk::getBufferDeviceAddress(m_device, m_normalBuffer.buffer);
  sceneDesc.uvAddress       = nvvk::getBufferDeviceAddress(m_device, m_uvBuffer.buffer);
  sceneDesc.materialAddress = nvvk::getBufferDeviceAddress(m_device, m_materialBuffer.buffer);
  sceneDesc.primInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_primInfo.buffer);
  sceneDesc.lightIDAddress = nvvk::getBufferDeviceAddress(m_device, m_lightToMeshIDBuffer.buffer);
  m_sceneDesc               = m_alloc.createBuffer(cmdBuf, sizeof(SceneDesc), &sceneDesc,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  // Creates all textures found
  createTextureImages(cmdBuf, tmodel);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  num_textures = m_textures.size();
  std::cout << "gltf images " << tmodel.images.size() << "gltf textures " << tmodel.textures.size() << "num textures "
            << num_textures << std::endl;



  NAME_VK(m_vertexBuffer.buffer);
  NAME_VK(m_indexBuffer.buffer);
  NAME_VK(m_normalBuffer.buffer);
  NAME_VK(m_uvBuffer.buffer);
  NAME_VK(m_materialBuffer.buffer);
  NAME_VK(m_primInfo.buffer);
  NAME_VK(m_lightToMeshIDBuffer.buffer);
  NAME_VK(m_sceneDesc.buffer);
}


//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  m_bGlobals = m_alloc.createBuffer(sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_bGlobals.buffer, "Globals");

  m_bReSTIRGlobals = m_alloc.createBuffer(sizeof(ReSTIRUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_bReSTIRGlobals.buffer, "ReSTIRGlobals");
}

VkSamplerCreateInfo HelloVulkan::gltfSamplerToVulkan(tinygltf::Sampler& tsampler)
{
  VkSamplerCreateInfo vk_sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

  std::map<int, VkFilter> filters;
  filters[9728] = VK_FILTER_NEAREST;  // NEAREST
  filters[9729] = VK_FILTER_LINEAR;   // LINEAR
  filters[9984] = VK_FILTER_NEAREST;  // NEAREST_MIPMAP_NEAREST
  filters[9985] = VK_FILTER_LINEAR;   // LINEAR_MIPMAP_NEAREST
  filters[9986] = VK_FILTER_NEAREST;  // NEAREST_MIPMAP_LINEAR
  filters[9987] = VK_FILTER_LINEAR;   // LINEAR_MIPMAP_LINEAR

  std::map<int, VkSamplerMipmapMode> mipmap;
  mipmap[9728] = VK_SAMPLER_MIPMAP_MODE_NEAREST;  // NEAREST
  mipmap[9729] = VK_SAMPLER_MIPMAP_MODE_NEAREST;  // LINEAR
  mipmap[9984] = VK_SAMPLER_MIPMAP_MODE_NEAREST;  // NEAREST_MIPMAP_NEAREST
  mipmap[9985] = VK_SAMPLER_MIPMAP_MODE_NEAREST;  // LINEAR_MIPMAP_NEAREST
  mipmap[9986] = VK_SAMPLER_MIPMAP_MODE_LINEAR;   // NEAREST_MIPMAP_LINEAR
  mipmap[9987] = VK_SAMPLER_MIPMAP_MODE_LINEAR;   // LINEAR_MIPMAP_LINEAR

  std::map<int, VkSamplerAddressMode> addressMode;
  addressMode[33071] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  addressMode[33648] = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  addressMode[10497] = VK_SAMPLER_ADDRESS_MODE_REPEAT;

  vk_sampler.magFilter  = filters[tsampler.magFilter];
  vk_sampler.minFilter  = filters[tsampler.minFilter];
  vk_sampler.mipmapMode = mipmap[tsampler.minFilter];

  vk_sampler.addressModeU = addressMode[tsampler.wrapS];
  vk_sampler.addressModeV = addressMode[tsampler.wrapT];

  // Always allow LOD
  vk_sampler.maxLod = FLT_MAX;
  return vk_sampler;
}

void HelloVulkan::createTextureImages(const VkCommandBuffer& cmdBuf, tinygltf::Model& gltfModel)
{

  VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;


    // Make dummy image(1,1), needed as we cannot have an empty array
  auto addDefaultImage = [this, cmdBuf]() {
    std::array<uint8_t, 4> white           = {255, 255, 255, 255};
    VkImageCreateInfo      imageCreateInfo = nvvk::makeImage2DCreateInfo(VkExtent2D{1, 1});
    nvvk::Image            image           = m_alloc.createImage(cmdBuf, 4, white.data(), imageCreateInfo);
    m_images.emplace_back(image, imageCreateInfo);
    m_debug.setObjectName(m_images.back().first.image, "dummy");
  };

    // Make dummy texture/image(1,1), needed as we cannot have an empty array
  auto addDefaultTexture = [this, cmdBuf]() {
    m_defaultTextures.push_back(m_textures.size());
    std::array<uint8_t, 4> white = {255, 255, 255, 255};
    VkSamplerCreateInfo    sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_textures.emplace_back(m_alloc.createTexture(cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(VkExtent2D{1, 1}), sampler));
    m_debug.setObjectName(m_textures.back().image, "dummy");
  };

  if(gltfModel.images.empty())
  {
    addDefaultTexture();
    return;
  }

  m_images.reserve(gltfModel.images.size());
  for(size_t i = 0; i < gltfModel.images.size(); i++)
  {
    size_t sourceImage = i;

    auto& gltfimage = gltfModel.images[sourceImage];
    if(gltfimage.width == -1 || gltfimage.height == -1 || gltfimage.image.empty())
    {
      // Image not present or incorrectly loaded (image.empty)
      addDefaultImage();
      continue;
    }

    void*        buffer     = &gltfimage.image[0];
    VkDeviceSize bufferSize = gltfimage.image.size();
    auto         imgSize    = VkExtent2D{(uint32_t)gltfimage.width, (uint32_t)gltfimage.height};

    // Creating an image, the sampler and generating mipmaps
    VkImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);
    nvvk::Image       image           = m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    // nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
    m_images.emplace_back(image, imageCreateInfo);

    NAME_IDX_VK(m_images[i].first.image, i);
  }



    // Creating the textures using the above images
  m_textures.reserve(gltfModel.textures.size());
  for(size_t i = 0; i < gltfModel.textures.size(); i++)
  {
    int sourceImage = gltfModel.textures[i].source;

    if(sourceImage >= gltfModel.images.size() || sourceImage < 0)
    {
      // Incorrect source image
      addDefaultTexture();
      continue;
    }

    // Sampler
    VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
    samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if(gltfModel.textures[i].sampler > -1)
    {
      // Retrieve the texture sampler
      auto gltfSampler  = gltfModel.samplers[gltfModel.textures[i].sampler];
      samplerCreateInfo = gltfSamplerToVulkan(gltfSampler);
    }
    std::pair<nvvk::Image, VkImageCreateInfo>& image  = m_images[sourceImage];
    VkImageViewCreateInfo                      ivInfo = nvvk::makeImageViewCreateInfo(image.first.image, image.second);
    m_textures.emplace_back(m_alloc.createTexture(image.first, ivInfo, samplerCreateInfo));

    NAME_IDX_VK(m_textures[i].image, i);
  }

}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

  m_alloc.destroy(m_bGlobals);
  m_alloc.destroy(m_bReSTIRGlobals);
  //m_alloc.destroy(m_bObjDesc);
  m_alloc.destroy(m_vertexBuffer);
  m_alloc.destroy(m_normalBuffer);
  m_alloc.destroy(m_uvBuffer);
  m_alloc.destroy(m_indexBuffer);
  m_alloc.destroy(m_materialBuffer);
  m_alloc.destroy(m_primInfo);
  m_alloc.destroy(m_lightToMeshIDBuffer);
  m_alloc.destroy(m_sceneDesc);

  for(auto& i : m_images)
  {
    m_alloc.destroy(i.first);
    i = {};
  }
  m_images.clear();

  for(size_t i = 0; i < m_defaultTextures.size(); i++)
  {
    size_t last_index = m_defaultTextures[m_defaultTextures.size() - 1 - i];
    m_alloc.destroy(m_textures[last_index]);
    m_textures.erase(m_textures.begin() + last_index);
  }
  m_defaultTextures.clear();

  for(auto& t : m_textures)
  {
    vkDestroyImageView(m_device, t.descriptor.imageView, nullptr);
    t = {};
  }
  m_textures.clear();

  //for(auto& t : m_textures)
  //{
  //  m_alloc.destroy(t);
  //}


  //#Post
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_gBuffer);
  m_alloc.destroy(m_aoBuffer);
  m_alloc.destroy(m_offscreenDepth);
  vkDestroyPipeline(m_device, m_postPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);
  vkDestroyRenderPass(m_device, m_offscreenRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);

  // Compute
  m_alloc.destroy(m_restirDesc);
  m_alloc.destroy(m_previousReservoirBuffer);
  m_alloc.destroy(m_temporalToSpatialReservoirBuffer);
  vkDestroyPipeline(m_device, m_compPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_compPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_compDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_compDescSetLayout, nullptr);
  

  // #VKRay
  m_rtBuilder.destroy();
  m_alloc.deinit();
}

void HelloVulkan::animateLights()
{
  int i = 0;
  for(auto& node : m_gltfScene.m_nodes)
  {
    auto& primitive = m_gltfScene.m_primMeshes[node.primMesh];


    mat4 translate_matrix(1.0f);
    mat4 rotate_matrix(1.0f);
    if(m_gltfScene.m_materials[primitive.materialIndex].emissiveFactor.x > 0.01
       || m_gltfScene.m_materials[primitive.materialIndex].emissiveFactor.y > 0.01
       || m_gltfScene.m_materials[primitive.materialIndex].emissiveFactor.z > 0.01)
    {
      translate_matrix.as_translation(vec3(5.0 * sin(((float)m_time) / 50.0f) + 5.0f, 0, 0));
      rotate_matrix = rotate_matrix.rotate(((float)m_time) / 100.0f, vec3(0, 0, 1));
    }

    m_modelmatrices[i] = node.worldMatrix * rotate_matrix;

    VkAccelerationStructureInstanceKHR& tinst = m_tlas[i];
    tinst.transform                           = nvvk::toTransformMatrixKHR(m_modelmatrices[i]);

    ++i;
  }


  m_rtBuilder.buildTlas(m_tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
                        true);

  updatePrimitiveBuffer();
  return;
}

void HelloVulkan::updatePrimitiveBuffer()
{
  m_alloc.destroy(m_primInfo);
  m_alloc.destroy(m_sceneDesc);

  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = cmdBufGet.createCommandBuffer();

  int i = 0;
  for(auto& primMesh : m_primLookup)
  {

    //std::cout << primMesh.name << std::endl;
    m_primLookup[i].object_to_world = m_modelmatrices[i];
    m_primLookup[i].world_to_object = nvmath::invert(m_modelmatrices[i]),

    ++i;
  }
  
  m_primInfo = m_alloc.createBuffer(cmdBuf, m_primLookup, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  SceneDesc sceneDesc;
  sceneDesc.vertexAddress   = nvvk::getBufferDeviceAddress(m_device, m_vertexBuffer.buffer);
  sceneDesc.indexAddress    = nvvk::getBufferDeviceAddress(m_device, m_indexBuffer.buffer);
  sceneDesc.normalAddress   = nvvk::getBufferDeviceAddress(m_device, m_normalBuffer.buffer);
  sceneDesc.uvAddress       = nvvk::getBufferDeviceAddress(m_device, m_uvBuffer.buffer);
  sceneDesc.materialAddress = nvvk::getBufferDeviceAddress(m_device, m_materialBuffer.buffer);
  sceneDesc.primInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_primInfo.buffer);
  sceneDesc.lightIDAddress  = nvvk::getBufferDeviceAddress(m_device, m_lightToMeshIDBuffer.buffer);
  m_sceneDesc               = m_alloc.createBuffer(cmdBuf, sizeof(SceneDesc), &sceneDesc,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();


  updateDescriptorSet();
  updateCompDescriptors();
  updateSpatialReuseDescriptors();
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const VkCommandBuffer& cmdBuf)
{
  std::vector<VkDeviceSize> offsets = {0, 0, 0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  animateLights();

  // Dynamic Viewport
  setViewport(cmdBuf);

  // Drawing all triangles
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);

  std::vector<VkBuffer> vertexBuffers = {m_vertexBuffer.buffer, m_normalBuffer.buffer, m_uvBuffer.buffer};
  vkCmdBindVertexBuffers(cmdBuf, 0, static_cast<uint32_t>(vertexBuffers.size()), vertexBuffers.data(), offsets.data());
  vkCmdBindIndexBuffer(cmdBuf, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

  uint32_t idxNode = 0;
  int      i       = 0;
  for(auto& node : m_gltfScene.m_nodes)
  {
    auto& primitive = m_gltfScene.m_primMeshes[node.primMesh];

    m_pcRaster.modelMatrix = m_modelmatrices[i];
    m_pcRaster.objIndex    = node.primMesh;
    m_pcRaster.materialId  = primitive.materialIndex;
    
    vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstantRaster), &m_pcRaster);
    vkCmdDrawIndexed(cmdBuf, primitive.indexCount, 1, primitive.firstIndex, primitive.vertexOffset, 0);
    ++i;
  }
  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
  createOffscreenRender();
  updatePostDescriptorSet();
  createReservoirBuffers();
  updateCompDescriptors();
  updateSpatialReuseDescriptors();
  resetFrame();
}


//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void HelloVulkan::createOffscreenRender()
{
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_gBuffer);
  m_alloc.destroy(m_aoBuffer);
  m_alloc.destroy(m_offscreenDepth);

  VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
                                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                                           | VK_IMAGE_USAGE_STORAGE_BIT);


    nvvk::Image           image             = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo            = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    m_offscreenColor                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    m_debug.setObjectName(m_offscreenColor.image, "offscreen");
  }

  // The G-Buffer (rgba32f) - position(xyz) / normal(w-compressed)
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, VK_FORMAT_R32G32B32A32_SFLOAT,
                                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                                           | VK_IMAGE_USAGE_STORAGE_BIT);


    nvvk::Image           image      = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo     = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    m_gBuffer                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_gBuffer.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    m_debug.setObjectName(m_gBuffer.image, "G-Buffer");
  }

  // The ambient occlusion result (r32)
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, VK_FORMAT_R32G32B32A32_SFLOAT,
                                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                                           | VK_IMAGE_USAGE_STORAGE_BIT);


    nvvk::Image           image       = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo      = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    m_aoBuffer                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_aoBuffer.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    m_debug.setObjectName(m_aoBuffer.image, "gi Buffer");
  }


  // Creating the depth buffer
  auto depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  {
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);


    VkImageViewCreateInfo depthStencilView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthStencilView.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format           = m_offscreenDepthFormat;
    depthStencilView.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    depthStencilView.image            = image.image;

    m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_gBuffer.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_aoBuffer.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass =
        nvvk::createRenderPass(m_device, {m_offscreenColorFormat, m_offscreenColorFormat},  // RGBA + G-Buffer
                               m_offscreenDepthFormat, 1, true, true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  }

  // Creating the frame buffer for offscreen
  std::vector<VkImageView> attachments = {m_offscreenColor.descriptor.imageView, m_gBuffer.descriptor.imageView,
                                          m_offscreenDepth.descriptor.imageView};

  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);
  VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  info.renderPass      = m_offscreenRenderPass;
  info.attachmentCount = static_cast<int>(attachments.size());
  info.pAttachments    = attachments.data();
  info.width           = m_size.width;
  info.height          = m_size.height;
  info.layers          = 1;
  vkCreateFramebuffer(m_device, &info, nullptr, &m_offscreenFramebuffer);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void HelloVulkan::createPostPipeline()
{
  // Push constants in the fragment shader
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float)};

  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_postDescSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_postPipelineLayout);


  // Pipeline: completely generic, no vertices
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout, m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("spv/passthrough.vert.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_VERTEX_BIT);
  pipelineGenerator.addShader(nvh::loadFile("spv/post.frag.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;
  m_postPipeline                                = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createPostDescriptor()
{
  m_postDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_postDescSetLayoutBind.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
  m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet()
{
  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor));
  writes.emplace_back(m_postDescSetLayoutBind.makeWrite(m_postDescSet, 1, &m_aoBuffer.descriptor));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void HelloVulkan::drawPost(VkCommandBuffer cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Post");

  setViewport(cmdBuf);

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &aspectRatio);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, 1, &m_postDescSet, 0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Initialize Vulkan ray tracing
// #VKRay
void HelloVulkan::initRayTracing()
{
  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext = &m_rtProperties;
  vkGetPhysicalDeviceProperties2(m_physicalDevice, &prop2);

  m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
}

//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive in the Raytracing Geometry used for the BLAS
//
auto HelloVulkan::primitiveToVkGeometry(const nvh::GltfPrimMesh& prim)
{
  // BLAS builder requires raw device addresses.
  VkDeviceAddress vertexAddress = nvvk::getBufferDeviceAddress(m_device, m_vertexBuffer.buffer);
  VkDeviceAddress indexAddress  = nvvk::getBufferDeviceAddress(m_device, m_indexBuffer.buffer);

  uint32_t maxPrimitiveCount = prim.indexCount / 3;

  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
  triangles.vertexData.deviceAddress = vertexAddress;
  triangles.vertexStride             = sizeof(nvmath::vec3f);
  // Describe index data (32-bit unsigned int)
  triangles.indexType               = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress = indexAddress;
  // Indicate identity transform by setting transformData to null device pointer.
  //triangles.transformData = {};
  triangles.maxVertex = prim.vertexCount;

  // Identify the above data as containing opaque triangles.
  VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  asGeom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeom.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;  // For AnyHit
  asGeom.geometry.triangles = triangles;

  VkAccelerationStructureBuildRangeInfoKHR offset;
  offset.firstVertex     = prim.vertexOffset;
  offset.primitiveCount  = maxPrimitiveCount;
  offset.primitiveOffset = prim.firstIndex * sizeof(uint32_t);
  offset.transformOffset = 0;

  // Our blas is made from only one geometry, but could be made of many geometries
  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);

  return input;
}

//--------------------------------------------------------------------------------------------------
//
//
void HelloVulkan::createBottomLevelAS()
{
  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
  /* allBlas.reserve(m_objModel.size() + m_gltfScene.m_primMeshes.size());
  for(const auto& obj : m_objModel)
  {
    auto blas = objectToVkGeometryKHR(obj);

    // We could add more geometry in each BLAS, but we add only one for now
    allBlas.emplace_back(blas);
  }*/
  allBlas.reserve(m_gltfScene.m_primMeshes.size());
  for(auto& primMesh : m_gltfScene.m_primMeshes)
  {
    auto geo = primitiveToVkGeometry(primMesh);
    allBlas.push_back({geo});
  }
  m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
//
//
void HelloVulkan::createTopLevelAS()
{
  
  m_tlas.reserve(m_gltfScene.m_nodes.size());
  for(auto& node : m_gltfScene.m_nodes)
  {
    VkAccelerationStructureInstanceKHR rayInst{};
    rayInst.transform                      = nvvk::toTransformMatrixKHR(node.worldMatrix);
    rayInst.instanceCustomIndex            = node.primMesh;  // gl_InstanceCustomIndexEXT: to find which primitive
    rayInst.accelerationStructureReference = m_rtBuilder.getBlasDeviceAddress(node.primMesh);
    rayInst.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    rayInst.mask                           = 0xFF;
    rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
    m_tlas.emplace_back(rayInst);
  }
  m_rtBuilder.buildTlas(m_tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                    | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);
}


//////////////////////////////////////////////////////////////////////////
// Compute shader from ANIMATION tutorial
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Previous Reservoir Buffer creation
//
void HelloVulkan::createReservoirBuffers()
{
  m_alloc.destroy(m_previousReservoirBuffer);
  m_alloc.destroy(m_temporalToSpatialReservoirBuffer);
  m_alloc.destroy(m_restirDesc);

  using vkBU = VkBufferUsageFlagBits;

  // Create the buffers on Device
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = cmdBufGet.createCommandBuffer();

  std::vector<Reservoir> previous_reservoirs;
  for(int i = 0; i < m_size.width * m_size.height; ++i)
  {
    previous_reservoirs.emplace_back(Reservoir{0, 0, 0, 0, vec3(0, 0, 0) });
  }
  m_previousReservoirBuffer = m_alloc.createBuffer(cmdBuf, previous_reservoirs,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  
  std::vector<Reservoir> temporal_to_spatial_reservoirs;
  for(int i = 0; i < m_size.width * m_size.height; ++i)
  {
    temporal_to_spatial_reservoirs.emplace_back(Reservoir{ 0, 0, 0, 0, vec3(0, 0, 0) });
  }
  m_temporalToSpatialReservoirBuffer = m_alloc.createBuffer(cmdBuf, temporal_to_spatial_reservoirs,
                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  ReSTIRDesc restirDesc;
  restirDesc.previousReservoirAddress = nvvk::getBufferDeviceAddress(m_device, m_previousReservoirBuffer.buffer);
  restirDesc.temporalToSpatialReservoirAddress = nvvk::getBufferDeviceAddress(m_device, m_temporalToSpatialReservoirBuffer.buffer);
  m_restirDesc                    = m_alloc.createBuffer(cmdBuf, sizeof(ReSTIRDesc), &restirDesc,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  NAME_VK(m_previousReservoirBuffer.buffer);
  NAME_VK(m_temporalToSpatialReservoirBuffer.buffer);
  NAME_VK(m_restirDesc.buffer);
}

//--------------------------------------------------------------------------------------------------
// Compute shader descriptor
//
void HelloVulkan::createCompDescriptors()
{
  m_compDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // [in] G-Buffer (pos/nor)
  m_compDescSetLayoutBind.addBinding(1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // [in] TLAS
  m_compDescSetLayoutBind.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // [in] G-Buffer (color)
  m_compDescSetLayoutBind.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // Primitive info
  m_compDescSetLayoutBind.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // Previous Reservoirs

  m_compDescSetLayout = m_compDescSetLayoutBind.createLayout(m_device);
  m_compDescPool      = m_compDescSetLayoutBind.createPool(m_device, 1);
  m_compDescSet       = nvvk::allocateDescriptorSet(m_device, m_compDescPool, m_compDescSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the values to the descriptors
//
void HelloVulkan::updateCompDescriptors()
{
  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_compDescSetLayoutBind.makeWrite(m_compDescSet, 0, &m_gBuffer.descriptor));

  VkAccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
  VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  descASInfo.accelerationStructureCount = 1;
  descASInfo.pAccelerationStructures    = &tlas;
  VkDescriptorBufferInfo primitiveInfoDesc{m_primInfo.buffer, 0, VK_WHOLE_SIZE};

  writes.emplace_back(m_compDescSetLayoutBind.makeWrite(m_compDescSet, 1, &descASInfo));

  writes.emplace_back(m_compDescSetLayoutBind.makeWrite(m_compDescSet, 2, &m_offscreenColor.descriptor));

  writes.emplace_back(m_compDescSetLayoutBind.makeWrite(m_compDescSet, 3, &primitiveInfoDesc));


  VkDescriptorBufferInfo restirDesc{m_restirDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_compDescSetLayoutBind.makeWrite(m_compDescSet, 4, &restirDesc));

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Creating the pipeline: shader ...
//
void HelloVulkan::createCompPipelines()
{
  // pushing time
  VkPushConstantRange        push_constants = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReSTIRControl)};
  VkPipelineLayoutCreateInfo plCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  std::vector<VkDescriptorSetLayout> compDescSetLayouts = {m_compDescSetLayout, m_descSetLayout};
  plCreateInfo.setLayoutCount         = static_cast<uint32_t>(compDescSetLayouts.size());
  plCreateInfo.pSetLayouts            = compDescSetLayouts.data();
  plCreateInfo.pushConstantRangeCount = 1;
  plCreateInfo.pPushConstantRanges    = &push_constants;
  vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_compPipelineLayout);

  VkComputePipelineCreateInfo cpCreateInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpCreateInfo.layout = m_compPipelineLayout;
  cpCreateInfo.stage = nvvk::createShaderStageInfo(m_device, nvh::loadFile("spv/temporal_reuse_ReSTIR.comp.spv", true, defaultSearchPaths, true),
                                                   VK_SHADER_STAGE_COMPUTE_BIT);

  vkCreateComputePipelines(m_device, {}, 1, &cpCreateInfo, nullptr, &m_compPipeline);

  vkDestroyShaderModule(m_device, cpCreateInfo.stage.module, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Running compute shader
//
#define GROUP_SIZE 16  // Same group size as in compute shader
void HelloVulkan::runCompute(VkCommandBuffer cmdBuf, ReSTIRControl& restirControl)
{
  updateFrame();

  m_debug.beginLabel(cmdBuf, "Compute");

  // Adding a barrier to be sure the fragment has finished writing to the G-Buffer
  // before the compute shader is using the buffer
  VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageMemoryBarrier    imgMemBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  imgMemBarrier.srcAccessMask    = VK_ACCESS_SHADER_WRITE_BIT;
  imgMemBarrier.dstAccessMask    = VK_ACCESS_SHADER_READ_BIT;
  imgMemBarrier.image            = m_gBuffer.image;
  imgMemBarrier.oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
  imgMemBarrier.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
  imgMemBarrier.subresourceRange = range;

  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 0, nullptr, 1, &imgMemBarrier);

  VkBufferMemoryBarrier bufMemBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  bufMemBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  bufMemBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  bufMemBarrier.buffer        = m_previousReservoirBuffer.buffer;
  bufMemBarrier.offset        = 0;
  bufMemBarrier.size          = VK_WHOLE_SIZE;

  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 1, &bufMemBarrier, 0, nullptr);


  // Preparing for the compute shader
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_compPipeline);

  std::vector<VkDescriptorSet> descSets{m_compDescSet, m_descSet};
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_compPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);


  // Sending the push constant information
  restirControl.frame = m_frame;
  vkCmdPushConstants(cmdBuf, m_compPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReSTIRControl), &restirControl);

  // Dispatching the shader
  vkCmdDispatch(cmdBuf, (m_size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (m_size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);


  // Adding a barrier to be sure the compute shader has finished
  // writing to the temporal to spatial buffer before the post shader is using it
  
  //VkBufferMemoryBarrier bufMemBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  //bufMemBarrier.srcAccessMask     = VK_ACCESS_SHADER_WRITE_BIT;
  //bufMemBarrier.dstAccessMask    = VK_ACCESS_SHADER_READ_BIT;
  bufMemBarrier.buffer            = m_temporalToSpatialReservoirBuffer.buffer;
  //bufMemBarrier.offset            = 0;
  //bufMemBarrier.size              = VK_WHOLE_SIZE;

  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 1, &bufMemBarrier, 0, nullptr);


  m_debug.endLabel(cmdBuf);
}

void HelloVulkan::createSpatialReuseDescriptors()
{
  m_spatialReuseDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // [in] G-Buffer
  m_spatialReuseDescSetLayoutBind.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // [out] AO
  m_spatialReuseDescSetLayoutBind.addBinding(2, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // [in] TLAS
  m_spatialReuseDescSetLayoutBind.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // [in] G-Buffer
  m_spatialReuseDescSetLayoutBind.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // Primitive info
  m_spatialReuseDescSetLayoutBind.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // Reservoirs from temporal pass

  m_spatialReuseDescSetLayout = m_spatialReuseDescSetLayoutBind.createLayout(m_device);
  m_spatialReuseDescPool      = m_spatialReuseDescSetLayoutBind.createPool(m_device, 1);
  m_spatialReuseDescSet = nvvk::allocateDescriptorSet(m_device, m_spatialReuseDescPool, m_spatialReuseDescSetLayout);
}

void HelloVulkan::updateSpatialReuseDescriptors()
{
  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_spatialReuseDescSetLayoutBind.makeWrite(m_spatialReuseDescSet, 0, &m_gBuffer.descriptor));
  writes.emplace_back(m_spatialReuseDescSetLayoutBind.makeWrite(m_spatialReuseDescSet, 1, &m_aoBuffer.descriptor));

  VkAccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
  VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  descASInfo.accelerationStructureCount = 1;
  descASInfo.pAccelerationStructures    = &tlas;
  VkDescriptorBufferInfo primitiveInfoDesc{m_primInfo.buffer, 0, VK_WHOLE_SIZE};

  writes.emplace_back(m_spatialReuseDescSetLayoutBind.makeWrite(m_spatialReuseDescSet, 2, &descASInfo));

  writes.emplace_back(m_spatialReuseDescSetLayoutBind.makeWrite(m_spatialReuseDescSet, 3, &m_offscreenColor.descriptor));

  writes.emplace_back(m_spatialReuseDescSetLayoutBind.makeWrite(m_spatialReuseDescSet, 4, &primitiveInfoDesc));


  VkDescriptorBufferInfo restirDesc{m_restirDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_spatialReuseDescSetLayoutBind.makeWrite(m_spatialReuseDescSet, 5, &restirDesc));

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void HelloVulkan::createSpatialReusePipelines()
{
  // pushing time
  VkPushConstantRange                push_constants = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReSTIRControl)};
  VkPipelineLayoutCreateInfo         plCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  std::vector<VkDescriptorSetLayout> spatialReuseDescSetLayouts = {m_spatialReuseDescSetLayout, m_descSetLayout};
  plCreateInfo.setLayoutCount                           = static_cast<uint32_t>(spatialReuseDescSetLayouts.size());
  plCreateInfo.pSetLayouts                              = spatialReuseDescSetLayouts.data();
  plCreateInfo.pushConstantRangeCount                   = 1;
  plCreateInfo.pPushConstantRanges                      = &push_constants;
  vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_spatialReusePipelineLayout);

  VkComputePipelineCreateInfo cpCreateInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpCreateInfo.layout = m_spatialReusePipelineLayout;
  cpCreateInfo.stage = nvvk::createShaderStageInfo(m_device, nvh::loadFile("spv/spatial_reuse_ReSTIR.comp.spv", true, defaultSearchPaths, true),
                                                   VK_SHADER_STAGE_COMPUTE_BIT);

  vkCreateComputePipelines(m_device, {}, 1, &cpCreateInfo, nullptr, &m_spatialReusePipeline);

  vkDestroyShaderModule(m_device, cpCreateInfo.stage.module, nullptr);
}

void HelloVulkan::runSpatialReuse(VkCommandBuffer cmdBuf, ReSTIRControl& restirControl)
{

  m_debug.beginLabel(cmdBuf, "SpatialReuse");

  // Preparing for the compute shader
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_spatialReusePipeline);

  std::vector<VkDescriptorSet> descSets{m_spatialReuseDescSet, m_descSet};
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_spatialReusePipelineLayout, 0,
                          (uint32_t)descSets.size(),
                          descSets.data(), 0, nullptr);


  // Sending the push constant information
  restirControl.frame = m_frame;
  vkCmdPushConstants(cmdBuf, m_spatialReusePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReSTIRControl), &restirControl);

  // Dispatching the shader
  vkCmdDispatch(cmdBuf, (m_size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (m_size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);


  // Adding a barrier to be sure the compute shader has finished
  // writing to the AO buffer before the post shader is using it
  VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageMemoryBarrier    imgMemBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  imgMemBarrier.srcAccessMask    = VK_ACCESS_SHADER_WRITE_BIT;
  imgMemBarrier.dstAccessMask    = VK_ACCESS_SHADER_READ_BIT;
  imgMemBarrier.image            = m_aoBuffer.image;
  imgMemBarrier.oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
  imgMemBarrier.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
  imgMemBarrier.subresourceRange = range;
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       VK_DEPENDENCY_DEVICE_GROUP_BIT, 0, nullptr, 0, nullptr, 1, &imgMemBarrier);


  m_debug.endLabel(cmdBuf);
}

//////////////////////////////////////////////////////////////////////////
// Reset from JITTER CAM tutorial
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// If the camera matrix has changed, resets the frame otherwise, increments frame.
//
void HelloVulkan::updateFrame()
{
  static nvmath::mat4f refCamMatrix;
  static float         fov = 0;

  auto& m = CameraManip.getMatrix();
  auto  f = CameraManip.getFov();
  if(memcmp(&refCamMatrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || f != fov)
  {
    resetFrame();
    refCamMatrix = m;
    fov          = f;
  }

  m_frame++;
}

void HelloVulkan::resetFrame()
{

    //m_frame = -1;
  
}
