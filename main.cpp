#include <array>
#include <iostream>

#include "backends/imgui_impl_glfw.h"
#include "imgui.h"

#include "hello_vulkan.h"
#include "imgui/imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvpsystem.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"

//#define CORNELL_4_LIGHTS
//#define CORNELL_SPECULAR
//#define CORNELL_EMISSION_MAP_TEST
//#define CORNELL_WORLD_SPACE_TEST
//#define CORNELL_ABSTRACT
#define SPONZA



//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;


// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Extra UI
void renderUI(HelloVulkan& helloVk)
{
  ImGuiH::CameraWidget();
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
static int const SAMPLE_WIDTH  = 1280;
static int const SAMPLE_HEIGHT = 720;


//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
  UNUSED(argc);

  // Setup GLFW window
  glfwSetErrorCallback(onErrorCallback);
  if(!glfwInit())
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, "Vulkan RT ReSTIR DI", nullptr, nullptr);

  ReSTIRControl restirControl;

  // Setup camera
  CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
#ifdef CORNELL_4_LIGHTS
  CameraManip.setLookat(nvmath::vec3f(0, -0.5, 40), nvmath::vec3f(0, -0.5, 0), nvmath::vec3f(0, 1, 0));
#endif
#ifdef CORNELL_SPECULAR
  CameraManip.setLookat(nvmath::vec3f(0, -0.5, 40), nvmath::vec3f(0, -0.5, 0), nvmath::vec3f(0, 1, 0));
#endif
#ifdef CORNELL_EMISSION_MAP_TEST
  CameraManip.setLookat(nvmath::vec3f(0, -0.5, 40), nvmath::vec3f(0, -0.5, 0), nvmath::vec3f(0, 1, 0));
#endif
#ifdef CORNELL_WORLD_SPACE_TEST
  CameraManip.setLookat(nvmath::vec3f(0, -0.5, 40), nvmath::vec3f(0, -0.5, 0), nvmath::vec3f(0, 1, 0));
#endif
#ifdef CORNELL_ABSTRACT
  CameraManip.setLookat(nvmath::vec3f(0, -0.5, 40), nvmath::vec3f(0, -0.5, 0), nvmath::vec3f(0, 1, 0));
#endif
#ifdef SPONZA
  CameraManip.setLookat(nvmath::vec3f(-6.5, 1.0, -0.5), nvmath::vec3f(2.5, 1.0, -0.3), nvmath::vec3f(0, 1, 0));
  CameraManip.setFov(50.0f);
#endif

  // Setup Vulkan
  if(!glfwVulkanSupported())
  {
    printf("GLFW: Vulkan Not Supported\n");
    return 1;
  }

  // setup some basic things for the sample, logging file for example
  NVPSystem system(PROJECT_NAME);

  // Search path for shaders and other media
  defaultSearchPaths = {
      NVPSystem::exePath() + PROJECT_RELDIRECTORY,
      NVPSystem::exePath() + PROJECT_RELDIRECTORY "..",
      std::string(PROJECT_NAME),
  };

  // Vulkan required extensions
  assert(glfwVulkanSupported() == 1);
  uint32_t count{0};
  auto     reqExtensions = glfwGetRequiredInstanceExtensions(&count);

  // Requesting Vulkan extensions and layers
  nvvk::ContextCreateInfo contextInfo;
  contextInfo.setVersion(1, 2);                       // Using Vulkan 1.2
  for(uint32_t ext_id = 0; ext_id < count; ext_id++)  // Adding required extensions (surface, win32, linux, ..)
    contextInfo.addInstanceExtension(reqExtensions[ext_id]);
  contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);              // FPS in titlebar
  contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);  // Allow debug names
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);            // Enabling ability to present rendering

  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);  // To build acceleration structures
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);  // Ray tracing in compute shader
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline

  // Creating Vulkan base application
  nvvk::Context vkctx{};
  vkctx.initInstance(contextInfo);
  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
  assert(!compatibleDevices.empty());
  // Use a compatible device
  vkctx.initDevice(compatibleDevices[0], contextInfo);

  // Create example
  HelloVulkan helloVk;

  // Window need to be opened to get the surface on which to draw
  const VkSurfaceKHR surface = helloVk.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);

  helloVk.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
  helloVk.createSwapchain(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  helloVk.createDepthBuffer();
  helloVk.createRenderPass();
  helloVk.createFrameBuffers();

  // Setup Imgui
  helloVk.initGUI(0);  // Using sub-pass 0

  // Creation of the example
  nvmath::mat4f t = nvmath::translation_mat4(nvmath::vec3f{0, 0.0, 0});
#ifdef CORNELL_4_LIGHTS
  helloVk.loadSceneGLTF(nvh::findFile("media/scenes/cornell_561.gltf", defaultSearchPaths, true));
#endif
#ifdef CORNELL_SPECULAR
  helloVk.loadSceneGLTF(nvh::findFile("media/scenes/cornell_specular.gltf", defaultSearchPaths, true));
#endif
#ifdef CORNELL_EMISSION_MAP_TEST
  helloVk.loadSceneGLTF(nvh::findFile("media/scenes/cornell_emission_map_test.gltf", defaultSearchPaths, true));
#endif
#ifdef CORNELL_WORLD_SPACE_TEST
  helloVk.loadSceneGLTF(nvh::findFile("media/scenes/cornell_world_space_test.gltf", defaultSearchPaths, true));
#endif
#ifdef CORNELL_ABSTRACT
  helloVk.loadSceneGLTF(nvh::findFile("media/scenes/cornell_box_many_lights.gltf", defaultSearchPaths, true));
#endif
#ifdef SPONZA
  helloVk.loadSceneGLTF(nvh::findFile("media/scenes/emission/sponz.gltf", defaultSearchPaths, true));
#endif
  

  helloVk.createOffscreenRender();
  helloVk.createDescriptorSetLayout();
  helloVk.createGraphicsPipeline();
  helloVk.createUniformBuffer();
  //helloVk.createObjDescriptionBuffer();

  // #VKRay
  helloVk.initRayTracing();
  helloVk.createBottomLevelAS();
  helloVk.createTopLevelAS();

  // Need the Top level AS
  helloVk.updateDescriptorSet();

  helloVk.createPostDescriptor();
  helloVk.createPostPipeline();
  helloVk.updatePostDescriptorSet();

  helloVk.createReservoirBuffers();

  helloVk.createCompDescriptors();
  helloVk.updateCompDescriptors();
  helloVk.createCompPipelines();

  helloVk.createSpatialReuseDescriptors();
  helloVk.updateSpatialReuseDescriptors();
  helloVk.createSpatialReusePipelines();


  nvmath::vec4f clearColor = nvmath::vec4f(0, 0, 0, 0);

  helloVk.setupGlfwCallbacks(window);
  ImGui_ImplGlfw_InitForVulkan(window, true);


  


  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    try
    {
      glfwPollEvents();
      if(helloVk.isMinimized())
        continue;

      // Start the Dear ImGui frame
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      // Show UI window.
      if(helloVk.showGui())
      {
        ImGuiH::Panel::Begin();
        ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(&clearColor));

        renderUI(helloVk);
        ImGui::SetNextItemOpen(true, ImGuiCond_Once);
        if(ImGui::CollapsingHeader("Rendering Settings"))
        {
          bool changed{false};
          changed |= ImGui::SliderInt("SPP", &restirControl.spp, 1, 128);
          changed |= ImGui::SliderInt("Max Ray Depth", &restirControl.ray_depth, 1, 5);
          changed |= ImGui::SliderFloat("Light Intensity", &restirControl.light_intensity, 0.1f, 100.0f);
          //float* light_color_c_arr[3] = {&restirControl.light_color.x, &restirControl.light_color.y, &restirControl.light_color.z};
          //changed |= ImGui::SliderFloat3("Light Color", *light_color_c_arr, 0.01, 1.0);
          changed |= ImGui::SliderFloat("Light Color R", &restirControl.light_color_X, 0.01f, 1.0f);
          changed |= ImGui::SliderFloat("Light Color G", &restirControl.light_color_Y, 0.01f, 1.0f);
          changed |= ImGui::SliderFloat("Light Color B", &restirControl.light_color_Z, 0.01f, 1.0f);
          changed |= ImGui::InputInt("Randomize Color", &restirControl.randomize_color);
          changed |= ImGui::InputInt("Two-Sided Lights", &restirControl.two_sided_lights);
          changed |= ImGui::InputInt("Light Type", &restirControl.light_type);
          changed |= ImGui::InputInt("DI", &restirControl.mode);
          changed |= ImGui::InputInt("GI DI", &restirControl.GI_mode);
          changed |= ImGui::InputInt("# RIS Samples", &restirControl.M);
          changed |= ImGui::InputInt("# Neighbor Samples", &restirControl.num_neighbors);
          changed |= ImGui::InputInt("Neighbor Radius", &restirControl.neighbor_radius);
          if(changed)
            helloVk.resetFrame();
        }

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGuiH::Control::Info("", "", "(F10) Toggle Pane", ImGuiH::Control::Flags::Disabled);
        ImGuiH::Panel::End();
      }
      helloVk.cur_rendering_mode = restirControl.mode;

      // Start rendering the scene
      helloVk.prepareFrame();

      // Start command buffer of this frame
      auto                   curFrame = helloVk.getCurFrame();
      const VkCommandBuffer& cmdBuf   = helloVk.getCommandBuffers()[curFrame];

      VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cmdBuf, &beginInfo);

      // Updating camera buffer
      helloVk.updateUniformBuffer(cmdBuf);

      // Clearing screen
      std::array<VkClearValue, 3> clearValues{};
      clearValues[0].color = {{clearColor[0], clearColor[1], clearColor[2], clearColor[3]}};


      // Offscreen render pass
      {
        clearValues[1].color        = {{0, 0, 0, 0}};
        clearValues[2].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo offscreenRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        offscreenRenderPassBeginInfo.clearValueCount = (uint32_t)clearValues.size();
        offscreenRenderPassBeginInfo.pClearValues    = clearValues.data();
        offscreenRenderPassBeginInfo.renderPass      = helloVk.m_offscreenRenderPass;
        offscreenRenderPassBeginInfo.framebuffer     = helloVk.m_offscreenFramebuffer;
        offscreenRenderPassBeginInfo.renderArea      = {{0, 0}, helloVk.getSize()};

        // Rendering Scene
        {
          vkCmdBeginRenderPass(cmdBuf, &offscreenRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
          helloVk.rasterize(cmdBuf);
          vkCmdEndRenderPass(cmdBuf);
          helloVk.runCompute(cmdBuf, restirControl);
          helloVk.runSpatialReuse(cmdBuf, restirControl);
        }
      }

      // 2nd rendering pass: tone mapper, UI
      {
        clearValues[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo postRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        postRenderPassBeginInfo.clearValueCount = 2;
        postRenderPassBeginInfo.pClearValues    = clearValues.data();
        postRenderPassBeginInfo.renderPass      = helloVk.getRenderPass();
        postRenderPassBeginInfo.framebuffer     = helloVk.getFramebuffers()[curFrame];
        postRenderPassBeginInfo.renderArea      = {{0, 0}, helloVk.getSize()};

        // Rendering tonemapper
        vkCmdBeginRenderPass(cmdBuf, &postRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        helloVk.drawPost(cmdBuf);
        // Rendering UI
        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);
        vkCmdEndRenderPass(cmdBuf);
      }

      // Submit for display
      vkEndCommandBuffer(cmdBuf);
      helloVk.submitFrame();
      helloVk.m_time++;
    }
    catch(const std::system_error& e)
    {
      if(e.code().value() == VK_ERROR_DEVICE_LOST)
      {
#if _WIN32
        MessageBoxA(nullptr, e.what(), "Fatal Error", MB_ICONERROR | MB_OK | MB_DEFBUTTON1);
#endif
      }
      std::cout << e.what() << std::endl;
      return e.code().value();
    }
  }

  // Cleanup
  vkDeviceWaitIdle(helloVk.getDevice());

  helloVk.destroyResources();
  helloVk.destroy();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
