#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cstdlib>
#include <cstring>

#include <glm/glm.hpp>
#include <glm/matrix.hpp>

#include <vector>

static constexpr const uint32_t WIDTH = 800;
static constexpr const uint32_t HEIGHT = 600;

std::vector<char const*> const validationLayers =
{
	"VK_LAYER_KHRONOS_validation"
};

#ifdef _DEBUG
#define VK_DEBUGGING
#endif

class HelloTriangleApplication
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:	// functions
	void initWindow() 
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}

	void initVulkan()
	{
		createInstance();
	}

	void mainLoop()
	{
		while ( !glfwWindowShouldClose(window) ) 
		{
			glfwPollEvents();

		}
	}

	void cleanup()
	{
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}

	void createInstance()
	{
#ifdef VK_DEBUGGING
		if (!checkValidationLayerSupport())
			throw std::runtime_error("[Vulkan] Validation layers requested, but not available!");
#endif

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;
		appInfo.pNext = nullptr;

		uint32_t glfwExtensionCount = 0;
		char const** glfwExtensions = nullptr;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		{
			std::cout << "Required GLFW extensions:" << std::endl;
			for (uint32_t i{}; i < glfwExtensionCount; ++i)
				std::cout << '\t' << *(glfwExtensions + i) << std::endl;
		}

		{	// Code use to check for extension support
			uint32_t extensionCount = 0;
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
			std::vector<VkExtensionProperties> extensions(extensionCount);
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

			std::cout << std::endl << "Available extensions:" << std::endl;
			for (VkExtensionProperties const& extension : extensions)
				std::cout << '\t' << extension.extensionName << std::endl;
		}

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledExtensionCount = glfwExtensionCount;
		createInfo.ppEnabledExtensionNames = glfwExtensions;
#ifdef VK_DEBUGGING
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
#else
		createInfo.enabledLayerCount = 0;
		createInfo.ppEnabledLayerNames = nullptr;
#endif
		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
			throw std::runtime_error("[Vulkan] Failed to create instance!");
	}

#ifdef VK_DEBUGGING
	bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (char const* layerName : validationLayers)
		{
			bool layerFound = false;

			for (VkLayerProperties const& layerProperties : availableLayers)
			{
				if (!strcmp(layerName, layerProperties.layerName))
				{
					layerFound = true;
					break;
				}
			}

			if (!layerFound)
				return false;
		}

		return true;
	}
#endif

private:	// variables
	GLFWwindow* window{ nullptr };
	VkInstance instance{};
};

int main() 
{
	HelloTriangleApplication app;

	try {
		app.run();
	}
	catch (std::exception const& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}