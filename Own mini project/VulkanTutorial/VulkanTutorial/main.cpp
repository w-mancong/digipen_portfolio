#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <set>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>

#include <glm/glm.hpp>
#include <glm/matrix.hpp>

#include <vector>

static constexpr const uint32_t WIDTH = 800;
static constexpr const uint32_t HEIGHT = 600;

std::vector<char const*> const validationLayers =
{
	"VK_LAYER_KHRONOS_validation",
};

std::vector<char const*> const deviceExtensions =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

#ifdef _DEBUG
#define VK_DEBUGGING
#endif

#ifdef VK_DEBUGGING
VkResult CreateDebugUtilsMessengerEXT(
	VkInstance									instance, 
	VkDebugUtilsMessengerCreateInfoEXT const*	pCreateInfo, 
	VkAllocationCallbacks const*				pAllocator, 
	VkDebugUtilsMessengerEXT*					pDebugMessenger)
{
	PFN_vkCreateDebugUtilsMessengerEXT func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>( vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT") );
	if (func)
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(
	VkInstance						instance, 
	VkDebugUtilsMessengerEXT		debugMessenger, 
	VkAllocationCallbacks const*	pAllocator)
{
	PFN_vkDestroyDebugUtilsMessengerEXT func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>( vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT") );
	if (func)
		func(instance, debugMessenger, pAllocator);
}
#endif

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete()
	{
		return graphicsFamily.has_value() && 
			   presentFamily.has_value();
	}
};

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
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
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
		vkDestroyDevice(device, nullptr);

#ifdef VK_DEBUGGING
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
#endif

		vkDestroySurfaceKHR(instance, surface, nullptr);
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

		std::vector<char const*> extensions = getRequiredExtensions();

		{
			std::cout << "Required GLFW extensions:" << std::endl;
			for (uint32_t i{}; i < extensions.size(); ++i)
				std::cout << '\t' << extensions[i] << std::endl;
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
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();
#ifdef VK_DEBUGGING
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		populateDebugMessengerCreateInfo(debugCreateInfo);
		createInfo.pNext = reinterpret_cast<VkDebugUtilsMessengerCreateInfoEXT*>(&debugCreateInfo);
#else
		createInfo.enabledLayerCount = 0;
		createInfo.ppEnabledLayerNames = nullptr;
#endif
		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
			throw std::runtime_error("[Vulkan] Failed to create instance!");
	}

	void pickPhysicalDevice()
	{
		uint32_t deviceCount = 0;	
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0)
			throw std::runtime_error("[Vulkan] Failed to find GPUs with Vulkan support!");

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (VkPhysicalDevice const& device : devices)
		{
			if (isDeviceSuitable(device))
			{
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE)
			throw std::runtime_error("[Vulkan] Failed to find a suitable GPU!");
	}

	bool isDeviceSuitable(VkPhysicalDevice device)
	{
		/*
			Example code of our application only using dedicated 
			graphics cards that support geometry shaders

			VkPhysicalDeviceProperties deviceProperties;
			VkPhysicalDeviceFeatures deviceFeatures;
			vkGetPhysicalDeviceProperties(device, &deviceProperties);
			vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

			return deviceProperties.deviceID == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
				   deviceFeatures.geometryShader;
			
			Can also choose a device by giving it an arbitary score system and choosing
			the GPU that scores the highest!
		*/
		QueueFamilyIndices indices = findQueueFamilies(device);
		bool extensionSupported = checkDeviceExtensionSupport(device);

		return indices.isComplete() && extensionSupported;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device)
	{
		uint32_t extensionCount{};
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions( deviceExtensions.begin(), deviceExtensions.end() );

		for (VkExtensionProperties const& extension : availableExtensions)
		{
			requiredExtensions.erase(extension.extensionName);
		}

		// if the required extensions all exist, the container will be empty
		return requiredExtensions.empty();
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int32_t i = 0;
		for (VkQueueFamilyProperties const& queueFamily : queueFamilies)
		{
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				indices.graphicsFamily = i;

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			if (presentSupport)
				indices.presentFamily = i;

			if (indices.isComplete())
				break;

			++i;
		}

		return indices;
	}

	void createLogicalDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.pEnabledFeatures = &deviceFeatures;
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

#ifdef VK_DEBUGGING
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
#else
		createInfo.enabledLayerCount = 0;
		createInfo.ppEnabledLayerNames = nullptr;
#endif

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
			throw std::runtime_error("[Vulkan] Failed to create logical device!");

		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
			throw std::runtime_error("[Vulkan] Failed to create window surface!");
	}

	std::vector<char const*> getRequiredExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		char const** glfwExtensions = nullptr;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		std::vector<char const*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifdef VK_DEBUGGING
		extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

		return extensions;
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

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		VkDebugUtilsMessengerCallbackDataEXT const * pCallbackData,
		void* pUserData)
	{
		std::cerr << "[Vulkan] Validation layer: " << pCallbackData->pMessage << std::endl;
		return VK_FALSE;
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
	{
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
									 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
									 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType	   = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
								     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
								     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

#endif

	void setupDebugMessenger()
	{
#ifdef VK_DEBUGGING
		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
			throw std::runtime_error("[Vulkan] Failed to set up debug messenger!");
#endif 
	}

private:	// variables
	GLFWwindow* window{ nullptr };
	VkInstance instance{};
	VkSurfaceKHR surface{};
	VkQueue presentQueue{};
#ifdef VK_DEBUGGING
	VkDebugUtilsMessengerEXT debugMessenger{};
#endif
	VkPhysicalDevice physicalDevice{ VK_NULL_HANDLE }; // This variable represents the actual physical GPU
	VkDevice device;								   // This variable represents the application view of the actual GPU. "Logical Device"
	VkQueue graphicsQueue;
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