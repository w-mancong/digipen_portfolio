////////////////////////////////////////////////////////////////////////
// The scene class contains all the parameters needed to define and
// draw a simple scene, including:
//   * Geometry
//   * Light parameters
//   * Material properties
//   * Viewport size parameters
//   * Viewing transformation values
//   * others ...
//
// Some of these parameters are set when the scene is built, and
// others are set by the framework in response to user mouse/keyboard
// interactions.  All of them can be used to draw the scene.

#include "shapes.h"
#include "object.h"
#include "texture.h"
#include "fbo.h"
#include "geomlib.h"

enum ObjectIds {
    nullId	= 0,
    skyId	= 1,
    seaId	= 2,
    groundId	= 3,
    treeId	= 4,
    debugId	= 5
};

class Shader;


class Scene
{
public:
    int polyCount;
    int ellipsoidCount;
    int triangleCount;
    GLFWwindow* window;

    // Light parameters
    float lightSpin, lightTilt, lightDist;
    glm::vec3 lightPos;

    bool drawReflective;
    bool nav;
    bool w_down, s_down, a_down, d_down;
    float spin, tilt, speed, ry, front, back;
    glm::vec3 eye, tr;
    float last_time;
    int mode; // Extra mode indicator hooked up to number keys and sent to shader
    
    // Viewport
    int width, height;

    // Transformations
    glm::mat4 WorldProj, WorldView, WorldInverse;

    // All objects in the scene are children of this single root object.
    Object* objectRoot;
    Object *sky, *ground, *sea, *objects;

    ProceduralGround* proceduralground;

    // Shader programs
    ShaderProgram* lightingProgram;

    void InitializeScene();
    void BuildTransforms();
    void DrawGUI();
    void DrawScene();
    void DestroyScene();

    // ImGui variables
    bool show_demo_window;
    bool drawAABB{};
    bool drawTriangleAABB{};
    bool drawLineSegments{};
    bool drawTriangle{};
    bool drawSphere{};

    // All Debugging objects delcared here
    unsigned int VaoFromPoints(std::vector<glm::vec4> pnt, std::vector<int> ind);

    // Line segment
    void DrawLineSegment();
    void DrawLineSegment(glm::vec3 const& p0, glm::vec3 const& p1);
    
    // AABB
    void DrawAABB();
    void DrawTriangleAABB();

    // Triangle
    void DrawTriangle();

    // Sphere
    void DrawSphere();

    glm::mat4 RotateZtoV(glm::vec3 v);

    std::vector<std::pair<glm::vec3, glm::vec3>> lineSegments;
    unsigned int segmentVao, aabbVao, triangleVao, sphereVao;
    glm::vec4 debugColor{};
    static size_t constexpr N = 48;

private:
    enum class NodeType
    {
        Invalid = -1,
        Internal,
        Leaf
    };

    struct Node
    {
        Box3D aabb{};
        Triangle3D tri{};
        glm::mat4 const* modelTr{};
    };

    struct TreeNode
    {
        NodeType type{ NodeType::Invalid };
        Node const* node;
        Box3D aabb{};
        int numOfObjects{};
        TreeNode* lChild{ nullptr }, *rChild{ nullptr };
    };

    using MinMax = std::pair<glm::vec3, glm::vec3>;

    void BuildTopDownTree(TreeNode* node, std::vector<Node>& nodes_, size_t start, size_t end);
    Box3D CombineBV(std::vector<Node> const& nodes_, size_t start, size_t numOfObjects) const;
    size_t PartitionNodes(std::vector<Node>& nodes_, size_t start, size_t numOfObjects) const;
    // first = min, second = max
    MinMax GetMinMax(std::vector<Node> const& nodes_, size_t start, size_t end) const;
    void GetHeightOfTree(void);
    int Height(TreeNode const* node);
    void RenderTree(TreeNode const* node, int depth) const;

    bool renderTree;
    TreeNode* root{ nullptr };
    std::vector<Node> nodes{};
    int heightOfTree{}, drawDepth{};
};
