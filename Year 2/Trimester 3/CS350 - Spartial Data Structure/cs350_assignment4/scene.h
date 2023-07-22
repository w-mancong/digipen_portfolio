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
#include <map>

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
    // BVH
    enum class NodeType
    {
        Invalid = -1,
        Internal,
        Leaf
    };

    struct Node
    {
        Box3D aabb{};                   // Triangle's aabb
        Triangle3D tri{};               // Vertices to the current triangle
        //glm::mat4 const* modelTr{};     // Model transformation to transform this current triangle node
    };

    struct TreeNode
    {
        NodeType type{ NodeType::Invalid };                 // Type of current tree node
        Node const* node;                                   // If this current tree node is a leaf node, then this will be used
        Box3D aabb{};                                       // AABB of this current tree node
        int numOfObjects{};                                 // Total number of objects in this tree node
        TreeNode* lChild{ nullptr }, *rChild{ nullptr };    // If this current tree node is an internal node, will contain a left/right child
    };

    using MinMax = std::pair<glm::vec3, glm::vec3>;

    void BuildTopDownTree(TreeNode* node, std::vector<Node>& nodes_, size_t start, size_t end);
    Box3D CombineBV(std::vector<Node> const& nodes_, size_t start, size_t numOfObjects) const;
    size_t PartitionNodes(std::vector<Node>& nodes_, size_t start, size_t numOfObjects) const;
    void TestRayAgainstNode(Ray3D const& ray, TreeNode const* node, float& min_t) const;
    // first = min, second = max
    MinMax GetMinMax(std::vector<Node> const& nodes_, size_t start, size_t end) const;
    void GetHeightOfTree(void);
    int Height(TreeNode const* node);
    void RenderTree(TreeNode const* node, int depth) const;
    void RenderAllTree(TreeNode const* node, int depth) const;

    //bool renderTree{}, renderTreeAll{};
    TreeNode* root{ nullptr };
    std::vector<Node> nodes{};
    int heightOfTree{}, drawDepth{};
    std::vector<glm::vec3> treeLayerColors;

    // KD Tree
    static float constexpr const cI = 1.0f, cT = 1.0f;

    struct KDTree
    {
        KDTree* lChild{ nullptr }, *rChild{ nullptr };
        NodeType type{ NodeType::Invalid };
        Node const* node{};
        Box3D aabb{};
        std::vector<Node*> nodes{};
    } *kd_root{ nullptr };

    struct SweepAlgoVar
    {
        float cost{ 0.0f };             // cmin - minimum cost
        float splitPoint{ 0.0f };       // split point that produces the cmin
        int splitAxis{ 0 };             // split axis  that produces the cmin
    };

    struct SharedPoints // Group the shared points together
    {
        int start{ 0 }, end{ 0 }, in_plane{ 0 };    // integers used to keep track of the number shared points
    };

    enum class Grouping
    {
        Start = 0,
        End, 
        In_Plane,
    };

    using GroupSharedPoints = std::pair<float, Grouping>;
    using Box3DPair = std::pair<Box3D, Box3D>;
    using SplitTriangle = std::pair<std::vector<Node*>, std::vector<Node*>>;    // first: lhs of triangles, second: rhs of triangles

    int kd_renderDepth{};
    int min_depth{}, max_depth{};
    // Rendering variables for kd tree
    bool renderKDTree{ false }, renderAllKDTree{ false };

    void BuildKDTree(void);
    KDTree* BuildNode(Box3D const& box, std::vector<Node*>& tri_nodes, int depth);
    float ComputeSurfaceArea(Box3D const& box) const;
    float ProbabilityOfIntersection(Box3D const& lhs, Box3D const& rhs) const;
    Box3DPair SplitBoxAtSPlane(Box3D const& box, float splitPoint, int splitAxis) const;
    SplitTriangle SplitUpTriangles(std::vector<Node*> const& triangles, SweepAlgoVar const& result);
    Box3DPair SplitUpBoundingBox(SplitTriangle triangles) const;
    Box3D MakeBoundingBox(std::vector<Node*> nodes_) const;
    int GetMinDepth(KDTree* node) const;
    int GetMaxDepth(KDTree* node) const;

    void RenderKDTree(KDTree* node, int depth) const;
    void RenderAllKDTree(KDTree* node, int depth) const;
    void KdIntersect(Ray3D const& ray, KDTree const* node, float& min_t) const;
};
