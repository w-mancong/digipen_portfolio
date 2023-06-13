////////////////////////////////////////////////////////////////////////
// All keyboard, mouse, and other interactions are implemented here.
// The single entry point, InitInteraction, sets up GLFW callbacks for
// various events that an interactive graphics program needs to
// handle.
//

#include "framework.h"

extern Scene scene;       // Declared in framework.cpp, but used here.

// Some globals used for mouse handling.
double mouseX, mouseY;
bool shifted = false;
bool leftDown = false;
bool middleDown = false;
bool rightDown = false;
bool control = false;

////////////////////////////////////////////////////////////////////////
// Function called to exit
void Quit(void *clientData)
{
    glfwSetWindowShouldClose(scene.window, 1);
}

std::string ACTION[3] = {"Release", "Press", "Repeat"};

////////////////////////////////////////////////////////////////////////
// Called for keyboard actions.

void Keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action == GLFW_REPEAT) return; // Because keyboard autorepeat is evil.
        
    //printf("Keyboard %c(%d);  S%d %s M%d\n", key, key, scancode, ACTION[action].c_str(), mods);
    fflush(stdout);
    
    // Track SHIFT/NO-SHIFT transitions. (The mods parameter should do this, but doesn't.)
    if (key==GLFW_KEY_LEFT_SHIFT || key==GLFW_KEY_RIGHT_SHIFT)
        shifted = !shifted;
    if (key==GLFW_KEY_LEFT_CONTROL || key==GLFW_KEY_RIGHT_CONTROL)
        control = !control;
  
    if  (action == GLFW_PRESS) {
        switch(key) {

        case GLFW_KEY_TAB:
            scene.nav = !scene.nav;
            break;
        
        case GLFW_KEY_W: scene.w_down = true; break;
        case GLFW_KEY_S: scene.s_down = true; break;
        case GLFW_KEY_A: scene.a_down = true; break;
        case GLFW_KEY_D: scene.d_down = true; break;

        case GLFW_KEY_0: case GLFW_KEY_1: case GLFW_KEY_2: case GLFW_KEY_3: case GLFW_KEY_4:
        case GLFW_KEY_5: case GLFW_KEY_6: case GLFW_KEY_7: case GLFW_KEY_8: case GLFW_KEY_9:
            scene.mode = key-GLFW_KEY_0;
            break;
        case GLFW_KEY_ESCAPE: case GLFW_KEY_Q: // Escape and 'q' keys quit the application
            exit(0); } }
        
    else if (action == GLFW_RELEASE) {

        switch(key) {
        case GLFW_KEY_W: scene.w_down = false; break;
        case GLFW_KEY_S: scene.s_down = false; break;
        case GLFW_KEY_A: scene.a_down = false; break;
        case GLFW_KEY_D: scene.d_down = false; break; } }
}

////////////////////////////////////////////////////////////////////////
// Called when a mouse button changes state.
void MouseButton(GLFWwindow* window, int button, int action, int mods)
{        

    if (ImGui::GetIO().WantCaptureMouse) return;

    glfwGetCursorPos(window, &mouseX, &mouseY);
    //printf("MouseButton %d %d %d %f %f\n", button, action, mods, mouseX, mouseY);
    
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        leftDown = (action == GLFW_PRESS); }

    else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        middleDown = (action == GLFW_PRESS);  }

    else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        rightDown = (action == GLFW_PRESS); }
}

////////////////////////////////////////////////////////////////////////
// Called by GLFW when a mouse moves (while a button is down)
void MouseMotion(GLFWwindow* window, double x, double y)
{
    if (ImGui::GetIO().WantCaptureMouse) return;
                     
    // Calculate the change in the mouse position
    int dx = x-mouseX;
    int dy = y-mouseY;

    if (leftDown && shifted) {  // Rotate light position
        scene.lightSpin += dx/3.0;
        scene.lightTilt -= dy/3.0; }

    else if (leftDown) {
        // Rotate eye position
        scene.spin += dx/3.0;
        scene.tilt += dy/3.0; 
    }

    if (middleDown) { }

    if (rightDown) {
        scene.tr[0] += dx/40.0f;
        scene.tr[1] -= dy/40.0f; 
    }

    // Record this position
    mouseX = x;
    mouseY = y;
}


void Scroll(GLFWwindow* window, double x, double y)
{
    if (ImGui::GetIO().WantCaptureMouse)  return;

    //printf("Scroll %f %f\n", x, y);
    
    // Figure out the mouse action, and handle accordingly
    if (y>0.0 && shifted) { // Scroll light in
        scene.lightDist = pow(scene.lightDist, 1.0f/1.02f); }

    else if (y<0.0 && shifted) { // Scroll light out
        scene.lightDist = pow(scene.lightDist, 1.02f); }

    else if (control) { // Scroll light out
        float a = y<0.0 ? 1.03 : 1.0/1.03;
        scene.ry /= a;
        scene.tr[2] *= a; }
        
    else if (y>0.0) {
        scene.tr[2] = pow(scene.tr[2], 1.0f/1.02f);
    }

    else if (y<0.0) {
        scene.tr[2] = pow(scene.tr[2], 1.02f);
    }
}

void InitInteraction()
{
    glfwSetKeyCallback(scene.window, Keyboard);
    glfwSetMouseButtonCallback(scene.window, MouseButton);
    glfwSetCursorPosCallback(scene.window, MouseMotion);
    glfwSetScrollCallback(scene.window, Scroll);
}
