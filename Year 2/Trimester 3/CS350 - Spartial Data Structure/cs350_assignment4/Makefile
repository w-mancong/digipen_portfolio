########################################################################
# Makefile for Linux
#

ifeq ($(v), std)
    VFLAG := 
    ODIR := stdobjs
    shaders = lightingPhong.vert lightingPhong.frag
    udflags = -USOLUTION  -UGEOMLIB_SOLUTION
    pkgName = CS350-framework
else
    VFLAG := -DSOLUTION -DGEOMLIB_SOLUTION
    ODIR := solobjs
    shaders = lightingPhong.vert lightingPhong.frag
    udflags = -DSOLUTION -DGEOMLIB_SOLUTION
    pkgName = CS350-sol
endif

ifdef c
    pkgName=$(c)-framework
else
endif

# ifndef v
# $(error This makefile needs a flag: v=basic, v=tr, or v=sol)
# endif

# Search for the libs directory
ifneq (,$(wildcard libs))
    LIBDIR := libs
else
    ifneq (,$(wildcard ../libs))
        LIBDIR := ../libs
    else
        ifneq (,$(wildcard ../../libs))
            LIBDIR := ../../libs
        else
            LIBDIR := ../../../libs
        endif
    endif
endif

# Where the compiler will search for source files.
VPATH = $(LIBDIR)/imgui-master $(LIBDIR)/imgui-master/backends

# Where the .o files go
vpath %.o  $(ODIR)

CXX = g++
CFLAGS = -g $(VFLAG) -I. -I$(LIBDIR)/glm -I$(LIBDIR)/imgui-master -I$(LIBDIR)/imgui-master/backends -I$(LIBDIR)  -I$(LIBDIR)/glfw/include

CXXFLAGS = -std=c++17 $(CFLAGS) -Wno-deprecated

LIBS =  -L/usr/lib/x86_64-linux-gnu -L../$(LIBDIR) -L/usr/lib -L/usr/local/lib -lglbinding -lX11 -lGLU -lGL `pkg-config --static --libs glfw3`

CPPsrc = framework.cpp interact.cpp transform.cpp scene.cpp texture.cpp shapes.cpp object.cpp shader.cpp simplexnoise.cpp fbo.cpp geomlib-advanced.cpp
IMGUIsrc = imgui.cpp imgui_widgets.cpp imgui_draw.cpp imgui_demo.cpp imgui_impl_glfw.cpp imgui_impl_opengl3.cpp

headers = framework.h interact.h texture.h shapes.h object.h scene.h shader.h transform.h simplexnoise.h fbo.h geomlib.h
srcFiles = $(CPPsrc) $(shaders) $(headers)
extraFiles = framework.vcxproj Makefile

pkgDir = /home/gherron/packages
objs = $(patsubst %.cpp,%.o,$(CPPsrc)) $(patsubst %.cpp,%.o,$(IMGUIsrc))
target = $(ODIR)/framework.exe

$(target): $(objs)
	@echo Link $(target)
	cd $(ODIR) && $(CXX) -g  -o ../$@  $(objs) $(LIBS)

print:
	@echo  VFLAG=$(VFLAG)

help:
	@echo "Try:"
	@echo "    make -j8         run  // for base level -- no transformations or shading"
	@echo "    make -j8 v=tr    run  // for transformation level -- no shading"
	@echo "    make -j8 v=phong run  // for lighting level"            
	@echo "    make -j8 v=sol   run  // for full solution level"    
	@echo "    make -j8 v=em    run  // for GPU emulator"  
	@echo "    make -j8 v=emsol run  // for GPU emulator solution"
	@echo "Also:"
	@echo "   make v=em    c=CS200 zip // For CS200 -- bare bones"
	@echo "   make         c=CS251 zip // For CS251 -- bare bones"
	@echo "   make         c=CS541 zip // For CS541 -- bare bones"
	@echo "   make v=tr    c=CS300 zip // For CS300 -- includes transformations"
	@echo "   make v=phong c=CS562 zip // For CS562 -- includes transformations and Phong"
	@echo "   make v=sol   c=sol   zip // For whatever -- includes everything but GPU emulator"
	@echo "   make v=emsol c=emsol zip // For whatever -- includes everything"

run: $(target)
	LD_LIBRARY_PATH="$(LIBDIR);$(LD_LIBRARY_PATH)" ./$(target)

gdb: $(target)
	LD_LIBRARY_PATH="$(LIBDIR);$(LD_LIBRARY_PATH)" gdb ./$(target)

what:
	@echo VPATH = $(VPATH)
	@echo LIBS = $(LIBDIR)
	@echo CFLAGS = $(CFLAGS)
	@echo VFLAG = $(VFLAG)
	@echo objs = $(objs)
	@echo shaders = $(shaders)
	@echo pkgName = $(pkgName)
	@echo pkgDir = $(pkgDir)
	@echo srcFiles = $(srcFiles)
	@echo extraFiles = $(extraFiles)
	@echo udflags = $(udflags)

clean:
	rm -rf solobjs stdobjs dependencies

%.o: %.cpp
	@echo Compile $<  $(VFLAG)
	@mkdir -p $(ODIR)
	@$(CXX) -c $(CXXFLAGS) $< -o $(ODIR)/$@

%.o: %.c
	@echo Compile $< $(VFLAG)
	@mkdir -p $(ODIR)
	@$(CC) -c $(CFLAGS) $< -o $(ODIR)/$@

zip:
	rm -rf $(pkgDir)/$(pkgName) $(pkgDir)/$(pkgName).zip
	mkdir $(pkgDir)/$(pkgName)
	cp $(srcFiles) $(pkgDir)/$(pkgName)
	cp -r $(extraFiles) $(pkgDir)/$(pkgName)
	cp -r ../libs $(pkgDir)/$(pkgName)
	rm -rf $(pkgDir)/$(pkgName)/libs/.hg $(pkgDir)/$(pkgName)/libs/assimp $(pkgDir)/$(pkgName)/libs/bvh
	cd $(pkgDir)/$(pkgName);  unifdef $(udflags)  -m  $(srcFiles) || /bin/true

	cd $(pkgDir);  zip -r $(pkgName).zip $(pkgName)

ws:
	unix2dos $(srcFiles)
	@echo
	@echo ========= TABS:
	@grep -P '\t' $(srcFiles)

dependencies: 
	g++ -MM $(CXXFLAGS) $(CPPsrc) > dependencies

include dependencies
