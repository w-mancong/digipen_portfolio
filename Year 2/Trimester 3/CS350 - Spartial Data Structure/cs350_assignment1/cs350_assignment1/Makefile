ifeq ($(v), sol)
    VFLAG := -DGEOMLIB_SOLUTION
    ODIR := sobjs
    udflags = -DGEOMLIB_SOLUTION
else
    VFLAG := 
    ODIR := objs
    udflags = -UGEOMLIB_SOLUTION
endif

# Where the compiler will search for source files.
VPATH = $(ODIR)

# Where the .o files go
vpath %.o  $(ODIR)


LIBDIR := /home/gherron/projects/frameworks/libs

CXXFLAGS =  $(VFLAG)  -I. -g -I$(LIBDIR)/glm 
LIBS = 

target = $(ODIR)/framework.exe

src = geomlib-main.cpp geomlib.cpp geomlib-advanced.cpp
headers = geomlib.h
extras = framework.vcxproj Makefile

objs = $(patsubst %.cpp,%.o,$(src))

srcFiles = $(src) $(headers) $(extras)
pkgName = CS350-geomlib
pkgDir = /home/gherron/packages

$(target): $(objs)
	@echo Link $(target)
	cd $(ODIR) && g++ -g  -o ../$@  $(objs) $(LIBS)

help:
	@echo "Try:"
	@echo "    make -j8         run  // for base level"
	@echo "    make -j8 v=sol    run  // for solution level"

%.o: %.cpp
	@echo Compile $<
	@mkdir -p $(ODIR)
	$(CXX) -c $(CXXFLAGS) $< -o $(ODIR)/$@  $(VFLAG)

run: $(target)
	./$(target)

zip: 
	rm -rf $(pkgDir)/$(pkgName) $(pkgDir)/$(pkgName).zip
	mkdir $(pkgDir)/$(pkgName)
	cp $(srcFiles) $(pkgDir)/$(pkgName)
	cp -r ~/projects/frameworks/libs $(pkgDir)/$(pkgName)
	rm -rf $(pkgDir)/$(pkgName)/libs/.hg $(pkgDir)/$(pkgName)/libs/assimp $(pkgDir)/$(pkgName)/libs/bvh
	cd $(pkgDir)/$(pkgName);  unifdef -UGEOMLIB_SOLUTION -m  $(srcFiles) || /bin/true
	cd $(pkgDir);  zip -r $(pkgName).zip $(pkgName)
	cd $(pkgDir);  cp -r $(pkgName) ~/classes/test

clean:
	rm -f *.o *~ framework.exe dependencies

dependencies: 
	g++ -MM $(CXXFLAGS) $(src) > dependencies

include dependencies
