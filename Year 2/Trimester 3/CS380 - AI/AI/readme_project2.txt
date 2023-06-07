Student Name: Wong Man Cong

Special Directions (if any):
If you're rebuilding the solution in release mode, do the following before building it.

Setting up of PGO:
1) Go to project settings
2) Configuration Properties -> Advanced -> Whole Program Optimization: Select "Use Link Time Code Generation"
3) C/C++ -> Linker -> Optimization -> Link Time Code Generation: Select "Use Link Time Code Generation (/LTCG)
4) Run the program. This will generate a .pgd file. Click on "Run Speed Test" only for pure speed.

Follow the steps below only after a .pgd file is generated
Reusing PGO:
1) Configuration Properties -> Advance -> Whole Program Optimization: Select "Profile Guided Optimization - Optimize"
2) Linker -> Optimization -> Link Time Code Generation: Select "Profile Guided Optimization - Optimization (LTCG:PGOptimize)
3) Run the program. Click on Speed test


Missing features (if any):

My experience working on this project:
Not very familiar with how A* algorithm works.

Hours spent:
15-25 hours

Extra credits: