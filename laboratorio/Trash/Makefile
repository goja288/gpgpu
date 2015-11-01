#---------------------------------
# 
#---------------------------------
CIMG_VERSION = 1.6.6_pre093015
X11PATH = /usr/X11R6
CC = g++
NVCC = nvcc
ARCH = -arch=compute_20 -code=sm_20
#WARNING		 = -I.. -Wall -W -std=c++11 -ansi -pedantic -Dcimg_use_vt100 -Dcimg_display=1 -I/usr/X11R6/include  -lm -L/usr/X11R6/lib -lpthread -lX11 
WARNING		 = -I.. -std=c++11 -Dcimg_use_vt100 -Dcimg_display=1 -I/usr/X11R6/include  -lm -L/usr/X11R6/lib64 -lpthread -lX11 

# No muestra los warning no me gusta, pero la libreria salta por todos lados
FLAGS_NVCC = -w

# ifeq ($(MSYSTEM),MINGW32)
# EXESFX       = .exe
# else
# EXESFX       =
# endif
# CCVER       = $(CC)
# ifeq ($(notdir $(CC)),g++)
# CCVER        = `$(CC) -v 2>&1 | tail -n 1`
# endif
# ifeq ($(notdir $(CC)),clang++)
# CCVER        = `$(CC) -v 2>&1 | head -n 1`
# endif
# ifeq ($(notdir $(CC)),icpc)
# CCVER        = "icpc \( `$(CC) -v 2>&1`\)"
# CFLAGS       = -I..
# LIBS         =
# else
# CFLAGS       = -I.. -Wall -W
# LIBS         = -lm
# endif


all: voronoi

#cimg.o:
#	$(CC) -c main.cu $(WARNING)

# main: cimg.o

# 	$(NVCC) main.cu -o main -arch=compute_20 -code=sm_20

#main: cimg.o
voronoi: 

	$(NVCC) voronoi.cu -o voronoi $(FLAGS_NVCC) $(ARCH) $(WARNING)


clean:
	rm -rf *.o voronoi
