CXX = nvcc
CXXFLAGS = --default-stream per-thread -O3 -arch=sm_75 -lineinfo -Xcompiler -Icutlass -Iinclude -DCUTLASS_USE_INT_WMMA -DCUTLASS_USE_SUBBYTE_WMMA -DBLOCK_SIZE=256 -DNUM_STREAMS=8 -g
EXE_NAME = tensor-episdet
SOURCES = src/helper.cu src/tensor-episdet.cu
BINDIR = bin


triplets_k2:
	[ -d $(BINDIR) ] || mkdir $(BINDIR)
	$(CXX) $(SOURCES) src/search-triplets.cu $(CXXFLAGS) -DTRIPLETS -o $(BINDIR)/$(EXE_NAME).triplets.k2.bin

triplets_mi:
	[ -d $(BINDIR) ] || mkdir $(BINDIR)
	$(CXX) $(SOURCES) src/search-triplets.cu $(CXXFLAGS) -DTRIPLETS -DMI_SCORE -o $(BINDIR)/$(EXE_NAME).triplets.mi.bin

pairs_k2:
	[ -d $(BINDIR) ] || mkdir $(BINDIR)
	$(CXX) $(SOURCES) src/search-pairs.cu $(CXXFLAGS) -o $(BINDIR)/$(EXE_NAME).pairs.k2.bin

pairs_mi:
	[ -d $(BINDIR) ] || mkdir $(BINDIR)
	 $(CXX) $(SOURCES) src/search-pairs.cu $(CXXFLAGS) -DMI_SCORE -o $(BINDIR)/$(EXE_NAME).pairs.mi.bin

clean: 
	rm -rf $(BINDIR)/$(EXE_NAME)

