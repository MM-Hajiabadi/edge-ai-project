SIM ?= verilator
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES += $(PWD)/conv3x3_16f.v

TOPLEVEL = conv3x3_16f
MODULE = test_16f

include $(shell cocotb-config --makefiles)/Makefile.sim
