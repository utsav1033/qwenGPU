"""PTX reference interpreter (qgpu.sim) vs the lowered machine blocks (kernel.run_blocks) on
tiny/repro/argmax.cu. Run: uv run python tiny/repro/argmax.py"""
import subprocess, sys
from pathlib import Path
from qgpu import ptx, sim
from qgpu.circuit import kernel
from qgpu.lower import lower
from qgpu.merge import STAGES

here = Path(__file__).resolve().parent
subprocess.run(["/opt/homebrew/opt/llvm/bin/clang", "-x", "cuda", "--cuda-device-only", "-nocudainc",
                "-nocudalib", "--cuda-gpu-arch=sm_70", "-Xclang", "-target-feature", "-Xclang", "+ptx70",
                "-O2", "-S", str(here / "argmax.cu"), "-o", str(here / "argmax.ptx")], check=True)
k = ptx.load(here / "argmax.ptx")["argmax"]
a = [3, 1, 4, 1, 5, 9, 2, 6]
mem = sim.Memory(1 << 16, sim.GLOBAL_BASE)
pa, po = mem.alloc(4 * len(a)), mem.alloc(4)
mem.write_i32(pa, a)
sim.launch(k, mem, (1, 1, 1), (4, 1, 1), [pa, po, len(a)])
mul, logic, atom = STAGES[4]
lw = lower(k, (1,), (4,), [0, 1, len(a)], {0: a, 1: [0]}, lanes=4, stages=4, mul_stages=mul,
           logic_stages=logic, atom_stages=atom, write_through=True)
tr = kernel.run_blocks(lw.blocks, lw.mem, lw.warps, lanes=4)
print(f"a = {a}: PTX interpreter argmax = {mem.read_i32(po, 1)[0]}, "
      f"lowered (run_blocks) = {tr.mem.get(lw.buffers[1], 0)}")
