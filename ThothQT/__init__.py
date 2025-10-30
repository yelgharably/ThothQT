"""
ThothQT - Quantum Transport Library v1.0.0
===========================================

A high-performance Python library for quantum transport calculations using 
Non-Equilibrium Green's Functions (NEGF).

Simple Usage:
-------------
# Import modules directly (recommended):
from thothqt import NEGFEngine, make_1d_chain
from builders import GrapheneBuilder
from utils import temperature_to_thermal_energy

# Or use package interface (if working):
import thothqt as tqt
device = tqt.make_1d_chain(10, 1.0)
engine = tqt.NEGFEngine(device, Temp=300)

Author: ThothQT Development Team
License: MIT
"""

__version__ = "1.0.0"

# Try package-relative imports first, then direct imports
try:
    from .thothqt import (
    NEGFEngine,
    PeriodicLead,
    Device,
    SanchoRubioDecimator,
    GrapheneBuilder,
    make_1d_chain,
    make_graphene_nanoribbon,
    fermi_dirac,
    info
)
    _core_loaded = True
except Exception:
    try:
        # Fallback to direct import (if running from ThothQT directory)
        from thothqt import Device, PeriodicLead, NEGFEngine, make_1d_chain, fermi_dirac, info, KB_EV
        _core_loaded = True
    except Exception as e:
        print(f"Core import issue: {e}")
        _core_loaded = False

try:
    from .builders import GrapheneBuilder, TMDBuilder, CustomSystemBuilder, make_quantum_dot
    _builders_loaded = True
except Exception:
    try:
        from builders import GrapheneBuilder, TMDBuilder, CustomSystemBuilder, make_quantum_dot
        _builders_loaded = True
    except Exception as e:
        print(f"Builders import issue: {e}")
        _builders_loaded = False

try:
    from .utils import temperature_to_thermal_energy, quantum_of_conductance, EnergyMesh
    _utils_loaded = True
except Exception:
    try:
        from utils import temperature_to_thermal_energy, quantum_of_conductance, EnergyMesh
        _utils_loaded = True
    except Exception as e:
        print(f"Utils import issue: {e}")
        _utils_loaded = False

def status():
    """Show ThothQT loading status"""
    print("ThothQT Loading Status:")
    print(f"  Core: {'✓' if _core_loaded else '❌'}")
    print(f"  Builders: {'✓' if _builders_loaded else '❌'}")
    print(f"  Utils: {'✓' if _utils_loaded else '❌'}")
    
    if _core_loaded:
        print("\n✓ Ready for quantum transport!")
        print("Example: device = make_1d_chain(10, 1.0)")
    
    if not all([_core_loaded, _builders_loaded, _utils_loaded]):
        print("\nFor guaranteed access, import directly:")
        print("  from thothqt import NEGFEngine, make_1d_chain")
        print("  from builders import GrapheneBuilder") 
        print("  from utils import temperature_to_thermal_energy")

# Show status on import
loaded_count = sum([_core_loaded, _builders_loaded, _utils_loaded])
print(f"ThothQT v{__version__}: {loaded_count}/3 modules loaded")
if loaded_count == 3:
    print("✓ All modules available!")
elif _core_loaded:
    print("✓ Core available - use direct imports for other modules")
else:
    print("❌ Import issues - use direct imports")

# Define what's available
__all__ = []
if _core_loaded:
    __all__.extend(['Device', 'PeriodicLead', 'NEGFEngine', 'make_1d_chain', 'fermi_dirac', 'info', 'KB_EV'])
if _builders_loaded:
    __all__.extend(['GrapheneBuilder', 'TMDBuilder', 'CustomSystemBuilder', 'make_quantum_dot'])
if _utils_loaded:
    __all__.extend(['temperature_to_thermal_energy', 'quantum_of_conductance', 'EnergyMesh'])
__all__.append('status')