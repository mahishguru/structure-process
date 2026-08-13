"""ICME-Mg: inverse composition-process linkages for extruded Mg alloys."""

__version__ = "0.1.0"

# Canonical autoregressive token order (see strategy/01, section 1.1).
# Seven informative elements; Cu/Ni/Si/Fe/Pr are ppm impurities, Y is a
# single-alloy marker in ME21 and redundant with Ce+Nd+Mn for identification.
ELEMENTS = ["Al", "Zn", "Mn", "Ce", "Gd", "Ca", "Nd"]
PROCESS_PARAMS = ["T_ext", "v_ext"]
LABEL_ORDER = ELEMENTS + PROCESS_PARAMS
