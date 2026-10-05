"""ICME-Mg: inverse composition-process linkages for extruded Mg alloys."""

__version__ = "0.1.0"

# Canonical autoregressive token order (see strategy/01, section 1.1).
# Eight elements; Cu/Ni/Si/Fe/Pr are ppm impurities and stay out.
ELEMENTS = ["Al", "Zn", "Mn", "Ce", "Gd", "Ca", "Nd", "Y"]
PROCESS_PARAMS = ["T_ext", "v_ext"]
LABEL_ORDER = ELEMENTS + PROCESS_PARAMS
