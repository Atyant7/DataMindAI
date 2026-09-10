# Navigation constants for DataMind AI UI overhaul

# Primary sections displayed in the left sidebar
PRIMARY_SECTIONS = [
    "Dashboard",
    "Projects",
    "Chat",
    "Workspace",
    "Predictions",
    "Settings",
]

# Sub‑sections that belong under the "Workspace" umbrella
WORKSPACE_SUBSECTIONS = [
    "Dataset",
    "Visualization",
    "Preprocessing",
    "Models",
    "Prediction",
]

# Mapping of section names to icon identifiers defined in `components._ICON_SVGS`
SECTION_ICONS = {
    "Dashboard": "dashboard",
    "Projects": "project",
    "Chat": "chat",
    "Workspace": "workspace",
    "Predictions": "prediction",
    "Settings": "settings",
    # Workspace sub‑sections can reuse the generic "workspace" icon or a more specific one if added later
    "Dataset": "workspace",
    "Visualization": "workspace",
    "Preprocessing": "workspace",
    "Models": "workspace",
    "Prediction": "prediction",
}
