# Copyright (C) 2026, Adarsh Kumar <adarsh23072005@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
AI-powered Sugar Activity Generator.

Kids or teachers describe an Activity in plain language and the AI
generates a working Sugar Activity project with all required files.

Example:
    "Make a drawing app where I can pick colors and draw with my finger"
    -> Generates: activity.info, setup.py, activity icon, main Python file
"""

import os
import json
import logging
import zipfile
import tempfile
from typing import Optional, Dict

logger = logging.getLogger("activity-generator")

ACTIVITY_INFO_TEMPLATE = """[Activity]
name = {name}
activity_version = 1
bundle_id = org.sugarlabs.{bundle_id}
exec = sugar-activity3 activity.{class_name}
icon = activity-icon
license = GPLv3+
"""

SETUP_PY = """from sugar3.activity import bundlebuilder
bundlebuilder.start()
"""

ICON_SVG_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [
  <!ENTITY stroke_color "#010101">
  <!ENTITY fill_color "#FFFFFF">
]>
<svg xmlns="http://www.w3.org/2000/svg" width="55" height="55">
  <rect x="5" y="5" width="45" height="45" rx="8" ry="8"
    style="fill:&fill_color;;stroke:&stroke_color;;stroke-width:3"/>
  <text x="27.5" y="35" text-anchor="middle"
    style="font-size:20px;fill:&stroke_color;;">{icon_letter}</text>
</svg>
"""

ACTIVITY_GEN_PROMPT = """You are a Sugar Activity code generator.
Given a description, generate a complete working Python file for
a Sugar Activity.

Rules:
1. Class must inherit from sugar3.activity.activity.Activity
2. Use sugar3.graphics widgets (ToolbarBox, StopButton, etc.)
3. Use GTK3 widgets (Gtk.DrawingArea, Gtk.Label, Gtk.Button)
4. Include toolbar with ActivityToolbarButton and StopButton
5. Keep code simple enough for a kid to understand
6. Add comments explaining each section
7. Use self.set_canvas() and self.show_all()
8. Include write_file() and read_file() for Journal integration
9. Do NOT use external libraries beyond sugar3 and GTK3
10. Output ONLY Python code, no explanations or markdown

Description: {description}
"""


def sanitize_name(name: str) -> str:
    """Convert a human-readable name to a valid Python identifier."""
    clean = "".join(c if c.isalnum() else "_" for c in name)
    clean = clean.strip("_")
    if clean and clean[0].isdigit():
        clean = "My" + clean
    return clean or "MyActivity"


def generate_activity_files(
    name: str,
    description: str,
    generated_code: str
) -> Dict[str, str]:
    """Create all files needed for a Sugar Activity bundle.

    Args:
        name: Human-readable Activity name
        description: What the Activity does
        generated_code: AI-generated Python code

    Returns:
        Dictionary mapping file paths to their contents.
    """
    class_name = sanitize_name(name) + "Activity"
    bundle_id = sanitize_name(name).lower()
    icon_letter = name[0].upper() if name else "A"

    files = {
        "activity/activity.info": ACTIVITY_INFO_TEMPLATE.format(
            name=name,
            bundle_id=bundle_id,
            class_name=class_name
        ),
        "setup.py": SETUP_PY,
        "activity/activity-icon.svg": ICON_SVG_TEMPLATE.format(
            icon_letter=icon_letter
        ),
        "activity.py": generated_code,
    }
    return files


def bundle_activity(
    name: str,
    files: Dict[str, str],
    output_dir: Optional[str] = None
) -> str:
    """Package Activity files into an installable .xo bundle.

    Args:
        name: Activity name for the bundle filename.
        files: Dictionary of filepath to content.
        output_dir: Directory for output .xo file.

    Returns:
        Path to the generated .xo bundle file.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    bundle_name = sanitize_name(name)
    xo_path = os.path.join(output_dir, f"{bundle_name}-1.xo")

    with zipfile.ZipFile(xo_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filepath, content in files.items():
            arcname = f"{bundle_name}.activity/{filepath}"
            zf.writestr(arcname, content)

    logger.info("Created bundle: %s", xo_path)
    return xo_path


def get_generation_prompt(description: str) -> str:
    """Build the prompt for the LLM to generate Activity code."""
    return ACTIVITY_GEN_PROMPT.format(description=description)
