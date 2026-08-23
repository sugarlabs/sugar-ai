# Copyright (C) 2026 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Sugar's activity-bundle-ID-to-category table.

Host-specific data: the engine only ever sees the abstract category, so a
non-Sugar host doesn't inherit this map along with it.
"""

_CATEGORY_ACTIVITIES = {
    "creative": [
        "org.laptop.TurtleArtActivity",
        "org.laptop.Oficina",
        "org.laptop.AbiWordActivity",
        "org.laptop.Record",
        "org.sugarlabs.MusicBlocksActivity",
    ],
    "programming": [
        "org.laptop.PippyActivity",
        "org.laptop.Pippy",
        "org.laptop.Calculate",
        "org.laptop.Terminal",
        "org.laptop.Physics",
        "org.laptop.Measure",
    ],
    "exploration": [
        "org.laptop.WebActivity",
        "org.laptop.Log",
        "org.laptop.ImageViewerActivity",
    ],
    "game": [
        "org.laptop.Memorize",
        "org.sugarlabs.Maze",
        "org.sugarlabs.Clock",
        "org.sugarlabs.Abacus",
    ],
    "communication": [
        "org.laptop.Chat",
        "org.laptop.Speak",
    ],
}

DEFAULT_CATEGORY = "creative"

_ACTIVITY_TO_CATEGORY = {
    bundle_id: category
    for category, bundle_ids in _CATEGORY_ACTIVITIES.items()
    for bundle_id in bundle_ids
}


def category_for_activity(activity_id: str) -> str:
    return _ACTIVITY_TO_CATEGORY.get(activity_id, DEFAULT_CATEGORY)
