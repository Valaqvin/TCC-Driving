mods = [
    "carla",
    "numpy",
    "networkx",
    "six",
    "yaml",
    "torch",
    "cv2",
    "pygame",
    "leaderboard",
    "srunner",
]

for mod in mods:
    try:
        imported = __import__(mod)
        path = getattr(imported, "__file__", "")
        print(f"OK {mod} {path}")
    except Exception as exc:
        print(f"NO {mod} {type(exc).__name__}: {exc}")
