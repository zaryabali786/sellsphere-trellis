"""
Headless Blender Script to Convert GLB to Apple USDZ
Preserves all PBR materials, base color textures, roughness, and metallic channels.
"""
import os
import sys

try:
    import bpy

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if len(argv) < 2:
        print("Usage: blender -b --python convert_glb_to_usdz.py -- <input.glb> <output.usdz>")
        sys.exit(1)

    glb_file = os.path.abspath(argv[0])
    usdz_file = os.path.abspath(argv[1])

    if not os.path.exists(glb_file):
        print(f"Error: Input GLB file does not exist: {glb_file}")
        sys.exit(1)

    # Clear current scene objects and meshes
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import GLB
    print(f"Importing GLB from: {glb_file}")
    bpy.ops.import_scene.gltf(filepath=glb_file)

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(usdz_file), exist_ok=True)

    # Export USDZ with embedded materials and textures
    print(f"Exporting USDZ to: {usdz_file}")
    bpy.ops.wm.usd_export(
        filepath=usdz_file,
        export_materials=True,
        export_textures=True,
        relative_paths=True
    )

    if os.path.exists(usdz_file) and os.path.getsize(usdz_file) > 100:
        print(f"USDZ export success: {os.path.getsize(usdz_file)} bytes")
        sys.exit(0)
    else:
        print("Error: USDZ file not created or empty.")
        sys.exit(1)

except Exception as err:
    print(f"Blender USDZ conversion error: {err}")
    sys.exit(1)
