# Optional: print more diagnostics for debugging
# from sapien import disable_renderer
# disable_renderer()  # <-- Uncomment to skip the renderer


import os
import logging
import subprocess
import json
import sys
from pathlib import Path

# Prefer a headless-safe NVIDIA Vulkan ICD in containers.
if "VK_ICD_FILENAMES" not in os.environ:
    nvidia_icd = Path("/etc/vulkan/icd.d/nvidia_icd.json")
    if nvidia_icd.exists():
        egl_icd = Path("/tmp/nvidia_egl_icd.json")
        egl_icd.write_text(
            json.dumps(
                {
                    "file_format_version": "1.0.1",
                    "ICD": {
                        "library_path": "libEGL_nvidia.so.0",
                        "api_version": "1.3.0",
                    },
                }
            )
        )
        os.environ["VK_ICD_FILENAMES"] = str(egl_icd)
os.environ.setdefault("DISPLAY", "")

import mani_skill2_real2sim
from simpler_env.utils.env.env_builder import build_maniskill2_env

logging.basicConfig(level=logging.DEBUG)

env_name = "PutEggplantInBasketScene-v0"

ms2r2s_root = Path(mani_skill2_real2sim.__file__).resolve().parents[1]
rgb_overlay_path = ms2r2s_root / "data" / "real_inpainting" / "bridge_sink.png"

kwargs = {
    "obs_mode": "rgbd",
    "robot": "widowx_sink_camera_setup",
    "sim_freq": 500,
    "control_mode": "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
    "control_freq": 5,
    "max_episode_steps": 120,
    "scene_name": "bridge_table_1_v2",
    "camera_cfgs": {"add_segmentation": True},
    "rgb_overlay_path": str(rgb_overlay_path),
    "renderer_kwargs": {"offscreen_only": True},
}

additional_env_build_kwargs = {}

def _has_working_vulkan() -> bool:
    probe = (
        "import os; "
        "from pathlib import Path; "
        "os.environ.setdefault('DISPLAY',''); "
        "p=Path('/etc/vulkan/icd.d/nvidia_icd.json'); "
        "import mani_skill2_real2sim; "
        "from simpler_env.utils.env.env_builder import build_maniskill2_env; "
        "root=Path(mani_skill2_real2sim.__file__).resolve().parents[1]; "
        "kwargs={"
        "'obs_mode':'rgbd',"
        "'robot':'widowx_sink_camera_setup',"
        "'sim_freq':500,"
        "'control_mode':'arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos',"
        "'control_freq':5,"
        "'max_episode_steps':120,"
        "'scene_name':'bridge_table_1_v2',"
        "'camera_cfgs':{'add_segmentation':True},"
        "'rgb_overlay_path':str(root/'data'/'real_inpainting'/'bridge_sink.png'),"
        "'renderer_kwargs':{'offscreen_only':True}"
        "}; "
        "build_maniskill2_env('PutEggplantInBasketScene-v0', **kwargs)"
    )
    env = os.environ.copy()
    if "VK_ICD_FILENAMES" not in env and Path("/tmp/nvidia_egl_icd.json").exists():
        env["VK_ICD_FILENAMES"] = "/tmp/nvidia_egl_icd.json"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        print("Renderer preflight failed.")
        if result.stderr:
            print(result.stderr.strip().splitlines()[-1])
    return result.returncode == 0


if not _has_working_vulkan():
    raise RuntimeError(
        "No working Vulkan renderer in current environment. "
        "This script requires SAPIEN Vulkan rendering for obs_mode='rgbd'. "
        "Please run on a machine/container with compatible Vulkan driver stack."
    )

print("🔧 Start building ManiSkill2 env...")
env = build_maniskill2_env(
    env_name,
    **additional_env_build_kwargs,
    **kwargs,
)
print("✅ Env built successfully:", env)

obs = env.reset()
print("📷 First observation keys:", obs.keys() if isinstance(obs, dict) else type(obs))
